"""Confidential image analysis gRPC server for Google Cloud Confidential Space.

Two services, chosen by the client per image: FACE_RECOGNITION (InsightFace
buffalo_l) and IMAGE_TAGGING (SigLIP ViT-B/16). See faces.py and tagger.py."""

import asyncio
import hashlib
import logging
import os
import signal
import ssl
import string
import time
from collections import defaultdict, deque

import grpc

import classifier_pb2 as pb2
import classifier_pb2_grpc as pb2_grpc
from attestation import fetch_attestation_token
from inference import InferenceEngine
from tls_util import generate_self_signed_cert

PORT = int(os.environ.get("PORT", "50051"))
AUDIENCE = os.environ.get("AUDIENCE", "image-classifier-clients")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
MAX_INFLIGHT_PER_STREAM = int(os.environ.get("MAX_INFLIGHT_PER_STREAM", "32"))
MAX_CONCURRENT_RPCS = int(os.environ.get("MAX_CONCURRENT_RPCS", "256"))
DEFAULT_TOP_K, MAX_TOP_K = 5, 50
MAX_FACES = 50

# Mutual TLS: the client CA certificate is baked into the image (covered by the
# attested digest), so clients can see exactly which CA gates access.
CLIENT_CA_FILE = os.environ.get("CLIENT_CA_FILE", "/app/client-ca.pem")
# Comma-separated SHA-256 fingerprints of revoked client certificates. Set via
# tee-env-REVOKED_CLIENT_CERTS metadata; takes effect on VM reset.
REVOKED_CLIENT_CERTS = frozenset(
    f.strip().lower() for f in os.environ.get("REVOKED_CLIENT_CERTS", "").split(",") if f.strip()
)
ATTEST_PER_MINUTE = int(os.environ.get("ATTEST_PER_MINUTE", "30"))

# Never log image contents, faces, embeddings or tags: Cloud Logging is outside the TEE.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("classifier")


def client_identity(context):
    """Returns (common_name, cert_sha256) of the verified client certificate.

    gRPC has already validated the chain against the client CA during the TLS
    handshake (require_client_auth=True); this just reads the result.
    """
    auth = context.auth_context()
    pem = (auth.get("x509_pem_cert") or [b""])[0]
    if not pem:
        return None
    fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem.decode())).hexdigest()
    cn = (auth.get("x509_common_name") or [b"unknown"])[0].decode(errors="replace")
    return cn, fingerprint


class RequestError(ValueError):
    """A problem with the request itself; its message is safe to return."""


def face_result(faces, width: int, height: int) -> pb2.FaceResult:
    return pb2.FaceResult(image_width=width, image_height=height, faces=[
        pb2.Face(
            box=pb2.BoundingBox(x1=float(f.box[0]), y1=float(f.box[1]), x2=float(f.box[2]), y2=float(f.box[3])),
            score=f.score,
            landmarks=[pb2.Point(x=float(x), y=float(y)) for x, y in f.landmarks],
            embedding=f.embedding.tolist(),
        ) for f in faces])


class AnalyzerService(pb2_grpc.ImageAnalyzerServicer):
    def __init__(self, engine: InferenceEngine, cert_sha256: str, token_fetcher=fetch_attestation_token,
                 revoked=REVOKED_CLIENT_CERTS, attest_per_minute=ATTEST_PER_MINUTE):
        self.engine = engine
        self.cert_sha256 = cert_sha256
        self.token_fetcher = token_fetcher
        self.revoked = revoked
        self.attest_per_minute = attest_per_minute
        self._attest_calls = defaultdict(deque)  # client fingerprint -> call timestamps

    async def _authorize(self, context):
        ident = client_identity(context)
        if ident is None:  # unreachable with require_client_auth=True; defence in depth
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "client certificate required")
        if ident[1] in self.revoked:
            log.warning("rejected revoked client cert cn=%s sha256=%s", *ident)
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "client certificate revoked")
        return ident

    async def Attest(self, request, context):
        cn, fingerprint = await self._authorize(context)
        calls, now = self._attest_calls[fingerprint], time.monotonic()
        while calls and now - calls[0] > 60:
            calls.popleft()
        if len(calls) >= self.attest_per_minute:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "attestation rate limit exceeded")
        calls.append(now)
        nonce = request.nonce
        if not (32 <= len(nonce) <= 64 and all(c in string.hexdigits for c in nonce)):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "nonce must be 32-64 hex characters")
        try:
            token = await asyncio.to_thread(self.token_fetcher, AUDIENCE, [nonce.lower(), self.cert_sha256])
        except Exception:
            log.exception("attestation token request failed")
            await context.abort(grpc.StatusCode.UNAVAILABLE, "attestation service unavailable")
        return pb2.AttestResponse(token=token, cert_sha256=self.cert_sha256)

    async def _analyze(self, req) -> pb2.ImageResult:
        result = pb2.ImageResult(request_id=req.request_id, service=req.service)
        if not req.image or len(req.image) > MAX_IMAGE_BYTES:
            raise RequestError("image missing or too large")
        if req.service == pb2.FACE_RECOGNITION:
            max_faces = min(req.max_faces, MAX_FACES) if req.max_faces > 0 else MAX_FACES
            faces, width, height = await self.engine.recognise_faces(req.image, max_faces)
            result.faces.CopyFrom(face_result(faces, width, height))
        elif req.service == pb2.IMAGE_TAGGING:
            top_k = min(max(req.top_k or DEFAULT_TOP_K, 1), MAX_TOP_K)
            min_score = req.min_score if req.HasField("min_score") else None
            tags = await self.engine.tag_image(req.image, top_k, min_score)
            result.tags.CopyFrom(pb2.TagResult(tags=[pb2.Tag(name=n, score=s) for n, s in tags]))
        else:
            raise RequestError("service must be FACE_RECOGNITION or IMAGE_TAGGING")
        return result

    async def AnalyzeStream(self, request_iterator, context):
        cn, _ = await self._authorize(context)
        log.info("stream opened by client cn=%s", cn)
        out: asyncio.Queue = asyncio.Queue()
        slots = asyncio.Semaphore(MAX_INFLIGHT_PER_STREAM)  # per-client backpressure
        pending: set[asyncio.Task] = set()

        async def handle(req):
            try:
                pred = await self._analyze(req)
            except asyncio.CancelledError:
                raise
            except RequestError as exc:
                pred = pb2.ImageResult(request_id=req.request_id, service=req.service, error=str(exc))
            except Exception as exc:
                pred = pb2.ImageResult(request_id=req.request_id, service=req.service,
                                       error=f"analysis failed: {type(exc).__name__}")
            finally:
                slots.release()
            await out.put(pred)

        async def reader():
            try:
                async for req in request_iterator:
                    await slots.acquire()
                    task = asyncio.create_task(handle(req))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                if pending:
                    await asyncio.gather(*list(pending), return_exceptions=True)
            finally:
                await out.put(None)  # sentinel: no more predictions

        reader_task = asyncio.create_task(reader())
        try:
            while (pred := await out.get()) is not None:
                yield pred
        finally:
            reader_task.cancel()
            for task in list(pending):
                task.cancel()


async def serve(engine=None, token_fetcher=fetch_attestation_token, port: int = PORT,
                client_ca_file: str = CLIENT_CA_FILE):
    with open(client_ca_file, "rb") as f:
        client_ca_pem = f.read()
    key_pem, cert_pem, cert_sha256 = generate_self_signed_cert()
    log.info("TLS certificate generated in TEE, sha256=%s", cert_sha256)

    engine = engine or InferenceEngine.load()

    server = grpc.aio.server(
        maximum_concurrent_rpcs=MAX_CONCURRENT_RPCS,
        options=[
            ("grpc.max_receive_message_length", MAX_IMAGE_BYTES + 1024),
            ("grpc.keepalive_time_ms", 60_000),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )
    pb2_grpc.add_ImageAnalyzerServicer_to_server(AnalyzerService(engine, cert_sha256, token_fetcher), server)
    credentials = grpc.ssl_server_credentials(
        [(key_pem, cert_pem)],
        root_certificates=client_ca_pem,  # only certs issued by this CA are accepted
        require_client_auth=True,         # handshake fails without a valid client cert
    )
    server.add_secure_port(f"0.0.0.0:{port}", credentials)
    await server.start()
    log.info("listening on :%d (mTLS, audience=%s, %d revoked client certs)", port, AUDIENCE,
             len(REVOKED_CLIENT_CERTS))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    log.info("shutting down")
    await server.stop(grace=10)
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(serve())
