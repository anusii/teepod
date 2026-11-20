"""Client for the confidential image classifier.

Flow:
  1. Fetch the server's self-signed TLS certificate and compute its SHA-256.
     The client presents its own certificate (mutual TLS); the server rejects
     the handshake unless it was issued by the client CA baked into the image.
  2. Open a gRPC channel that trusts ONLY that certificate, again with the client cert.
  3. Call Attest() with a fresh random nonce.
  4. Verify the JWT: Google signature, issuer, audience, both nonces (ours + cert hash),
     production Confidential Space image, debugging disabled, and the exact
     container image digest we expect.
  5. Only then stream images over the same pinned channel, each naming the
     service to run: FACE_RECOGNITION or IMAGE_TAGGING.

Usage:
  # Image tagging
  python client.py --service tags --host 34.1.2.3 --images ./photos \
      --client-cert ~/classifier-client/client.pem --client-key ~/classifier-client/client-key.pem \
      --expected-image-digest sha256:abcd... --audience image-classifier-clients

  # Face recognition; with --gallery, faces are named by comparing them with
  # reference photos in GALLERY/<person name>/*.jpg (matching happens here, on
  # the client: the server keeps no faces or identities).
  python client.py --service faces --gallery ./people --host 34.1.2.3 --images ./photos \
      --client-cert ~/classifier-client/client.pem --client-key ~/classifier-client/client-key.pem \
      --expected-image-digest sha256:abcd... --audience image-classifier-clients
"""

import argparse
import json
import hashlib
import pathlib
import secrets
import socket
import ssl
import sys
import uuid

import grpc
import jwt
import requests

import classifier_pb2 as pb2
import classifier_pb2_grpc as pb2_grpc

ISSUER = "https://confidentialcomputing.googleapis.com"
TLS_SERVER_NAME = "classifier.confidential-space.internal"
ALLOWED_HW = {"GCP_AMD_SEV", "GCP_AMD_SEV_ES", "GCP_INTEL_TDX"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class AttestationError(Exception):
    pass


def fetch_server_cert(host: str, port: int, client_cert: str, client_key: str):
    """Reads the server's certificate, presenting our client certificate.

    The server cert is NOT verified here (it is self-signed inside the TEE);
    it is pinned and only trusted after attestation binds its hash to the TEE.
    Our client cert proves possession of the key without revealing it, so an
    impostor server learns nothing reusable.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.set_alpn_protocols(["h2"])
    ctx.load_cert_chain(client_cert, client_key)
    with socket.create_connection((host, port), timeout=10) as sock:
        with ctx.wrap_socket(sock, server_hostname=TLS_SERVER_NAME) as tls:
            der = tls.getpeercert(binary_form=True)
    return ssl.DER_cert_to_PEM_cert(der).encode(), hashlib.sha256(der).hexdigest()


def decode_and_verify_signature(token: str, audience: str) -> dict:
    """Checks the JWT signature against Google's published Confidential Space keys."""
    oidc = requests.get(f"{ISSUER}/.well-known/openid-configuration", timeout=10).json()
    signing_key = jwt.PyJWKClient(oidc["jwks_uri"]).get_signing_key_from_jwt(token)
    return jwt.decode(
        token, signing_key.key, algorithms=["RS256"],
        audience=audience, issuer=ISSUER, leeway=30,
    )


def check_claims(claims: dict, *, nonce: str, cert_sha256: str, expected_digest: str,
                 allow_debug: bool = False) -> None:
    """Workload-specific policy. Raises AttestationError on any mismatch."""
    nonces = claims.get("eat_nonce", [])
    nonces = [nonces] if isinstance(nonces, str) else nonces
    if nonce not in nonces:
        raise AttestationError("token does not contain our nonce (possible replay)")
    if cert_sha256 not in nonces:
        raise AttestationError("token is not bound to the TLS certificate we are connected to")
    if claims.get("swname") != "CONFIDENTIAL_SPACE":
        raise AttestationError("not running on a Confidential Space image")
    if claims.get("hwmodel") not in ALLOWED_HW:
        raise AttestationError(f"unexpected hardware model {claims.get('hwmodel')}")
    if not allow_debug:
        if claims.get("dbgstat") != "disabled-since-boot":
            raise AttestationError("workload is running on a debug image")
        support = claims.get("submods", {}).get("confidential_space", {}).get("support_attributes", [])
        if "STABLE" not in support:
            raise AttestationError("Confidential Space image is not a supported STABLE release")
    digest = claims.get("submods", {}).get("container", {}).get("image_digest")
    if digest != expected_digest:
        raise AttestationError(f"container digest {digest} != expected {expected_digest}")


SERVICES = {"faces": pb2.FACE_RECOGNITION, "tags": pb2.IMAGE_TAGGING}


def list_images(path: pathlib.Path):
    items = path.rglob("*") if path.is_dir() else [path]
    return sorted(p for p in items if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def analyze(stub, paths, args):
    """Streams every image to the server; yields (path, result) as results arrive."""
    by_id = {}

    def requests():
        for p in paths:
            rid = uuid.uuid4().hex
            by_id[rid] = p
            req = pb2.ImageRequest(request_id=rid, image=p.read_bytes(), service=SERVICES[args.service])
            if args.service == "faces":
                req.max_faces = args.max_faces
            else:
                req.top_k = args.top_k
                if args.min_score is not None:
                    req.min_score = args.min_score
            yield req

    for result in stub.AnalyzeStream(requests()):
        yield by_id.pop(result.request_id), result


def unit(v):
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def load_gallery(stub, root: pathlib.Path, args):
    """One reference vector per person: the average of the largest face in each
    of their photos (GALLERY/<person>/*.jpg)."""
    people = {d.name: list_images(d) for d in sorted(root.iterdir()) if d.is_dir()}
    owner = {p: name for name, ps in people.items() for p in ps}
    gallery_args = argparse.Namespace(**{**vars(args), "max_faces": 1})
    vectors = {name: [] for name in people}
    for path, result in analyze(stub, list(owner), gallery_args):
        if result.error or not result.faces.faces:
            print(f"gallery: no usable face in {path} ({result.error or 'no face found'})", file=sys.stderr)
            continue
        vectors[owner[path]].append(list(result.faces.faces[0].embedding))
    gallery = {name: unit([sum(xs) / len(vs) for xs in zip(*vs)]) for name, vs in vectors.items() if vs}
    print(f"gallery: {len(gallery)} people ({', '.join(gallery)})", file=sys.stderr)
    return gallery


def identify(embedding, gallery, threshold):
    if not gallery:
        return None, None
    name, sim = max(((n, dot(embedding, v)) for n, v in gallery.items()), key=lambda t: t[1])
    return (name if sim >= threshold else "unknown"), sim


def report(path, result, gallery, args):
    if args.json:
        record = {"image": str(path), "service": args.service, "error": result.error or None}
        if result.HasField("faces"):
            record["faces"] = []
            for f in result.faces.faces:
                face = {"box": [f.box.x1, f.box.y1, f.box.x2, f.box.y2], "score": f.score,
                        "landmarks": [[p.x, p.y] for p in f.landmarks], "embedding": list(f.embedding)}
                if gallery is not None:
                    face["identity"], face["similarity"] = identify(f.embedding, gallery, args.threshold)
                record["faces"].append(face)
        elif result.HasField("tags"):
            record["tags"] = [{"name": t.name, "score": t.score} for t in result.tags.tags]
        print(json.dumps(record))
        return
    if result.error:
        print(f"{path}: ERROR {result.error}")
    elif result.HasField("tags"):
        print(f"{path}: " + (", ".join(f"{t.name} ({t.score:.3f})" for t in result.tags.tags) or "no tags"))
    else:
        faces = result.faces.faces
        print(f"{path}: {len(faces)} face(s)")
        for i, f in enumerate(faces, 1):
            line = (f"  face {i}: box ({f.box.x1:.0f},{f.box.y1:.0f})-({f.box.x2:.0f},{f.box.y2:.0f}),"
                    f" detection {f.score:.2f}")
            if gallery is not None:
                name, sim = identify(f.embedding, gallery, args.threshold)
                line += f" -> {name}" + (f" (similarity {sim:.2f})" if sim is not None else "")
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", required=True, choices=sorted(SERVICES),
                    help="faces = face recognition, tags = image tagging")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--images", required=True, help="image file or directory (searched recursively)")
    ap.add_argument("--json", action="store_true", help="print one JSON object per image")
    tags = ap.add_argument_group("image tagging")
    tags.add_argument("--top-k", type=int, default=5)
    tags.add_argument("--min-score", type=float, help="drop tags with a lower cosine similarity")
    faces = ap.add_argument_group("face recognition")
    faces.add_argument("--max-faces", type=int, default=0, help="0 = all (server caps at 50)")
    faces.add_argument("--gallery", type=pathlib.Path, help="reference photos: GALLERY/<person>/*.jpg")
    faces.add_argument("--threshold", type=float, default=0.4,
                       help="minimum similarity to name a face (default 0.4)")
    ap.add_argument("--audience", default="image-classifier-clients")
    ap.add_argument("--client-cert", required=True, help="client certificate issued by the client CA")
    ap.add_argument("--client-key", required=True, help="client private key (never leaves this machine)")
    ap.add_argument("--expected-image-digest", help="sha256:... of the workload container")
    ap.add_argument("--allow-debug-image", action="store_true")
    ap.add_argument("--insecure-skip-attestation", action="store_true",
                    help="LOCAL TESTING ONLY: skip attestation (no confidentiality guarantee)")
    args = ap.parse_args()
    if args.gallery and args.service != "faces":
        ap.error("--gallery needs --service faces")

    paths = list_images(pathlib.Path(args.images))
    if not paths:
        sys.exit("no images found")

    cert_pem, cert_sha256 = fetch_server_cert(args.host, args.port, args.client_cert, args.client_key)
    credentials = grpc.ssl_channel_credentials(
        root_certificates=cert_pem,  # trust only the pinned TEE certificate
        private_key=pathlib.Path(args.client_key).read_bytes(),
        certificate_chain=pathlib.Path(args.client_cert).read_bytes(),
    )
    channel = grpc.secure_channel(
        f"{args.host}:{args.port}",
        credentials,
        options=[("grpc.ssl_target_name_override", TLS_SERVER_NAME),
                 ("grpc.max_send_message_length", 32 * 1024 * 1024),
                 ("grpc.max_receive_message_length", 32 * 1024 * 1024)],
    )
    stub = pb2_grpc.ImageAnalyzerStub(channel)

    if args.insecure_skip_attestation:
        print("WARNING: attestation skipped -- do not send sensitive images", file=sys.stderr)
    else:
        if not args.expected_image_digest:
            sys.exit("--expected-image-digest is required")
        nonce = secrets.token_hex(32)
        resp = stub.Attest(pb2.AttestRequest(nonce=nonce), timeout=30)
        claims = decode_and_verify_signature(resp.token, args.audience)
        check_claims(claims, nonce=nonce, cert_sha256=cert_sha256,
                     expected_digest=args.expected_image_digest, allow_debug=args.allow_debug_image)
        print(f"Attestation OK: {claims['hwmodel']}, image {claims['submods']['container']['image_digest']}",
              file=sys.stderr)

    gallery = load_gallery(stub, args.gallery, args) if args.gallery else None
    for path, result in analyze(stub, paths, args):
        report(path, result, gallery, args)
    channel.close()


if __name__ == "__main__":
    main()
