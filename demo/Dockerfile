# For production, pin the base image by digest so builds are reproducible
# and the attested image digest is meaningful, e.g.:
#   FROM python:3.12-slim@sha256:<digest>

# ---------------------------------------------------------------------------
# Stage 1: fetch and verify the models, embed the tag vocabulary.
# Every download is checked against the SHA-256 pinned in app/models.py.
# SigLIP's text model is only needed here, so it never reaches the final image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS models

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /build
COPY requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip==26.2.2 \
 && pip install --no-cache-dir -r requirements-server.txt

COPY app/models.py app/faces.py app/tagger.py app/prepare_models.py app/tags.txt ./
RUN python prepare_models.py fetch --models /models \
 && python prepare_models.py embed-tags --models /models --tags tags.txt

# ---------------------------------------------------------------------------
# Stage 2: the runtime image (this is what gets attested).
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir --upgrade pip==26.2.2 \
 && pip install --no-cache-dir -r requirements-server.txt

# Only the runtime models: face detector + recogniser, SigLIP's image half,
# and the precomputed tag vectors. The server re-verifies their digests at start.
COPY --from=models /models/det_10g.onnx /models/w600k_r50.onnx /models/tags.npz /models/
COPY --from=models /models/siglip-b16-webli/visual.onnx /models/siglip-b16-webli/visual.onnx

# Build context must be the project root with this layout (see cs-classifier.zip):
#   Dockerfile  cloudbuild.yaml  deploy_server.sh  requirements-server.txt
#   proto/classifier.proto
#   app/server.py  app/inference.py  app/faces.py  app/tagger.py  app/models.py
#   app/attestation.py  app/tls_util.py  app/prepare_models.py  app/tags.txt
#   client-ca.pem   (public client CA cert, from tools/client_ca.py init-ca)
# Files are copied explicitly so a missing one fails with a clear message.
COPY proto/classifier.proto proto/classifier.proto
COPY app/server.py app/inference.py app/faces.py app/tagger.py app/models.py \
     app/attestation.py app/tls_util.py ./
RUN python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/classifier.proto

# Mutual TLS trust anchor. Baked in (not passed via metadata) so it is covered
# by the attested image digest: the operator cannot swap in their own CA, and
# clients can verify exactly which CA gates access.
COPY client-ca.pem /app/client-ca.pem

# Confidential Space launch policy (enforced by the launcher, covered by the digest):
# only these env vars may be overridden by the operator via tee-env-* metadata,
# and container stdout/stderr may be sent to Cloud Logging.
LABEL "tee.launch_policy.allow_env_override"="AUDIENCE,MAX_INFLIGHT_PER_STREAM,INFERENCE_WORKERS,REVOKED_CLIENT_CERTS,ATTEST_PER_MINUTE"
LABEL "tee.launch_policy.log_redirect"="always"

EXPOSE 50051
CMD ["python", "server.py"]
