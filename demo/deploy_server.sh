#!/usr/bin/env bash
# Build, push and deploy the confidential image classifier to Confidential Space.
# Run from the repo root:  PROJECT_ID=my-proj CLIENT_CIDR=203.0.113.0/24 ./deploy_server.sh
set -euo pipefail

CLIENT_CIDR="0.0.0.0/0"
CS_IMAGE_FAMILY="confidential-space-debug"

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
CLIENT_CIDR="${CLIENT_CIDR:?set CLIENT_CIDR (who may reach port 50051)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-c}"
REPO="${REPO:-confidential-workloads}"
IMAGE_NAME="${IMAGE_NAME:-image-classifier}"
VM_NAME="${VM_NAME:-classifier-cs-1}"
MACHINE_TYPE="${MACHINE_TYPE:-n2d-standard-2}"       # AMD SEV; for TDX use c3-standard-4 + CC_TYPE=TDX
CC_TYPE="${CC_TYPE:-SEV}"
CS_IMAGE_FAMILY="${CS_IMAGE_FAMILY:-confidential-space}"  # confidential-space-debug while developing
AUDIENCE="${AUDIENCE:-image-classifier-clients}"
SA_NAME="${SA_NAME:-classifier-workload}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SA_NAME="${BUILD_SA_NAME:-classifier-builder}"
BUILD_SA_EMAIL="${BUILD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_BUCKET="${BUILD_BUCKET:-${PROJECT_ID}-classifier-build}"
NET_TAG="confidential-classifier"
# Comma-separated SHA-256 fingerprints of revoked client certs (printed by tools/client_ca.py sign)
REVOKED_CLIENT_CERTS="${REVOKED_CLIENT_CERTS:-}"

echo "==> Preflight: mutual TLS files"
[[ -f client-ca.pem ]] || { echo "client-ca.pem missing: run 'python tools/client_ca.py init-ca' first" >&2; exit 1; }
if grep -rlqs "PRIVATE KEY" --include='*.pem' --exclude-dir=.git . ; then
  echo "A private key (*.pem) is inside the build context; move it out before building:" >&2
  grep -rls "PRIVATE KEY" --include='*.pem' --exclude-dir=.git . >&2; exit 1
fi

gcloud config set project "$PROJECT_ID" >/dev/null

echo "==> Enabling APIs"
gcloud services enable compute.googleapis.com confidentialcomputing.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com iamcredentials.googleapis.com logging.googleapis.com

echo "==> Artifact Registry repository"
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"

echo "==> Dedicated build service account and source staging bucket"
# Cloud Build's default identity differs between projects (legacy Cloud Build SA or
# Compute Engine default SA) and may lack push rights. Use our own least-privilege SA.
gcloud iam service-accounts describe "$BUILD_SA_EMAIL" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$BUILD_SA_NAME" --display-name="Confidential classifier image builder"
gcloud storage buckets describe "gs://${BUILD_BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${BUILD_BUCKET}" --location="$REGION" --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding "gs://${BUILD_BUCKET}" \
  --member="serviceAccount:${BUILD_SA_EMAIL}" --role=roles/storage.objectViewer >/dev/null
gcloud artifacts repositories add-iam-policy-binding "$REPO" --location="$REGION" \
  --member="serviceAccount:${BUILD_SA_EMAIL}" --role=roles/artifactregistry.writer >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${BUILD_SA_EMAIL}" \
  --role=roles/logging.logWriter --condition=None >/dev/null

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}"
echo "==> Building and pushing ${IMAGE}"
# New IAM grants can take a minute or two to propagate; retry the build a few times.
for ATTEMPT in 1 2 3; do
  if gcloud builds submit --config=cloudbuild.yaml --substitutions="_IMAGE=${IMAGE}:latest" \
       --service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA_EMAIL}" \
       --gcs-source-staging-dir="gs://${BUILD_BUCKET}/source" .; then
    break
  fi
  [[ $ATTEMPT -eq 3 ]] && { echo "Build failed after 3 attempts" >&2; exit 1; }
  echo "Build failed (attempt ${ATTEMPT}); waiting 60s for IAM propagation..." >&2
  sleep 60
done
DIGEST="$(gcloud artifacts docker images describe "${IMAGE}:latest" --format='value(image_summary.digest)')"
IMAGE_REF="${IMAGE}@${DIGEST}"   # deploy by digest, never by mutable tag

echo "==> Workload service account"
gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="Confidential classifier workload"
for ROLE in roles/confidentialcomputing.workloadUser roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" --condition=None >/dev/null
done
gcloud artifacts repositories add-iam-policy-binding "$REPO" --location="$REGION" \
  --member="serviceAccount:${SA_EMAIL}" --role=roles/artifactregistry.reader >/dev/null

echo "==> Firewall rule"
gcloud compute firewall-rules describe allow-classifier-grpc >/dev/null 2>&1 || \
  gcloud compute firewall-rules create allow-classifier-grpc --network=default --direction=INGRESS \
    --action=ALLOW --rules=tcp:50051 --source-ranges="$CLIENT_CIDR" --target-tags="$NET_TAG"

echo "==> Creating Confidential Space VM"
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --confidential-compute-type="$CC_TYPE" \
  --maintenance-policy=TERMINATE \
  --shielded-secure-boot \
  --image-project=confidential-space-images \
  --image-family="$CS_IMAGE_FAMILY" \
  --service-account="$SA_EMAIL" \
  --scopes=cloud-platform \
  --tags="$NET_TAG" \
  --metadata="^~^tee-image-reference=${IMAGE_REF}~tee-restart-policy=Always~tee-container-log-redirect=true~tee-env-AUDIENCE=${AUDIENCE}~tee-env-REVOKED_CLIENT_CERTS=${REVOKED_CLIENT_CERTS}"

IP="$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

cat <<MSG

Deployed.
  Endpoint:              ${IP}:50051
  Expected image digest: ${DIGEST}
  Audience:              ${AUDIENCE}

Give clients the digest and audience, then:
  python client.py --service tags --host ${IP} --images ./photos --expected-image-digest ${DIGEST} --audience ${AUDIENCE} \\
    --client-cert <client.pem> --client-key <client-key.pem>

Logs: gcloud logging read 'resource.type="gce_instance" AND logName:"confidential-space-launcher"' --limit=50
MSG
