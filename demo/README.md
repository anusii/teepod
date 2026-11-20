## Image analysis with TEE

Perform image analysis (e.g., facial recognition and image tagging) inside a trusted execution environment.

### Deploy service in GCP confidential space

1. Create a project using [GCP cloud console](https://console.cloud.google.com)

2. Create certificate authority (CA) to issue client certificates
```bash
python tools/client_ca.py init-ca
```
This generate a private-public key pair:
- Private key: `$HOME/.classifier-ca/ca-key.pem` should be stored in a secure place
- Public key: `$HOME/.classifier-ca/client-ca.pem` will be included in the confidential space container image (step 3)

3. Build a confidential space instance and deploy the service in the project created above (step 1)
```bash
PROJECT_ID="MY_PROJECT_ID" ./deploy_server.sh
```
The script outputs essential information of the deployed service (e.g., host IP, container image hash etc.)


### Set up a client

1. Prepare client certificate
```bash
python tools/client_ca.py new-client --name photopod --out-dir ~/client-certs
```
This generates the client's private key (`photopod-key.pem`) and a certificate signing request (`photopod.csr`, containing the paired public key and identifying information such as client common name).

2. CA signs the client certificate using its private key 
```bash
python tools/client_ca.py sign --csr ~/client-certs/photopod.csr --out ~/client-certs/photopod-signed.pem --days 90
```

3. Install dependencies
```bash
python3 -m venv client-venv; source client-venv/bin/activate; pip install -r requirements-client.txt
```

4. (Optional) Compile protobuf definition
```bash
python -m grpc_tools.protoc --proto_path=proto --python_out=. --grpc_python_out=. classifier.proto
```

4. Run the client
```bash

# For image tagging
SERVICE=tags; IMAGES=../data/images

# For facial recognition
# SERVICE=faces; IMAGES=../data/scientists

python client.py \
    --service $SERVICE --images $IMAGES \
    --host ${HOST_IP} \
    --expected-image-digest ${CONTAINER_IMAGE_HASH} \
    --audience image-classifier-clients \
    --client-cert ~/client-certs/photopod-signed.pem \
    --client-key ~/client-certs/photopod-key.pem \
    --allow-debug-image
```
where `HOST_IP` and `CONTAINER_IMAGE_HASH` are from the output of script `deploy_server.sh`
- `HOST_IP` is the IP address of the TEE virtual machine
- `CONTAINER_IMAGE_HASH` is the SHA-256 digest of the container image running in the TEE VM

### FAQ
Q: gRPC error after recompiling `proto/classifier.proto`

A: Make sure the protobuf compiler `protoc` is compatible with the client's Python dependency `protobuf`, see [Protocol-Buffers' versioning scheme](https://protobuf.dev/support/version-support/#numbering). If the version of `protobuf` Python package is `major.minor.point`, it is best to use `protoc` version `minor.point`.

