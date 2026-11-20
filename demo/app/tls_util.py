"""Ephemeral TLS identity generated inside the TEE.

The private key is created in enclave memory at startup and never written to
disk or exported, so only this attested workload can terminate TLS with it.
Clients pin the certificate after checking that its SHA-256 appears as a nonce
in a valid attestation token.
"""

import datetime
import hashlib

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

# Fixed name the client uses via grpc.ssl_target_name_override.
TLS_SERVER_NAME = "classifier.confidential-space.internal"


def generate_self_signed_cert(hostname: str = TLS_SERVER_NAME):
    """Returns (key_pem, cert_pem, cert_sha256_hex)."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_sha256 = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return key_pem, cert_pem, cert_sha256
