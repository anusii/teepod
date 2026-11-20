"""Minimal client-certificate authority for mutual TLS.

Three commands, run in three different places:

  init-ca     (operator, once)    Create the CA. The CA private key goes to a
                                   directory OUTSIDE the project (default ~/.classifier-ca)
                                   and should live offline; only the public
                                   client-ca.pem is written into the project and
                                   baked into the image.
  new-client  (on the client)      Generate the client's private key and a CSR.
                                   The private key never leaves the client machine.
  sign        (operator)           Sign a CSR -> short-lived client certificate.
                                   Prints the cert's SHA-256 fingerprint for the
                                   revocation list.

Examples:
  python tools/client_ca.py init-ca
  python tools/client_ca.py new-client --name client-name --out-dir ~/classifier-client
  python tools/client_ca.py sign --csr ~/classifier-client/client-name.csr --out client-name-signed.pem --days 90
"""

import argparse
import datetime
import hashlib
import os
import pathlib
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

DEFAULT_CA_DIR = pathlib.Path.home() / ".classifier-ca"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _write_private(path: pathlib.Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _key_pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def init_ca(args):
    ca_dir = pathlib.Path(args.ca_dir).expanduser()
    if (ca_dir / "ca-key.pem").exists():
        sys.exit(f"{ca_dir / 'ca-key.pem'} already exists; refusing to overwrite")
    key = ec.generate_private_key(ec.SECP384R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, args.name)])
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - datetime.timedelta(minutes=5))
        .not_valid_after(_now() + datetime.timedelta(days=args.days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False,
                                     key_encipherment=False, data_encipherment=False,
                                     key_agreement=False, key_cert_sign=True, crl_sign=True,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(ski, critical=False)
        .sign(key, hashes.SHA384())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    _write_private(ca_dir / "ca-key.pem", _key_pem(key))
    (ca_dir / "client-ca.pem").write_bytes(cert_pem)
    out = PROJECT_ROOT / "client-ca.pem"
    out.write_bytes(cert_pem)
    print(f"CA private key: {ca_dir / 'ca-key.pem'}  (keep offline, never in the build context)")
    print(f"CA certificate: {out}  (public; baked into the image)")


def new_client(args):
    name = args.name
    out = pathlib.Path(args.out_dir).expanduser()
    key = ec.generate_private_key(ec.SECP256R1())
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .sign(key, hashes.SHA256())
    )
    key_path = out / f"{name}-key.pem"
    csr_path = out / f"{name}.csr"
    _write_private(key_path, _key_pem(key))
    csr_path.write_bytes(csr.public_bytes(serialization.Encoding.PEM))
    print(f"Private key: {key_path}  (stays on this machine)")
    print(f"CSR:         {csr_path}  (send this to the CA operator)")


def sign(args):
    ca_dir = pathlib.Path(args.ca_dir).expanduser()
    ca_key = serialization.load_pem_private_key((ca_dir / "ca-key.pem").read_bytes(), password=None)
    ca_cert = x509.load_pem_x509_certificate((ca_dir / "client-ca.pem").read_bytes())
    csr = x509.load_pem_x509_csr(pathlib.Path(args.csr).expanduser().read_bytes())
    if not csr.is_signature_valid:
        sys.exit("CSR signature is invalid")
    cert = (
        x509.CertificateBuilder()
        .subject_name(csr.subject)  # only the CN is copied; any extensions in the CSR are ignored
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - datetime.timedelta(minutes=5))
        .not_valid_after(_now() + datetime.timedelta(days=args.days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False,
                                     key_encipherment=False, data_encipherment=False,
                                     key_agreement=False, key_cert_sign=False, crl_sign=False,
                                     encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.public_key()), critical=False)
        .sign(ca_key, hashes.SHA384())
    )
    der = cert.public_bytes(serialization.Encoding.DER)
    pathlib.Path(args.out).expanduser().write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    print(f"Issued {args.out} for CN={cn}, valid {args.days} days")
    print(f"Fingerprint (for REVOKED_CLIENT_CERTS): {hashlib.sha256(der).hexdigest()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-ca"); p.set_defaults(fn=init_ca)
    p.add_argument("--ca-dir", default=str(DEFAULT_CA_DIR))
    p.add_argument("--name", default="Confidential Classifier Client CA")
    p.add_argument("--days", type=int, default=3650)
    p = sub.add_parser("new-client"); p.set_defaults(fn=new_client)
    p.add_argument("--name", required=True, help="client identity (CN), e.g. acme-prod")
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("sign"); p.set_defaults(fn=sign)
    p.add_argument("--ca-dir", default=str(DEFAULT_CA_DIR))
    p.add_argument("--csr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=int, default=90)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
