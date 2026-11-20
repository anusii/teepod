"""Fetch custom attestation tokens from the Confidential Space launcher.

Inside a Confidential Space VM the launcher exposes a local HTTP API on a Unix
socket. POST /v1/token with a custom audience and nonces returns a JWT signed by
Google's attestation service. The JWT contains claims about the hardware
(AMD SEV / Intel TDX), the Confidential Space image, and the exact container
image digest that is running -- which is what remote clients verify.
"""

import http.client
import json
import socket

TEE_SERVER_SOCKET = "/run/container_launcher/teeserver.sock"
TOKEN_ENDPOINT = "/v1/token"


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._socket_path)


def fetch_attestation_token(audience: str, nonces: list[str], token_type: str = "OIDC") -> str:
    """Blocking call; run it in a worker thread from async code."""
    body = json.dumps({"audience": audience, "nonces": nonces, "token_type": token_type})
    conn = _UnixHTTPConnection(TEE_SERVER_SOCKET)
    try:
        conn.request("POST", TOKEN_ENDPOINT, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read().decode("utf-8")
    finally:
        conn.close()
    if resp.status != 200:
        raise RuntimeError(f"launcher returned HTTP {resp.status}: {payload[:200]}")
    return payload.strip()
