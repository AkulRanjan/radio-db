"""Authentication (header / JWT secret / JWKS), health check, upload limits and quotas."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from radiodb.api import create_app
from radiodb.auth import Authenticator
from radiodb.config import Settings
from tests.conftest import conversation, wav_bytes

SECRET = "test-secret-at-least-32-bytes-long!!"
AUDIO = wav_bytes(conversation([("A", 2.0)]))


def _with(db, **overrides):
    for k, v in overrides.items():
        object.__setattr__(db.settings, k, v)
    return TestClient(create_app(db, inline_worker=False))


def _token(key=SECRET, alg="HS256", exp_in=300, headers=None, **claims):
    payload = {"sub": "alice", "exp": int(time.time()) + exp_in, **claims}
    return jwt.encode(payload, key, algorithm=alg, headers=headers)


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


# ---- JWT with shared secret -------------------------------------------------------------------

def test_jwt_secret_mode(db):
    c = _with(db, auth_mode="jwt", jwt_secret=SECRET, jwt_audience="radiodb", jwt_issuer="https://idp.example")
    good = _token(aud="radiodb", iss="https://idp.example")
    r = c.post("/recordings", headers=bearer(good), data={"process": "false"}, files={"file": ("a.wav", AUDIO)})
    assert r.status_code == 201 and r.json()["recording"]["user_id"] == "alice"
    assert len(c.get("/recordings", headers=bearer(good)).json()) == 1

    cases = {
        "no token": {},
        "header only": {"X-User-Id": "alice"},  # the dev header is ignored in jwt mode
        "expired": bearer(_token(aud="radiodb", iss="https://idp.example", exp_in=-120)),
        "wrong secret": bearer(_token(key="another-secret-that-is-32-bytes-long!", aud="radiodb", iss="https://idp.example")),
        "wrong audience": bearer(_token(aud="other-app", iss="https://idp.example")),
        "wrong issuer": bearer(_token(aud="radiodb", iss="https://evil.example")),
        "alg none": bearer(jwt.encode({"sub": "alice", "exp": int(time.time()) + 300}, None, algorithm="none")),
    }
    for name, headers in cases.items():
        r = c.get("/recordings", headers=headers)
        assert r.status_code == 401, name
        assert r.headers["www-authenticate"] == "Bearer"


def test_jwt_user_claim_and_isolation(db):
    c = _with(db, auth_mode="jwt", jwt_secret=SECRET, jwt_user_claim="email")
    alice, bob = _token(email="alice@example.com"), _token(email="bob@example.com")
    rid = c.post("/recordings", headers=bearer(alice), data={"process": "false"},
                 files={"file": ("a.wav", AUDIO)}).json()["recording"]["id"]
    assert c.get(f"/recordings/{rid}", headers=bearer(bob)).status_code == 404
    assert c.get("/recordings", headers=bearer(_token())).status_code == 401  # no email claim


def test_jwt_config_validation(tmp_path):
    with pytest.raises(ValueError, match="needs"):
        Authenticator(Settings(home=tmp_path, auth_mode="jwt"))
    with pytest.raises(ValueError, match="must be"):
        Authenticator(Settings(home=tmp_path, auth_mode="basic"))


# ---- JWT via JWKS (identity provider public keys) --------------------------------------------

@pytest.fixture
def jwks_server():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    body = json.dumps({"keys": [{**jwk, "kid": "k1", "use": "sig", "alg": "RS256"}]}).encode()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield key, f"http://127.0.0.1:{srv.server_port}/.well-known/jwks.json"
    srv.shutdown()


def test_jwks_mode(db, jwks_server):
    key, url = jwks_server
    c = _with(db, auth_mode="jwt", jwt_jwks_url=url, jwt_audience="radiodb")
    good = _token(key=key, alg="RS256", headers={"kid": "k1"}, aud="radiodb")
    assert c.get("/recordings", headers=bearer(good)).status_code == 200

    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = _token(key=other, alg="RS256", headers={"kid": "k1"}, aud="radiodb")
    assert c.get("/recordings", headers=bearer(forged)).status_code == 401
    unknown_kid = _token(key=key, alg="RS256", headers={"kid": "rotated-away"}, aud="radiodb")
    assert c.get("/recordings", headers=bearer(unknown_kid)).status_code == 401
    hs_with_public_key = _token(key=SECRET, aud="radiodb", headers={"kid": "k1"})  # algorithm confusion
    assert c.get("/recordings", headers=bearer(hs_with_public_key)).status_code == 401


# ---- health, limits, quota --------------------------------------------------------------------

def test_health(db):
    c = _with(db)
    r = c.get("/health")  # no auth needed
    assert r.status_code == 200 and r.json()["checks"] == {"catalog": "ok", "blobs": "ok"}
    db.blobs.root = db.blobs.root / "missing" / "dir"
    r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded" and r.json()["checks"]["blobs"].startswith("error")


def test_upload_size_limit(db):
    c = _with(db, max_upload_mb=1)
    big = wav_bytes(conversation([("A", 40.0)]))  # ~1.3 MB
    r = c.post("/recordings", headers={"X-User-Id": "u1"}, files={"file": ("big.wav", big)})
    assert r.status_code == 413
    assert c.post("/recordings", headers={"X-User-Id": "u1"}, data={"process": "false"},
                  files={"file": ("ok.wav", AUDIO)}).status_code == 201


def test_user_quota(db):
    c = _with(db, user_quota_hours=5 / 3600)  # 5 seconds of audio
    h = {"X-User-Id": "u1"}
    assert c.post("/recordings", headers=h, data={"process": "false"},
                  files={"file": ("a.wav", wav_bytes(conversation([("A", 3.0)])))}).status_code == 201
    r = c.post("/recordings", headers=h, files={"file": ("b.wav", wav_bytes(conversation([("B", 3.0)])))})
    assert r.status_code == 403 and "quota" in r.json()["detail"]
    # quotas are per user, and re-uploading a stored file doesn't count twice
    assert c.post("/recordings", headers={"X-User-Id": "u2"}, data={"process": "false"},
                  files={"file": ("b.wav", wav_bytes(conversation([("B", 3.0)])))}).status_code == 201
    assert c.post("/recordings", headers=h, data={"process": "false"},
                  files={"file": ("a.wav", wav_bytes(conversation([("A", 3.0)])))}).json()["created"] is False
