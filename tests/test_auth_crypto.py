"""Exercise real signature verification when the cloud extra is installed."""

import json
import time

import httpx
import pytest

from vlm_bench import auth


def test_real_jwt_signature_nonce_and_issuer_validation():
    jwt = pytest.importorskip("jwt")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="test-key", alg="RS256", use="sig")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"keys": [jwk]}))
    claims = {
        "iss": auth.ISSUER,
        "aud": "test-client",
        "exp": int(time.time()) + 300,
        "sub": "subject",
        "nonce": "expected",
    }
    token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})
    assert (
        auth._validate_id_token(token, "test-client", "expected", transport=transport)["sub"]
        == "subject"
    )
    with pytest.raises(RuntimeError, match="sign-in attempt"):
        auth._validate_id_token(token, "test-client", "wrong", transport=transport)
    claims["iss"] = "https://attacker.example"
    token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})
    with pytest.raises(RuntimeError, match="signature or claim"):
        auth._validate_id_token(token, "test-client", "expected", transport=transport)
