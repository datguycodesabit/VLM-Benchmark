import json
import sys
import threading
import types
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from vlm_bench import auth


class FakeKeyring:
    def __init__(self):
        self.passwords = {}

    def get_keyring(self):
        return types.SimpleNamespace(priority=1)

    def get_password(self, service, username):
        return self.passwords.get((service, username))

    def set_password(self, service, username, value):
        self.passwords[(service, username)] = value

    def delete_password(self, service, username):
        self.passwords.pop((service, username), None)


@pytest.fixture
def secure_store(tmp_path, monkeypatch):
    keyring = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", keyring)
    monkeypatch.setattr(auth, "_auth_dir", lambda: tmp_path / "private-auth")
    return keyring


def _jwt_stub(claims):
    module = types.ModuleType("jwt")
    module.get_unverified_header = lambda token: {"alg": "RS256", "kid": "test-key"}

    class PyJWK:
        @classmethod
        def from_dict(cls, key, algorithm=None):
            assert key["kid"] == "test-key"
            assert algorithm == "RS256"
            return types.SimpleNamespace(key="verified-key")

    module.PyJWK = PyJWK

    def decode(token, key, algorithms, audience, issuer, options):
        assert token == "signed-id-token"
        assert key == "verified-key"
        assert algorithms == ["RS256"]
        assert audience == "oaiapp_created"
        assert issuer == auth.ISSUER
        assert "nonce" in options["require"]
        return dict(claims)

    module.decode = decode
    return module


class FakeLoopbackServer:
    next_query = {}
    callback_ready = threading.Event()

    def __init__(self, address, handler_class):
        self.handler_class = handler_class
        self.server_address = ("127.0.0.1", 14559)
        self.timeout = None
        type(self).next_query = {}
        type(self).callback_ready.clear()

    def serve_forever(self, poll_interval=0.1):
        if type(self).callback_ready.wait(2):
            self.handle_request()

    def shutdown(self):
        type(self).callback_ready.set()

    def handle_request(self):
        self.handler_class.result = dict(type(self).next_query)
        self.handler_class.received.set()

    def server_close(self):
        pass


def _fake_transport(claims_box, tokens=None, requests=None):
    tokens = tokens or {
        "access_token": "access-secret",
        "refresh_token": "refresh-secret",
        "id_token": "signed-id-token",
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": auth.SCOPES,
    }

    def handler(request):
        if requests is not None:
            requests.append(request)
        if request.url.path == "/.well-known/jwks.json":
            return httpx.Response(
                200,
                json={
                    "keys": [
                        {
                            "kid": "test-key",
                            "kty": "RSA",
                            "alg": "RS256",
                            "use": "sig",
                            "n": "AQAB",
                            "e": "AQAB",
                        }
                    ]
                },
            )
        assert request.url.path == "/api/accounts/oauth/token"
        body = parse_qs(request.content.decode())
        if body.get("grant_type") == ["authorization_code"]:
            assert body["client_id"] == ["oaiapp_created"]
            assert body["code"] == ["one-time-code"]
            assert body["redirect_uri"][0].startswith("http://127.0.0.1:")
            assert body["redirect_uri"][0].endswith("/auth/callback")
            assert body["resource"] == [auth.RESOURCE]
            assert body["code_verifier"]
            claims_box["nonce"] = claims_box["expected_nonce"]
        return httpx.Response(200, json=tokens)

    return httpx.MockTransport(handler)


def test_login_uses_pkce_validates_jwt_and_keeps_tokens_in_keyring(
    tmp_path, monkeypatch, secure_store
):
    monkeypatch.setattr(auth, "ThreadingHTTPServer", FakeLoopbackServer)
    claims = {
        "iss": auth.ISSUER,
        "aud": "oaiapp_created",
        "sub": "stable-subject",
        "email": "user@example.test",
        "exp": 4_000_000_000,
        "nonce": "will-be-filled",
    }
    monkeypatch.setitem(sys.modules, "jwt", _jwt_stub(claims))
    claims_box = {"expected_nonce": None}
    requests = []
    transport = _fake_transport(claims_box, requests=requests)
    authorization = {}

    def open_browser(url):
        authorization.update(parse_qs(urlsplit(url).query))
        claims_box["expected_nonce"] = authorization["nonce"][0]
        claims["nonce"] = authorization["nonce"][0]
        FakeLoopbackServer.next_query = {
            "state": authorization["state"][0],
            "code": "one-time-code",
            "client_id": "oaiapp_created",
        }
        FakeLoopbackServer.callback_ready.set()
        return True

    result = auth.login(open_browser=open_browser, transport=transport)
    assert result["connected"] is True
    assert result["email"] == "user@example.test"
    assert authorization["client_id"] == ["dynamic_agent_client"]
    assert authorization["ext_agent_host_id"][0].startswith("urn:uuid:")
    assert authorization["resource"] == [auth.RESOURCE]
    assert "chatgpt.tokens.use.direct" in authorization["scope"][0].split()
    assert authorization["code_challenge_method"] == ["S256"]
    assert len(requests) == 2

    keyring_value = secure_store.get_password(auth.KEYRING_SERVICE, "oaiapp_created")
    assert "access-secret" in keyring_value and "refresh-secret" in keyring_value
    config_path = tmp_path / "private-auth" / "auth.json"
    saved_config = config_path.read_text(encoding="utf-8")
    assert "access-secret" not in saved_config
    assert "refresh-secret" not in saved_config
    assert "signed-id-token" not in saved_config
    assert auth.get_access_token() == "access-secret"
    status = auth.status()
    assert status["accounts"][0]["connected"] is True
    assert "access-secret" not in repr(status)
    assert "refresh-secret" not in repr(status)
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert config_path.parent.stat().st_mode & 0o777 == 0o700


def test_login_rejects_callback_state_mismatch_before_token_exchange(
    tmp_path, monkeypatch, secure_store
):
    monkeypatch.setattr(auth, "ThreadingHTTPServer", FakeLoopbackServer)
    monkeypatch.setitem(sys.modules, "jwt", _jwt_stub({}))
    calls = []

    def open_browser(url):
        FakeLoopbackServer.next_query = {
            "state": "attacker-state",
            "code": "one-time-code",
            "client_id": "oaiapp_created",
        }
        FakeLoopbackServer.callback_ready.set()
        return True

    with pytest.raises(RuntimeError, match="state did not match"):
        auth.login(
            open_browser=open_browser,
            transport=_fake_transport({}, requests=calls),
        )
    assert calls == []
    assert not secure_store.passwords


def test_identity_signature_failure_does_not_disclose_token(monkeypatch):
    module = _jwt_stub({})
    module.decode = lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("sensitive signature details")
    )
    monkeypatch.setitem(sys.modules, "jwt", module)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"keys": [{"kid": "test-key", "alg": "RS256", "use": "sig"}]}
        )
    )
    with pytest.raises(RuntimeError, match="signature or claim validation") as caught:
        auth._validate_id_token("signed-id-token", "oaiapp_created", "nonce", transport=transport)
    assert "sensitive signature details" not in str(caught.value)


def test_access_token_refresh_rotates_refresh_token_atomically(monkeypatch, secure_store, tmp_path):
    config = {
        "schema_version": 1,
        "host_id": "urn:uuid:test-host",
        "accounts": [
            {
                "client_id": "oaiapp_created",
                "subject": "subject",
                "email": "user@example.test",
                "scopes": ["chatgpt.tokens.use.direct"],
                "expires_at": "expired",
            }
        ],
    }
    auth._write_config(config)
    secure_store.set_password(
        auth.KEYRING_SERVICE,
        "oaiapp_created",
        json.dumps(
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "scopes": ["chatgpt.tokens.use.direct"],
                "expires_at": 900,
            }
        ),
    )
    monkeypatch.setattr(auth, "_now", lambda: 1000)
    forms = []

    def refresh(form, **kwargs):
        forms.append(form)
        assert form["grant_type"] == "refresh_token"
        assert form["refresh_token"] == "old-refresh"
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "scope": "chatgpt.tokens.use.direct",
            "expires_in": 3600,
        }

    monkeypatch.setattr(auth, "_exchange_form", refresh)
    assert auth.get_access_token("oaiapp_created") == "new-access"
    refreshed = json.loads(secure_store.get_password(auth.KEYRING_SERVICE, "oaiapp_created"))
    assert refreshed["refresh_token"] == "new-refresh"
    assert refreshed["access_token"] == "new-access"
    assert refreshed["expires_at"] == 4600
    assert len(forms) == 1
    assert "new-access" not in (tmp_path / "private-auth" / "auth.json").read_text()


def test_status_and_logout_never_echo_or_write_tokens(secure_store, tmp_path):
    auth._write_config(
        {
            "schema_version": 1,
            "host_id": "urn:uuid:test-host",
            "accounts": [{"client_id": "oaiapp_created", "subject": "subject", "scopes": []}],
        }
    )
    secure_store.set_password(auth.KEYRING_SERVICE, "oaiapp_created", '{"access_token":"secret"}')
    status = auth.status()
    assert "secret" not in repr(status)
    result = auth.logout("oaiapp_created")
    assert result["disconnected"] is True
    assert not secure_store.passwords
    assert json.loads((tmp_path / "private-auth" / "auth.json").read_text())["accounts"] == []


def test_login_requires_secure_keyring(monkeypatch):
    monkeypatch.setattr(
        auth,
        "_keyring",
        lambda: (_ for _ in ()).throw(
            RuntimeError("No secure operating-system keychain is available")
        ),
    )
    with pytest.raises(RuntimeError, match="keychain"):
        auth.login(open_browser=lambda _: True)
