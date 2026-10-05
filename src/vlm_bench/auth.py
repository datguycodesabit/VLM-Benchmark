"""Local Sign in with ChatGPT support for eligible Responses API use.

OAuth credentials are kept in the operating system's keyring. The small local
configuration file stores only account references and a stable host identifier.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import httpx

AUTHORIZATION_ENDPOINT = "https://auth.openai.com/api/accounts/authorize"
TOKEN_ENDPOINT = "https://auth.openai.com/api/accounts/oauth/token"
JWKS_ENDPOINT = "https://auth.openai.com/.well-known/jwks.json"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
KEYRING_SERVICE = "vlm-bench-chatgpt"
CONFIG_NAME = "auth.json"


def _auth_dir() -> Path:
    configured = os.environ.get("VLM_BENCH_AUTH_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "vlm-bench"
    return Path.home() / ".config" / "vlm-bench"


def _config_path() -> Path:
    return _auth_dir() / CONFIG_NAME


def _load_config() -> dict[str, Any]:
    path = _config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": 1, "host_id": None, "accounts": []}
    except (OSError, UnicodeDecodeError, ValueError):
        raise RuntimeError("Local authentication metadata is unreadable") from None
    if not isinstance(data, dict) or not isinstance(data.get("accounts", []), list):
        raise RuntimeError("Local authentication metadata has an invalid format")
    return data


def _write_config(data: dict[str, Any]) -> None:
    directory = _auth_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            directory.chmod(0o700)
        temporary = _config_path().with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(_config_path())
    except OSError:
        raise RuntimeError("Could not safely save local authentication metadata") from None


def _host_id() -> str:
    config = _load_config()
    current = config.get("host_id")
    if isinstance(current, str) and current.startswith("urn:uuid:"):
        return current
    current = f"urn:uuid:{uuid.uuid4()}"
    config["host_id"] = current
    _write_config(config)
    return current


def _keyring():
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError(
            "ChatGPT sign-in needs OS keychain support; install with `uv sync --extra cloud`"
        ) from exc
    try:
        backend = keyring.get_keyring()
        if getattr(backend, "priority", 0) <= 0:
            raise RuntimeError("No secure operating-system keychain is available")
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Could not access the operating-system keychain") from None
    return keyring


def _get_secret(client_id: str) -> dict[str, Any]:
    keyring = _keyring()
    try:
        value = keyring.get_password(KEYRING_SERVICE, client_id)
    except Exception:
        raise RuntimeError(
            "Could not read ChatGPT credentials from the operating-system keychain"
        ) from None
    if not value:
        raise RuntimeError("ChatGPT credentials are not available in the operating-system keychain")
    try:
        data = json.loads(value)
    except ValueError:
        raise RuntimeError(
            "ChatGPT credentials in the operating-system keychain are invalid"
        ) from None
    if not isinstance(data, dict):
        raise RuntimeError("ChatGPT credentials in the operating-system keychain are invalid")
    return data


def _put_secret(client_id: str, data: dict[str, Any]) -> None:
    keyring = _keyring()
    try:
        keyring.set_password(KEYRING_SERVICE, client_id, json.dumps(data, separators=(",", ":")))
    except Exception:
        raise RuntimeError(
            "Could not safely store ChatGPT credentials in the operating-system keychain"
        ) from None


def _accounts(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config if config is not None else _load_config()
    accounts = config.get("accounts", [])
    return [
        item
        for item in accounts
        if isinstance(item, dict) and isinstance(item.get("client_id"), str)
    ]


def _account_for(client_id: str | None) -> dict[str, Any]:
    accounts = _accounts()
    if client_id is not None:
        selected = next((item for item in accounts if item.get("client_id") == client_id), None)
        if selected is None:
            raise RuntimeError("No ChatGPT account is registered for that client ID")
        return selected
    if len(accounts) == 1:
        return accounts[0]
    if not accounts:
        raise RuntimeError(
            "ChatGPT is not connected; run `vlm-bench auth login --provider chatgpt`"
        )
    raise RuntimeError("Multiple ChatGPT accounts are connected; select a client ID")


def status() -> dict[str, Any]:
    """Return account status with no access or refresh credentials."""
    config = _load_config()
    connected = []
    for account in _accounts(config):
        item = {
            "provider": "chatgpt",
            "client_id": account["client_id"],
            "email": account.get("email"),
            "subject": account.get("subject"),
            "scopes": account.get("scopes", []),
            "expires_at": account.get("expires_at"),
            "connected": False,
        }
        try:
            secret = _get_secret(account["client_id"])
            item["connected"] = bool(secret.get("refresh_token") or secret.get("access_token"))
        except RuntimeError:
            item["connected"] = False
        connected.append(item)
    return {
        "provider": "chatgpt",
        "host_id_configured": isinstance(config.get("host_id"), str),
        "accounts": connected,
    }


def logout(client_id: str | None = None) -> dict[str, Any]:
    """Remove a selected account's keychain credentials and local reference."""
    config = _load_config()
    accounts = _accounts(config)
    if client_id is None:
        if len(accounts) != 1:
            if not accounts:
                return {"provider": "chatgpt", "disconnected": False}
            raise RuntimeError("Multiple ChatGPT accounts are connected; specify a client ID")
        client_id = accounts[0]["client_id"]
    selected = next((item for item in accounts if item.get("client_id") == client_id), None)
    if selected is None:
        return {"provider": "chatgpt", "client_id": client_id, "disconnected": False}
    try:
        _keyring().delete_password(KEYRING_SERVICE, client_id)
    except Exception:
        raise RuntimeError(
            "Could not remove ChatGPT credentials from the operating-system keychain"
        ) from None
    config["accounts"] = [item for item in accounts if item.get("client_id") != client_id]
    _write_config(config)
    return {"provider": "chatgpt", "client_id": client_id, "disconnected": True}


def _now() -> float:
    return time.time()


def _iso_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _token_expiry(tokens: dict[str, Any]) -> float:
    expires_in = tokens.get("expires_in")
    if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
        return _now() + max(0, float(expires_in))
    return _now() + 300


def _exchange_form(
    form: dict[str, str],
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    try:
        with httpx.Client(
            timeout=timeout,
            transport=transport,
            trust_env=transport is None,
        ) as client:
            response = client.post(TOKEN_ENDPOINT, data=form)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"ChatGPT token request failed ({type(exc).__name__})") from None
    if not response.is_success:
        code = "HTTP " + str(response.status_code)
        try:
            error = response.json().get("error")
            if isinstance(error, str) and error.replace("_", "").isalnum():
                code = error[:80]
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"ChatGPT token request failed: {code}")
    try:
        tokens = response.json()
    except ValueError:
        raise RuntimeError("ChatGPT token endpoint returned invalid JSON") from None
    if not isinstance(tokens, dict) or not isinstance(tokens.get("access_token"), str):
        raise RuntimeError("ChatGPT token response did not contain an access token")
    return tokens


def _validate_id_token(
    encoded: str,
    client_id: str,
    nonce: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError(
            "ChatGPT sign-in needs JWT verification support; install with `uv sync --extra cloud`"
        ) from exc
    try:
        header = jwt.get_unverified_header(encoded)
    except Exception:
        raise RuntimeError("ChatGPT returned an invalid identity token") from None
    if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
        raise RuntimeError("ChatGPT identity token uses an unsupported signing key")
    try:
        with httpx.Client(
            timeout=20,
            transport=transport,
            trust_env=transport is None,
        ) as client:
            response = client.get(JWKS_ENDPOINT)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not retrieve ChatGPT signing keys ({type(exc).__name__})"
        ) from None
    if not response.is_success:
        raise RuntimeError("Could not retrieve ChatGPT signing keys")
    try:
        keys = response.json().get("keys")
    except (ValueError, AttributeError):
        raise RuntimeError("ChatGPT signing-key response is invalid") from None
    if not isinstance(keys, list):
        raise RuntimeError("ChatGPT signing-key response is invalid")
    jwk = next(
        (
            item
            for item in keys
            if isinstance(item, dict)
            and item.get("kid") == header["kid"]
            and item.get("alg", "RS256") == "RS256"
            and item.get("use", "sig") == "sig"
        ),
        None,
    )
    if jwk is None:
        raise RuntimeError("ChatGPT identity token signing key was not found")
    try:
        key = jwt.PyJWK.from_dict(jwk, algorithm="RS256").key
        claims = jwt.decode(
            encoded,
            key=key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            options={"require": ["exp", "iss", "aud", "sub", "nonce"]},
        )
    except Exception:
        raise RuntimeError("ChatGPT identity token failed signature or claim validation") from None
    if claims.get("nonce") != nonce or not isinstance(claims.get("sub"), str):
        raise RuntimeError("ChatGPT identity token did not match this sign-in attempt")
    return claims


def _scope_set(scope: Any) -> set[str]:
    if isinstance(scope, str):
        return set(scope.split())
    if isinstance(scope, list):
        return {value for value in scope if isinstance(value, str)}
    return set()


class _CallbackHandler(BaseHTTPRequestHandler):
    callback_path = "/auth/callback"
    result: dict[str, Any] = {}
    received = threading.Event()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API name
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != self.callback_path:
            self.send_error(404)
            return
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        duplicates = [
            key for key in ("state", "code", "client_id", "error") if len(query.get(key, [])) > 1
        ]
        type(self).result = {key: values[-1] for key, values in query.items() if values}
        if duplicates:
            type(self).result["_duplicate_params"] = ",".join(duplicates)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(
            b"<!doctype html><title>VLM Bench</title><p>Sign-in response received. "
            b"You may close this window and return to the terminal.</p>"
        )
        type(self).received.set()

    def log_message(self, *_: object) -> None:
        # The default server logger includes the callback query string, which
        # contains the single-use authorization code.
        return


def login(
    *,
    client_id: str | None = None,
    agent_name: str = "VLM Bench",
    timeout: float = 180,
    open_browser: Callable[[str], bool] = webbrowser.open,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Run a local PKCE registration/sign-in flow and return safe identity data."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("Sign-in timeout must be positive")
    keyring = _keyring()
    del keyring  # Verify a secure keychain is usable before starting OAuth.
    host_id = _host_id()
    pending_client_id = client_id or "dynamic_agent_client"
    previous_secret: dict[str, Any] | None = None
    if client_id:
        try:
            previous_secret = _get_secret(client_id)
        except RuntimeError:
            previous_secret = None

    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    callback = type(
        "CallbackHandler", (_CallbackHandler,), {"result": {}, "received": threading.Event()}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), callback)
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        daemon=True,
    )
    server_thread.start()
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/auth/callback"
    params = {
        "client_id": pending_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "ext_agent_host_id": host_id,
    }
    if not client_id:
        params["agent_name_hint"] = agent_name
    if previous_secret and isinstance(previous_secret.get("id_token"), str):
        params["id_token_hint"] = previous_secret["id_token"]
    auth_url = AUTHORIZATION_ENDPOINT + "?" + urllib.parse.urlencode(params)
    try:
        try:
            opened = open_browser(auth_url)
        except Exception:
            opened = False
        if opened is False:
            raise RuntimeError("Could not open a browser for ChatGPT sign-in")
        if not callback.received.wait(float(timeout)):
            raise RuntimeError("ChatGPT sign-in timed out; no credentials were saved")
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    query = callback.result
    if query.get("_duplicate_params"):
        raise RuntimeError("ChatGPT sign-in callback repeated a required parameter")
    if query.get("state") != state:
        raise RuntimeError("ChatGPT sign-in callback state did not match")
    if query.get("error"):
        # Never pass through provider-controlled error descriptions.
        error_code = re.sub(r"[^A-Za-z0-9_-]", "_", query["error"])[:80]
        raise RuntimeError(f"ChatGPT sign-in was not completed ({error_code})")
    code = query.get("code")
    if not code:
        raise RuntimeError("ChatGPT sign-in callback did not contain an authorization code")
    callback_client_id = query.get("client_id")
    if client_id:
        if callback_client_id and callback_client_id != client_id:
            raise RuntimeError("ChatGPT sign-in returned a different registered client")
        issued_client_id = client_id
    else:
        if (
            not isinstance(callback_client_id, str)
            or not callback_client_id
            or callback_client_id == "dynamic_agent_client"
        ):
            raise RuntimeError("ChatGPT registration did not return an issued client ID")
        issued_client_id = callback_client_id

    tokens = _exchange_form(
        {
            "grant_type": "authorization_code",
            "client_id": issued_client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": RESOURCE,
        },
        transport=transport,
        timeout=min(float(timeout), 30),
    )
    id_token = tokens.get("id_token")
    if not isinstance(id_token, str):
        raise RuntimeError("ChatGPT token response did not include an identity token")
    claims = _validate_id_token(id_token, issued_client_id, nonce, transport=transport)
    scopes = sorted(_scope_set(tokens.get("scope")))
    if "chatgpt.tokens.use.direct" not in scopes:
        raise RuntimeError("ChatGPT sign-in did not grant ChatGPT plan inference permission")
    if not isinstance(tokens.get("refresh_token"), str) or not tokens.get("refresh_token"):
        raise RuntimeError("ChatGPT sign-in did not return a refresh token")

    expires_at = _token_expiry(tokens)
    secret = {
        "client_id": issued_client_id,
        "subject": claims["sub"],
        "email": claims.get("email"),
        "issuer": ISSUER,
        "id_token": id_token,
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "token_type": tokens.get("token_type", "Bearer"),
        "scopes": scopes,
        "expires_at": expires_at,
        "saved_at": _iso_from_epoch(_now()),
    }
    config = _load_config()
    existing = _accounts(config)
    previous = next((a for a in existing if a.get("client_id") == issued_client_id), None)
    if previous and previous.get("subject") != claims["sub"]:
        raise RuntimeError(
            "ChatGPT sign-in resolved to a different account than the saved registration"
        )
    _put_secret(issued_client_id, secret)
    profile = {
        "client_id": issued_client_id,
        "subject": claims["sub"],
        "email": claims.get("email"),
        "issuer": ISSUER,
        "scopes": scopes,
        "expires_at": _iso_from_epoch(expires_at),
    }
    config["host_id"] = host_id
    config["accounts"] = [
        account for account in existing if account.get("client_id") != issued_client_id
    ] + [profile]
    _write_config(config)
    return {"provider": "chatgpt", **profile, "connected": True}


def _refresh(client_id: str, secret: dict[str, Any]) -> dict[str, Any]:
    refresh_token = secret.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError("ChatGPT credentials have no refresh token; sign in again")
    tokens = _exchange_form(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
            "resource": RESOURCE,
        }
    )
    updated = dict(secret)
    updated["access_token"] = tokens["access_token"]
    updated["expires_at"] = _token_expiry(tokens)
    if isinstance(tokens.get("refresh_token"), str) and tokens["refresh_token"]:
        updated["refresh_token"] = tokens["refresh_token"]
    updated["scopes"] = sorted(_scope_set(tokens.get("scope")) or set(secret.get("scopes", [])))
    updated["saved_at"] = _iso_from_epoch(_now())
    _put_secret(client_id, updated)

    config = _load_config()
    for account in _accounts(config):
        if account.get("client_id") == client_id:
            account["scopes"] = updated["scopes"]
            account["expires_at"] = _iso_from_epoch(updated["expires_at"])
    _write_config(config)
    return updated


def get_access_token(client_id: str | None = None) -> str:
    """Return a valid bearer token, refreshing near expiry through the keychain."""
    account = _account_for(client_id)
    selected_id = account["client_id"]
    secret = _get_secret(selected_id)
    if "chatgpt.tokens.use.direct" not in _scope_set(secret.get("scopes")):
        raise RuntimeError(
            "Connected ChatGPT account lacks plan inference permission; sign in again"
        )
    expiry = secret.get("expires_at")
    if not isinstance(expiry, (int, float)) or expiry <= _now() + 60:
        secret = _refresh(selected_id, secret)
    token = secret.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("ChatGPT credentials have no access token; sign in again")
    if "chatgpt.tokens.use.direct" not in _scope_set(secret.get("scopes")):
        raise RuntimeError("ChatGPT account no longer has plan inference permission; sign in again")
    return token
