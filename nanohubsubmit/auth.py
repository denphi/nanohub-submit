from __future__ import annotations

"""Authentication and sign-on credential helpers for submit protocol sessions.

This module centralizes how credentials are discovered from environment/session
state and how optional challenge-response hashes are produced when the server
requests encrypted sign-on.
"""

import base64
import getpass
import hashlib
import os
import pwd
import subprocess
import uuid
from dataclasses import dataclass


def _read_resources_file(path: str) -> dict[str, str]:
    """Read legacy NanoHUB resources files and map auth-related keys."""
    attributes: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return attributes

    with open(path, "rb") as fp:
        for encoded in fp:
            try:
                record = encoded.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if record.startswith("session_token "):
                attributes["sessionToken"] = record.split()[1]
            elif record.startswith("sessionid "):
                attributes["sessionId"] = record.split()[1]
            elif record.startswith("cache_hosts "):
                attributes["cacheHosts"] = record.split()[1]
    return attributes


def _load_session_attributes() -> dict[str, str]:
    """Merge discovered session attributes from well-known resource locations."""
    paths: list[str] = []
    if "SESSIONDIR" in os.environ:
        paths.append(os.path.join(os.environ["SESSIONDIR"], "resources"))
    if "HOME" in os.environ:
        paths.append(os.path.join(os.environ["HOME"], ".default_resources"))

    merged: dict[str, str] = {}
    for resource_path in paths:
        merged.update(_read_resources_file(resource_path))
    return merged


def _compute_private_fingerprint(private_key_path: str) -> str | None:
    """Compute the submit-compatible private key fingerprint via OpenSSL tools."""
    if not private_key_path or not os.path.exists(private_key_path):
        return None

    modulus_proc = subprocess.run(
        ["openssl", "rsa", "-in", private_key_path, "-noout", "-modulus"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if modulus_proc.returncode != 0:
        return None

    if b"=" not in modulus_proc.stdout:
        return None
    modulus = modulus_proc.stdout.split(b"=", 1)[1].strip()

    digest_proc = subprocess.run(
        ["openssl", "dgst", "-sha256"],
        input=modulus,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if digest_proc.returncode != 0:
        return None

    try:
        return digest_proc.stdout.split(b" ", 1)[1].strip().decode("utf-8")
    except Exception:
        return None


@dataclass
class SignonCredentials:
    """Credential bundle used to build sign-on frames for the submit server."""

    user_name: str
    sudo_user_name: str | None = None
    ws_user_name: str | None = None
    password: str | None = None
    user_id: int | None = None
    session_id: str | None = None
    session_token: str | None = None
    cache_hosts: str | None = None
    private_fingerprint: str | None = None
    private_key_path: str | None = None

    def has_any_auth(self) -> bool:
        """Return whether any usable authentication mode is available."""
        return any(
            [
                self.session_token,
                self.password,
                self.private_fingerprint,
            ]
        )

    def to_signon_messages(self) -> list[dict[str, object]]:
        """Build ordered sign-on messages expected by submit protocol."""
        user_message: dict[str, object] = {
            "messageType": "userName",
            "userName": self.user_name,
        }
        if self.sudo_user_name:
            user_message["sudoUserName"] = self.sudo_user_name
        if self.ws_user_name:
            user_message["wsUserName"] = self.ws_user_name

        messages: list[dict[str, object]] = [user_message]
        ordered_attrs: list[tuple[str, str | int | None]] = [
            ("password", self.password),
            ("sessionId", self.session_id),
            ("sessionToken", self.session_token),
            ("cacheHosts", self.cache_hosts),
            ("privateFingerPrint", self.private_fingerprint),
        ]
        for message_type, value in ordered_attrs:
            if value is not None and value != "":
                messages.append({"messageType": message_type, message_type: value})

        messages.append({"messageType": "clientReadyForSignon"})
        return messages

    def build_authentication_hash(
        self,
        *,
        client_id_hex: str,
        server_id_hex: str,
        encrypted_message: str,
    ) -> str | None:
        """Build SHA256 authentication hash for encrypted sign-on challenges."""
        if not self.private_key_path:
            return None
        if not os.path.exists(self.private_key_path):
            return None

        try:
            encrypted_bytes = base64.b64decode(encrypted_message.encode("utf-8"))
        except Exception:
            return None

        decrypt_proc = subprocess.run(
            ["openssl", "rsautl", "-decrypt", "-inkey", self.private_key_path],
            input=encrypted_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if decrypt_proc.returncode != 0:
            return None

        secret = decrypt_proc.stdout.decode("utf-8", errors="replace").strip()
        if not secret:
            return None
        try:
            secret = uuid.UUID(secret).hex
        except Exception:
            pass

        digest = hashlib.sha256()
        digest.update(client_id_hex.encode("utf-8"))
        digest.update(secret.encode("utf-8"))
        digest.update(server_id_hex.encode("utf-8"))
        return digest.hexdigest()


def load_signon_credentials(
    *,
    username: str | None = None,
    password: str | None = None,
    session_token: str | None = None,
    session_id: str | None = None,
    cache_hosts: str | None = None,
    private_fingerprint: str | None = None,
    private_key_path: str | None = None,
) -> SignonCredentials:
    """Resolve runtime credentials from args, env vars, and session resources."""
    session_attributes = _load_session_attributes()

    resolved_user = username or os.environ.get("NANOHUB_SUBMIT_USER") or getpass.getuser()
    resolved_password = password or os.environ.get("NANOHUB_SUBMIT_PASSWORD")
    resolved_session_token = (
        session_token
        or os.environ.get("NANOHUB_SUBMIT_SESSION_TOKEN")
        or session_attributes.get("sessionToken")
    )
    resolved_session_id = session_id or session_attributes.get("sessionId")
    resolved_cache_hosts = cache_hosts or session_attributes.get("cacheHosts")

    resolved_private_key_path = private_key_path or os.environ.get("SUBMITPRIVATEKEYPATH")
    resolved_private_fingerprint = (
        private_fingerprint
        or os.environ.get("NANOHUB_SUBMIT_PRIVATE_FINGERPRINT")
        or (
            _compute_private_fingerprint(resolved_private_key_path)
            if resolved_private_key_path
            else None
        )
    )

    uid: int | None = None
    try:
        uid = pwd.getpwnam(resolved_user).pw_uid
    except Exception:
        uid = None

    creds = SignonCredentials(
        user_name=resolved_user,
        sudo_user_name=os.environ.get("SUDO_USER"),
        ws_user_name=os.environ.get("WS_USER"),
        password=resolved_password,
        user_id=uid,
        session_id=resolved_session_id,
        session_token=resolved_session_token,
        cache_hosts=resolved_cache_hosts,
        private_fingerprint=resolved_private_fingerprint,
        private_key_path=resolved_private_key_path,
    )
    return creds
