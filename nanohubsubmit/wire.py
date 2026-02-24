from __future__ import annotations

import json
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any


class WireProtocolError(RuntimeError):
    """Raised when message framing or handshake is invalid."""


class ConnectionClosedError(RuntimeError):
    """Raised when server closes the connection unexpectedly."""


@dataclass
class ParsedURI:
    protocol: str
    host: str = ""
    port: int = 0
    file_path: str = ""


def parse_submit_uri(uri: str) -> ParsedURI:
    # Legacy format is "tls://host:port" or "tcp://host:port" or "file:///path".
    parts = uri.split(":")
    if len(parts) == 3:
        protocol, host, port = parts
        return ParsedURI(protocol=protocol.lower(), host=host.lstrip("/"), port=int(port))
    if len(parts) == 2:
        protocol, file_path = parts
        # Match legacy behavior: remove first two leading slashes only.
        return ParsedURI(protocol=protocol.lower(), file_path=file_path.replace("/", "", 2))
    raise WireProtocolError(f"improper network specification: {uri}")


class SubmitWireConnection:
    """Socket/TLS transport for submit's JSON framed protocol."""

    def __init__(
        self,
        *,
        listen_uris: list[str],
        submit_ssl_ca: str | None,
        maximum_connection_passes: int,
        connect_timeout: float,
    ) -> None:
        self.listen_uris = listen_uris
        self.submit_ssl_ca = submit_ssl_ca
        self.maximum_connection_passes = max(1, maximum_connection_passes)
        self.connect_timeout = connect_timeout

        self._sock: socket.socket | ssl.SSLSocket | None = None
        self._in_buffer = b""
        self.default_buffer_size = 1024

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        if not self.listen_uris:
            raise WireProtocolError("no listen URIs configured")

        errors: list[str] = []
        for connection_pass in range(self.maximum_connection_passes):
            for uri in self.listen_uris:
                try:
                    self._sock = self._connect_uri(uri)
                    self._perform_handshake()
                    return
                except Exception as exc:
                    errors.append(f"{uri}: {exc}")
                    self.close()
            if connection_pass + 1 < self.maximum_connection_passes:
                time.sleep(1.0)
        raise ConnectionError("failed to connect to submit server; " + "; ".join(errors))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._in_buffer = b""

    def send_json(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message).encode("utf-8")
        header = f"json {len(payload)}\n".encode("utf-8")
        self._send_all(header + payload)

    def receive_json(self, timeout: float | None) -> dict[str, Any] | None:
        line = self._read_line(timeout=timeout)
        if line is None:
            return None
        args = line.split()
        if not args:
            return {"messageType": "null"}
        if args[0] == "null":
            return {"messageType": "null"}
        if args[0] != "json":
            raise WireProtocolError(f"unexpected server frame: {line!r}")
        if len(args) < 2:
            raise WireProtocolError(f"json frame missing payload length: {line!r}")

        payload_len = int(args[1])
        payload = self._read_exact(payload_len, timeout=timeout)
        try:
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise WireProtocolError("failed to decode server JSON message") from exc

    def _connect_uri(self, uri: str) -> socket.socket | ssl.SSLSocket:
        parsed = parse_submit_uri(uri)
        if parsed.protocol == "file":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect(parsed.file_path)
            return sock

        if parsed.protocol not in {"tcp", "tls"}:
            raise WireProtocolError(f"unknown protocol {parsed.protocol!r}")

        raw_sock = socket.create_connection((parsed.host, parsed.port), timeout=self.connect_timeout)
        if parsed.protocol == "tcp":
            return raw_sock

        context = ssl.create_default_context(cafile=self.submit_ssl_ca or None)
        if not self.submit_ssl_ca:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context.wrap_socket(raw_sock, server_hostname=parsed.host)

    def _perform_handshake(self) -> None:
        if self._sock is None:
            raise ConnectionError("connection is not open")
        handshake = "SUBMIT 0".ljust(32, " ").encode("utf-8")
        self._send_all(handshake)
        reply = self._read_exact(32, timeout=self.connect_timeout).decode("utf-8")
        if not reply.startswith("SUBMIT "):
            raise WireProtocolError(f"protocol mismatch during handshake: {reply!r}")
        parts = reply.split()
        if len(parts) < 2:
            raise WireProtocolError(f"invalid handshake response: {reply!r}")
        self.default_buffer_size = int(parts[-1])

    def _send_all(self, payload: bytes) -> None:
        if self._sock is None:
            raise ConnectionError("connection is not open")
        view = memoryview(payload)
        while view:
            written = self._sock.send(view)
            if written == 0:
                raise ConnectionClosedError("socket connection broken during write")
            view = view[written:]

    def _read_exact(self, size: int, timeout: float | None) -> bytes:
        if self._sock is None:
            raise ConnectionError("connection is not open")
        if size < 0:
            raise ValueError("size cannot be negative")
        while len(self._in_buffer) < size:
            self._sock.settimeout(timeout)
            try:
                chunk = self._sock.recv(max(4096, size - len(self._in_buffer)))
            except socket.timeout:
                raise TimeoutError from None
            if not chunk:
                raise ConnectionClosedError("socket connection closed by server")
            self._in_buffer += chunk

        data = self._in_buffer[:size]
        self._in_buffer = self._in_buffer[size:]
        return data

    def _read_line(self, timeout: float | None) -> str | None:
        if self._sock is None:
            raise ConnectionError("connection is not open")

        while True:
            nl_index = self._in_buffer.find(b"\n")
            if nl_index >= 0:
                line = self._in_buffer[:nl_index]
                self._in_buffer = self._in_buffer[nl_index + 1 :]
                return line.decode("utf-8")

            self._sock.settimeout(timeout)
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionClosedError("socket connection closed by server")
            self._in_buffer += chunk
