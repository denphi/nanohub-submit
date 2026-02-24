from __future__ import annotations

import json
import socket
import threading
import uuid

import pytest

from nanohubsubmit.client import AuthenticationError, NanoHUBSubmitClient
from nanohubsubmit.models import SubmitRequest
from nanohubsubmit.wire import SubmitWireConnection


def _send_json(conn: socket.socket, message: dict[str, object]) -> None:
    payload = json.dumps(message).encode("utf-8")
    conn.sendall(f"json {len(payload)}\n".encode("utf-8") + payload)


def _recv_line(conn: socket.socket, buffer: bytes) -> tuple[str, bytes]:
    while b"\n" not in buffer:
        chunk = conn.recv(4096)
        if not chunk:
            raise EOFError("connection closed")
        buffer += chunk
    line, buffer = buffer.split(b"\n", 1)
    return line.decode("utf-8"), buffer


def _recv_json(conn: socket.socket, buffer: bytes) -> tuple[dict[str, object], bytes]:
    line, buffer = _recv_line(conn, buffer)
    parts = line.split()
    if not parts:
        return {"messageType": "null"}, buffer
    if parts[0] == "null":
        return {"messageType": "null"}, buffer
    if parts[0] != "json":
        raise ValueError(f"unexpected frame header: {line!r}")
    payload_len = int(parts[1])
    while len(buffer) < payload_len:
        chunk = conn.recv(4096)
        if not chunk:
            raise EOFError("connection closed")
        buffer += chunk
    payload, buffer = buffer[:payload_len], buffer[payload_len:]
    return json.loads(payload.decode("utf-8")), buffer


class _FakeSubmitServer(threading.Thread):
    def __init__(self, scenario: str, conn: socket.socket) -> None:
        super().__init__(daemon=True)
        self.scenario = scenario
        self.conn = conn
        self.received_message_types: list[str] = []
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.conn.settimeout(5.0)

            handshake = b""
            while len(handshake) < 32:
                chunk = self.conn.recv(32 - len(handshake))
                if not chunk:
                    raise EOFError("missing handshake from client")
                handshake += chunk
            if not handshake.decode("utf-8").startswith("SUBMIT "):
                raise ValueError(f"invalid handshake {handshake!r}")
            self.conn.sendall("SUBMIT 1024".ljust(32).encode("utf-8"))

            if self.scenario == "status":
                self._run_status(self.conn)
            elif self.scenario == "submit":
                self._run_submit(self.conn)
            elif self.scenario == "auth-fail":
                self._run_auth_fail(self.conn)
            else:
                raise ValueError(f"unknown scenario {self.scenario!r}")
        except Exception as exc:  # pragma: no cover - test helper diagnostics
            self.error = exc
        finally:
            self.conn.close()

    def _run_status(self, conn: socket.socket) -> None:
        _send_json(conn, {"messageType": "serverId", "serverId": str(uuid.uuid4())})
        _send_json(conn, {"messageType": "serverVersion", "version": "1.0.0"})

        buffer = b""
        while True:
            message, buffer = _recv_json(conn, buffer)
            message_type = str(message.get("messageType", ""))
            self.received_message_types.append(message_type)

            if message_type == "clientReadyForSignon":
                _send_json(conn, {"messageType": "serverReadyForSignon"})
            elif message_type == "signon":
                _send_json(
                    conn,
                    {
                        "messageType": "authz",
                        "success": True,
                        "retry": False,
                        "hasDistributor": True,
                        "hasHarvester": True,
                        "hasJobStatus": True,
                        "hasJobKill": True,
                        "hasVenueProbe": True,
                    },
                )
            elif message_type == "parseArguments":
                _send_json(conn, {"messageType": "writeStdout", "text": "STATUS OK\n"})
                _send_json(conn, {"messageType": "exit", "exitCode": 0})
                return

    def _run_submit(self, conn: socket.socket) -> None:
        _send_json(conn, {"messageType": "serverId", "serverId": str(uuid.uuid4())})
        _send_json(conn, {"messageType": "serverVersion", "version": "1.0.0"})

        buffer = b""
        while True:
            message, buffer = _recv_json(conn, buffer)
            message_type = str(message.get("messageType", ""))
            self.received_message_types.append(message_type)

            if message_type == "clientReadyForSignon":
                _send_json(conn, {"messageType": "serverReadyForSignon"})
            elif message_type == "signon":
                _send_json(
                    conn,
                    {
                        "messageType": "authz",
                        "success": True,
                        "retry": False,
                        "hasDistributor": True,
                        "hasHarvester": True,
                        "hasJobStatus": True,
                        "hasJobKill": True,
                        "hasVenueProbe": True,
                    },
                )
            elif message_type == "submitCommandFileInodesSent":
                _send_json(conn, {"messageType": "noExportCommandFiles"})
            elif message_type == "parseArguments":
                _send_json(conn, {"messageType": "argumentsParsed"})
            elif message_type == "setupRemote":
                _send_json(conn, {"messageType": "serverReadyForInputMapping"})
            elif message_type == "inputFileInodesSent":
                _send_json(conn, {"messageType": "noExportFiles"})
            elif message_type == "startRemote":
                _send_json(conn, {"messageType": "jobId", "jobId": 42})
                _send_json(conn, {"messageType": "runName", "runName": "run-42", "jobId": 42})
                _send_json(conn, {"messageType": "writeStdout", "text": "SUBMITTED\n"})
                _send_json(conn, {"messageType": "serverExit", "exitCode": 0})
                return

    def _run_auth_fail(self, conn: socket.socket) -> None:
        _send_json(conn, {"messageType": "serverId", "serverId": str(uuid.uuid4())})
        buffer = b""
        while True:
            message, buffer = _recv_json(conn, buffer)
            message_type = str(message.get("messageType", ""))
            self.received_message_types.append(message_type)
            if message_type == "clientReadyForSignon":
                _send_json(conn, {"messageType": "serverReadyForSignon"})
            elif message_type == "signon":
                _send_json(conn, {"messageType": "authz", "success": False, "retry": False})
                return


def _make_client() -> NanoHUBSubmitClient:
    return NanoHUBSubmitClient(
        config_path="",
        listen_uris=["tcp://example.invalid:1"],
        username="test-user",
        session_token="session-token",
        connect_timeout=2.0,
        idle_timeout=0.1,
        keepalive_interval=0.2,
    )


def _assert_server_ok(server: _FakeSubmitServer) -> None:
    server.join(timeout=5.0)
    assert not server.is_alive()
    if server.error:
        raise server.error


def _patch_connection(monkeypatch: pytest.MonkeyPatch, client_conn: socket.socket) -> None:
    def _connect_uri(self: SubmitWireConnection, uri: str) -> socket.socket:  # noqa: ARG001
        return client_conn

    monkeypatch.setattr(SubmitWireConnection, "_connect_uri", _connect_uri)


def test_client_status_uses_socket_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    client_conn, server_conn = socket.socketpair()
    _patch_connection(monkeypatch, client_conn)
    server = _FakeSubmitServer("status", server_conn)
    server.start()

    client = _make_client()
    result = client.status([1001, 1002])

    _assert_server_ok(server)
    assert result.returncode == 0
    assert result.stdout == "STATUS OK\n"
    assert result.authenticated is True
    assert result.server_version == "1.0.0"
    assert "parseArguments" in server.received_message_types


def test_client_submit_drives_remote_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    client_conn, server_conn = socket.socketpair()
    _patch_connection(monkeypatch, client_conn)
    server = _FakeSubmitServer("submit", server_conn)
    server.start()

    client = _make_client()
    result = client.submit(
        SubmitRequest(
            command="echo",
            command_arguments=["hello"],
            venues=["workspace"],
        )
    )

    _assert_server_ok(server)
    assert result.returncode == 0
    assert result.stdout == "SUBMITTED\n"
    assert result.job_id == 42
    assert result.run_name == "run-42"
    assert "submitCommandFileInodesSent" in server.received_message_types
    assert "startRemote" in server.received_message_types


def test_client_raises_on_authentication_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client_conn, server_conn = socket.socketpair()
    _patch_connection(monkeypatch, client_conn)
    server = _FakeSubmitServer("auth-fail", server_conn)
    server.start()

    client = _make_client()
    with pytest.raises(AuthenticationError, match="authentication failed"):
        client.status([1])

    _assert_server_ok(server)
