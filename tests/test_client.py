from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import threading
import uuid

import pytest

from nanohubsubmit.client import (
    AuthenticationError,
    CommandExecutionError,
    NanoHUBSubmitClient,
    _is_stream_tty,
)
from nanohubsubmit.models import ProgressMode, SubmitRequest
from nanohubsubmit.utils import SubmitCatalog
import nanohubsubmit.utils as submit_utils
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
            elif self.scenario == "raw-hang":
                self._run_raw_hang(self.conn)
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
            elif message_type == "submitCommandFileInodesSent":
                _send_json(conn, {"messageType": "noExportCommandFiles"})
            elif message_type == "parseArguments":
                _send_json(conn, {"messageType": "argumentsParsed"})
            elif message_type == "setupRemote":
                _send_json(conn, {"messageType": "serverReadyForInputMapping"})
            elif message_type == "inputFileInodesSent":
                _send_json(conn, {"messageType": "noExportFiles"})
            elif message_type == "startRemote":
                _send_json(conn, {"messageType": "writeStdout", "text": "STATUS OK\n"})
                _send_json(conn, {"messageType": "serverExit", "exitCode": 0})
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

    def _run_raw_hang(self, conn: socket.socket) -> None:
        _send_json(conn, {"messageType": "serverId", "serverId": str(uuid.uuid4())})
        _send_json(conn, {"messageType": "serverVersion", "version": "1.0.0"})

        buffer = b""
        try:
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
                    # Keep the session open and never emit exit to trigger client timeout.
                    continue
        except EOFError:
            return


def _make_client(local_fast_path: bool = True) -> NanoHUBSubmitClient:
    return NanoHUBSubmitClient(
        config_path="",
        listen_uris=["tcp://example.invalid:1"],
        username="test-user",
        session_token="session-token",
        connect_timeout=2.0,
        idle_timeout=0.1,
        keepalive_interval=0.2,
        local_fast_path=local_fast_path,
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


class _NoClosedStream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _BrokenTTY:
    closed = False

    def isatty(self) -> bool:
        raise OSError("tty check failed")


def test_stream_tty_helper_handles_missing_closed_attr() -> None:
    assert _is_stream_tty(_NoClosedStream(True)) is True
    assert _is_stream_tty(_NoClosedStream(False)) is False


def test_stream_tty_helper_handles_broken_isatty() -> None:
    assert _is_stream_tty(_BrokenTTY()) is False


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
    assert "setupRemote" in server.received_message_types
    assert "startRemote" in server.received_message_types


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
    tracked = client.list_tracked_runs()
    assert len(tracked) == 1
    assert tracked[0].job_id == 42
    assert tracked[0].command == "echo"


def test_client_raises_on_authentication_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client_conn, server_conn = socket.socketpair()
    _patch_connection(monkeypatch, client_conn)
    server = _FakeSubmitServer("auth-fail", server_conn)
    server.start()

    client = _make_client()
    with pytest.raises(AuthenticationError, match="authentication failed"):
        client.status([1])

    _assert_server_ok(server)


def test_client_raw_operation_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client_conn, server_conn = socket.socketpair()
    _patch_connection(monkeypatch, client_conn)
    server = _FakeSubmitServer("raw-hang", server_conn)
    server.start()

    client = _make_client()
    with pytest.raises(CommandExecutionError, match="timed out waiting for raw response"):
        client.raw(["--help", "tools"], operation_timeout=0.25)

    _assert_server_ok(server)


def test_client_raw_help_uses_command_file_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    client_conn, server_conn = socket.socketpair()
    _patch_connection(monkeypatch, client_conn)

    class _RawHelpServer(_FakeSubmitServer):
        def _run_status(self, conn: socket.socket) -> None:  # pragma: no cover - not used
            raise NotImplementedError

        def run(self) -> None:  # noqa: D401 - helper override
            try:
                self.conn.settimeout(5.0)
                handshake = b""
                while len(handshake) < 32:
                    chunk = self.conn.recv(32 - len(handshake))
                    if not chunk:
                        raise EOFError("missing handshake from client")
                    handshake += chunk
                self.conn.sendall("SUBMIT 1024".ljust(32).encode("utf-8"))

                _send_json(
                    self.conn, {"messageType": "serverId", "serverId": str(uuid.uuid4())}
                )
                _send_json(self.conn, {"messageType": "serverVersion", "version": "1.0.0"})

                buffer = b""
                while True:
                    message, buffer = _recv_json(self.conn, buffer)
                    message_type = str(message.get("messageType", ""))
                    self.received_message_types.append(message_type)
                    if message_type == "clientReadyForSignon":
                        _send_json(self.conn, {"messageType": "serverReadyForSignon"})
                    elif message_type == "signon":
                        _send_json(
                            self.conn,
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
                        _send_json(self.conn, {"messageType": "noExportCommandFiles"})
                    elif message_type == "parseArguments":
                        _send_json(self.conn, {"messageType": "argumentsParsed"})
                    elif message_type == "setupRemote":
                        _send_json(self.conn, {"messageType": "serverReadyForInputMapping"})
                    elif message_type == "inputFileInodesSent":
                        _send_json(self.conn, {"messageType": "noExportFiles"})
                    elif message_type == "startRemote":
                        _send_json(
                            self.conn,
                            {"messageType": "writeStdout", "text": "tools:\n  abacus\n"},
                        )
                        _send_json(
                            self.conn, {"messageType": "childHasExited", "childHasExited": True}
                        )
                    elif message_type == "clientReadyForIO":
                        _send_json(self.conn, {"messageType": "noImportFile"})
                    elif message_type == "importFilesComplete":
                        _send_json(self.conn, {"messageType": "serverExit", "exitCode": 0})
                        return
            except Exception as exc:  # pragma: no cover - test helper diagnostics
                self.error = exc
            finally:
                self.conn.close()

    server = _RawHelpServer("status", server_conn)
    server.start()

    client = _make_client()
    result = client.raw(["--help", "tools"], operation_timeout=2.0)

    _assert_server_ok(server)
    assert result.returncode == 0
    assert "abacus" in result.stdout
    assert "submitCommandFileInodesSent" in server.received_message_types
    assert "parseArguments" in server.received_message_types
    assert "setupRemote" in server.received_message_types
    assert "startRemote" in server.received_message_types
    assert "clientReadyForIO" in server.received_message_types
    assert "importFilesComplete" in server.received_message_types


def test_client_submit_local_fast_path_runs_immediately() -> None:
    client = _make_client()
    result = client.submit(
        SubmitRequest(
            command=sys.executable,
            command_arguments=["-c", "print('local-fast-path-ok')"],
            local=True,
        ),
        operation_timeout=5.0,
    )
    assert result.returncode == 0
    assert "local-fast-path-ok" in result.stdout
    assert result.authenticated is False
    tracked = client.list_tracked_runs()
    assert len(tracked) == 1
    assert tracked[0].local is True
    assert tracked[0].command == sys.executable


def test_client_submit_local_fast_path_timeout() -> None:
    client = _make_client()
    with pytest.raises(
        CommandExecutionError, match="timed out waiting for local command"
    ):
        client.submit(
            SubmitRequest(
                command=sys.executable,
                command_arguments=["-c", "import time; time.sleep(5)"],
                local=True,
            ),
            operation_timeout=0.1,
        )


def test_client_submit_local_parameter_sweep_submit_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _make_client(local_fast_path=False)
    result = client.submit(
        SubmitRequest(
            command=sys.executable,
            command_arguments=[
                "-c",
                "import sys; print(sys.argv[1])",
                "@@name",
            ],
            local=True,
            run_name="echotest",
            separator=",",
            parameters=["@@name=hub1,hub2,hub3"],
            progress=ProgressMode.SUBMIT,
        ),
        operation_timeout=10.0,
    )
    assert result.returncode == 0, result.to_dict()
    progress_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("=SUBMIT-PROGRESS=>")
    ]
    assert progress_lines
    assert any("%done=100.00" in line for line in progress_lines)
    assert "hub1" in result.stdout
    assert "hub2" in result.stdout
    assert "hub3" in result.stdout
    run_path = tmp_path / "echotest"
    assert run_path.is_dir()
    assert (run_path / "parameterCombinations.csv").is_file()
    assert (run_path / "01" / "echotest_01.stdout").is_file()
    assert (run_path / "02" / "echotest_02.stdout").is_file()
    assert (run_path / "03" / "echotest_03.stdout").is_file()


def test_client_catalog_is_loaded_on_demand_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    calls: list[tuple[bool, float]] = []

    def _fake_load_catalog(
        _client: NanoHUBSubmitClient,
        include_raw_help: bool = False,
        operation_timeout: float = 60.0,
    ) -> SubmitCatalog:
        calls.append((include_raw_help, operation_timeout))
        return SubmitCatalog(
            tools=["abacus"],
            venues=["workspace"],
            managers=["pegasus"],
        )

    monkeypatch.setattr(submit_utils, "load_available_catalog", _fake_load_catalog)

    first = client.get_catalog(operation_timeout=10.0)
    second = client.get_catalog(operation_timeout=0.1)
    assert first.tools == ["abacus"]
    assert second.tools == ["abacus"]
    assert len(calls) == 1

    third = client.get_catalog(refresh=True, operation_timeout=5.0)
    assert third.tools == ["abacus"]
    assert len(calls) == 2


def test_validate_submit_request_catalog_and_extra_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _make_client()
    catalog = SubmitCatalog(
        tools=["abacus", "gem5"],
        venues=["workspace"],
        managers=["pegasus"],
    )

    def _fake_load_catalog(
        _client: NanoHUBSubmitClient,
        include_raw_help: bool = False,  # noqa: ARG001
        operation_timeout: float = 60.0,  # noqa: ARG001
    ) -> SubmitCatalog:
        return catalog

    monkeypatch.setattr(submit_utils, "load_available_catalog", _fake_load_catalog)

    input_file = tmp_path / "input.dat"
    input_file.write_text("x", encoding="utf-8")
    extra_file = tmp_path / "extra.cfg"
    extra_file.write_text("y", encoding="utf-8")

    ok_request = SubmitRequest(
        command="abacus",
        venues=["workspace"],
        manager="pegasus",
        input_files=[str(input_file)],
    )
    ok_validation = client.validate_submit_request(
        ok_request,
        check_catalog=True,
        extra_existing_paths=[str(extra_file)],
    )
    assert ok_validation.ok is True

    bad_request = SubmitRequest(
        command="missing-tool",
        venues=["missing-venue"],
        manager="missing-manager",
        input_files=[str(input_file)],
    )
    bad_validation = client.validate_submit_request(
        bad_request,
        check_catalog=True,
        extra_existing_paths=[str(tmp_path / "missing.file")],
    )
    assert bad_validation.ok is False
    messages = [check.message for check in bad_validation.checks]
    assert any("tool not found in catalog" in message for message in messages)
    assert any("manager not found in catalog" in message for message in messages)
    assert any("venue not found in catalog" in message for message in messages)
    assert any("extra path does not exist" in message for message in messages)
