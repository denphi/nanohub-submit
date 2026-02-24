from __future__ import annotations

import csv
import hashlib
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

try:
    from importlib.metadata import PackageNotFoundError, version as package_version
except ImportError:  # pragma: no cover - Python 3.7 compatibility
    from importlib_metadata import PackageNotFoundError, version as package_version

from .auth import SignonCredentials, load_signon_credentials
from .builder import CommandBuilder
from .config import DEFAULT_CONFIG_PATH, SubmitClientConfig, load_submit_client_config
from .models import SubmitRequest
from .wire import (
    ConnectionClosedError,
    SubmitWireConnection,
    WireProtocolError,
    parse_submit_uri,
)

_ENV_BLACKLIST = {
    "CLASSPATH",
    "DISPLAY",
    "EDITOR",
    "HOME",
    "LOGNAME",
    "LS_COLORS",
    "MAIL",
    "MANPATH",
    "OLDPWD",
    "PATH",
    "PERLLIB",
    "PYTHONPATH",
    "RESULTSDIR",
    "SESSIONDIR",
    "SHELL",
    "SHLVL",
    "SVN_EDITOR",
    "SUBMIT_DOUBLEDASH",
    "TERM",
    "TIMEOUT",
    "USER",
    "VISUAL",
    "WINDOWID",
    "XTERM_LOCALE",
    "XTERM_SHELL",
    "XTERM_VERSION",
    "_",
}

_STATUS_RUNNING_STATES = {"executing", "waiting", "setting_up", "setup"}
_STATUS_DONE_STATES = {"finished", "failed", "aborted"}
_JOB_ID_MARKER = re.compile(
    r"^\.__(?:timestamp_[a-z]+|time_results|exit_code)\.(\d+)_\d+$"
)


def _resolve_client_version() -> str:
    try:
        return package_version("nanohubsubmit")
    except PackageNotFoundError:
        return "0.0.0"


def _collect_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _ENV_BLACKLIST:
            continue
        if key.startswith("group_"):
            continue
        if key.startswith("etc"):
            continue
        if key.startswith("rpath_"):
            continue
        if key.startswith("session"):
            continue
        environment[key] = value
    return environment


def _is_stream_tty(stream: Any) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False

    if bool(getattr(stream, "closed", False)):
        return False

    try:
        return bool(isatty())
    except Exception:
        return False


def _discover_mount_device(path: str) -> str:
    try:
        proc = subprocess.run(
            ["df", "-P", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return ""

    if proc.returncode != 0:
        return ""
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return ""
    # "Filesystem ..." header on line 1, device on line 2.
    fields = lines[1].split()
    if not fields:
        return ""
    device = fields[0]
    if ":" not in device:
        return f"{socket.getfqdn()}:{device}"
    return device


def _file_properties(path: str, mount_device: str) -> dict[str, object]:
    abs_path = os.path.abspath(path)
    mount_point = ""
    if os.path.exists(abs_path):
        probe = abs_path
        while probe:
            if os.path.ismount(probe):
                mount_point = probe
                break
            if probe == os.sep:
                break
            probe = os.path.dirname(probe)

    checksum = ""
    if os.path.exists(abs_path) and not os.path.isdir(abs_path):
        md5_hash = hashlib.md5()
        with open(abs_path, "rb") as fp:
            for block in iter(lambda: fp.read(4096), b""):
                md5_hash.update(block)
        checksum = md5_hash.hexdigest()

    if os.path.exists(abs_path):
        file_size = os.lstat(abs_path).st_size
        inode = os.lstat(abs_path).st_ino
    else:
        file_size = 0
        inode = 0

    return {
        "filePath": abs_path,
        "mountPoint": mount_point,
        "mountDevice": mount_device,
        "checksum": checksum,
        "fileSize": file_size,
        "inode": inode,
    }


def _parse_parameter_status_counts(parameter_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not parameter_path.exists():
        return counts

    try:
        with parameter_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                if not row[0].strip().isdigit():
                    continue
                state = row[1].strip().lower().replace(" ", "_")
                if state:
                    counts[state] = counts.get(state, 0) + 1
    except OSError:
        return counts
    return counts


def _infer_session_state(
    *,
    status_counts: dict[str, int],
    saw_start_marker: bool,
    saw_finish_marker: bool,
) -> str:
    if any(status_counts.get(state, 0) > 0 for state in _STATUS_RUNNING_STATES):
        return "running"
    if status_counts:
        total = sum(status_counts.values())
        done = sum(status_counts.get(state, 0) for state in _STATUS_DONE_STATES)
        if done == total and total > 0:
            return "complete"
        return "mixed"
    if saw_start_marker and not saw_finish_marker:
        return "running_maybe"
    if saw_finish_marker:
        return "complete_maybe"
    return "unknown"


@dataclass
class CommandResult:
    """Structured submit protocol result."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    authenticated: bool
    server_disconnected: bool
    server_messages: list[dict[str, Any]] = field(default_factory=list)
    job_id: int | None = None
    run_name: str | None = None
    server_version: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class ValidationCheck:
    name: str
    ok: bool
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class SubmitValidationResult:
    ok: bool
    args: list[str]
    checks: list[ValidationCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "args": list(self.args),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class DoctorReport:
    ok: bool
    checks: list[ValidationCheck]
    server_version: str | None = None
    server_capabilities: dict[str, bool] = field(default_factory=dict)
    connected_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "server_version": self.server_version,
            "server_capabilities": dict(self.server_capabilities),
            "connected_uri": self.connected_uri,
        }


@dataclass
class SessionStatusProbe:
    job_ids: list[int]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_ids": list(self.job_ids),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "ok": self.ok,
        }


@dataclass
class LocalSessionInfo:
    path: str
    run_name: str
    inferred_job_ids: list[int]
    inferred_state: str
    status_counts: dict[str, int]
    marker_files: int
    last_updated: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "run_name": self.run_name,
            "inferred_job_ids": list(self.inferred_job_ids),
            "inferred_state": self.inferred_state,
            "status_counts": dict(self.status_counts),
            "marker_files": self.marker_files,
            "last_updated": self.last_updated,
        }


@dataclass
class SessionsReport:
    sessions: list[LocalSessionInfo]
    inferred_job_ids: list[int]
    live_probe: SessionStatusProbe | None = None

    @property
    def ok(self) -> bool:
        return self.live_probe is None or self.live_probe.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "sessions": [session.to_dict() for session in self.sessions],
            "inferred_job_ids": list(self.inferred_job_ids),
            "live_probe": self.live_probe.to_dict() if self.live_probe else None,
        }


class CommandExecutionError(RuntimeError):
    """Raised when submit protocol execution fails."""


class AuthenticationError(CommandExecutionError):
    """Raised when server rejects signon/authentication."""


@dataclass
class _SessionState:
    client_id_hex: str
    action: str
    local_execution: bool
    authenticated: bool = False
    server_disconnected: bool = False
    exit_code: int | None = None
    server_id_hex: str | None = None
    server_version: str | None = None
    job_id: int | None = None
    run_name: str | None = None
    stdout_chunks: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)
    server_messages: list[dict[str, Any]] = field(default_factory=list)


class NanoHUBSubmitClient:
    """Modern client that talks directly to submit server over its socket protocol."""

    def __init__(
        self,
        *,
        config_path: str = DEFAULT_CONFIG_PATH,
        listen_uris: list[str] | None = None,
        submit_ssl_ca: str | None = None,
        maximum_connection_passes: int | None = None,
        username: str | None = None,
        password: str | None = None,
        session_token: str | None = None,
        session_id: str | None = None,
        cache_hosts: str | None = None,
        private_fingerprint: str | None = None,
        private_key_path: str | None = None,
        connect_timeout: float = 10.0,
        idle_timeout: float = 1.0,
        keepalive_interval: float = 15.0,
        check: bool = False,
        verbose: bool = False,
        operation_timeout: float | None = None,
        verbose_stream: TextIO | None = None,
    ) -> None:
        self.config_path = config_path
        self.listen_uris_override = listen_uris
        self.submit_ssl_ca_override = submit_ssl_ca
        self.maximum_connection_passes_override = maximum_connection_passes

        self.username = username
        self.password = password
        self.session_token = session_token
        self.session_id = session_id
        self.cache_hosts = cache_hosts
        self.private_fingerprint = private_fingerprint
        self.private_key_path = private_key_path

        self.connect_timeout = connect_timeout
        self.idle_timeout = idle_timeout
        self.keepalive_interval = keepalive_interval
        self.check = check
        self.verbose = verbose
        self.operation_timeout = operation_timeout
        self.verbose_stream = verbose_stream
        self.client_version = _resolve_client_version()

    def _verbose_log(self, message: str) -> None:
        if not self.verbose:
            return
        stream = self.verbose_stream if self.verbose_stream is not None else sys.stderr
        try:
            stream.write(f"[nanohubsubmit] {message}\n")
            stream.flush()
        except Exception:
            pass

    def _trace_message(self, direction: str, message: dict[str, Any]) -> None:
        if not self.verbose:
            return

        message_type = message.get("messageType")
        if not isinstance(message_type, str):
            message_type = "<unknown>"
        details: list[str] = []
        for key in ("success", "jobId", "runName", "version", "exitCode"):
            if key in message:
                details.append(f"{key}={message[key]}")

        text = message.get("text")
        if isinstance(text, str):
            text_preview = text.replace("\n", "\\n")
            if len(text_preview) > 96:
                text_preview = text_preview[:93] + "..."
            details.append(f"text={text_preview!r}")

        summary = f"{direction} {message_type}"
        if details:
            summary += " (" + ", ".join(details) + ")"
        self._verbose_log(summary)

    def submit(
        self, request: SubmitRequest, *, operation_timeout: float | None = None
    ) -> CommandResult:
        args = CommandBuilder.build_submit_args(request)
        return self._run(
            action="submit",
            args=args,
            local_execution=request.local,
            operation_timeout=operation_timeout,
        )

    def status(
        self, job_ids: list[int], *, operation_timeout: float | None = None
    ) -> CommandResult:
        args = CommandBuilder.build_status_args(job_ids)
        return self._run(
            action="status",
            args=args,
            local_execution=False,
            operation_timeout=operation_timeout,
        )

    def kill(
        self, job_ids: list[int], *, operation_timeout: float | None = None
    ) -> CommandResult:
        args = CommandBuilder.build_kill_args(job_ids)
        return self._run(
            action="kill",
            args=args,
            local_execution=False,
            operation_timeout=operation_timeout,
        )

    def venue_status(
        self,
        venues: list[str] | None = None,
        *,
        operation_timeout: float | None = None,
    ) -> CommandResult:
        args = CommandBuilder.build_venue_status_args(venues)
        return self._run(
            action="venue-status",
            args=args,
            local_execution=False,
            operation_timeout=operation_timeout,
        )

    def raw(
        self, args: list[str], *, operation_timeout: float | None = None
    ) -> CommandResult:
        if not args:
            raise ValueError("args cannot be empty")
        return self._run(
            action="raw",
            args=args,
            local_execution=False,
            operation_timeout=operation_timeout,
        )

    def validate_submit_request(self, request: SubmitRequest) -> SubmitValidationResult:
        checks: list[ValidationCheck] = []
        args: list[str] = []

        try:
            args = CommandBuilder.build_submit_args(request)
            checks.append(
                ValidationCheck(
                    name="command-shape",
                    ok=True,
                    severity="info",
                    message="submit arguments were built successfully",
                )
            )
        except ValueError as exc:
            checks.append(
                ValidationCheck(
                    name="command-shape",
                    ok=False,
                    severity="error",
                    message=str(exc),
                )
            )
            return SubmitValidationResult(ok=False, args=[], checks=checks)

        if request.run_name:
            if request.run_name.isalnum():
                checks.append(
                    ValidationCheck(
                        name="run-name",
                        ok=True,
                        severity="info",
                        message=f"run name '{request.run_name}' is alphanumeric",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="run-name",
                        ok=False,
                        severity="error",
                        message="run_name must be alphanumeric to match submit server rules",
                    )
                )

        if request.local and request.venues:
            checks.append(
                ValidationCheck(
                    name="local-venues",
                    ok=True,
                    severity="warning",
                    message="local mode ignores --venue settings",
                )
            )
        if not request.local and not request.venues:
            checks.append(
                ValidationCheck(
                    name="remote-venues",
                    ok=True,
                    severity="warning",
                    message="no venue specified; server defaults will be used",
                )
            )

        if request.detach and request.wait:
            checks.append(
                ValidationCheck(
                    name="detach-wait",
                    ok=True,
                    severity="warning",
                    message="--detach and --wait together are contradictory in practice",
                )
            )

        if request.data_file:
            if os.path.exists(request.data_file):
                checks.append(
                    ValidationCheck(
                        name="data-file",
                        ok=True,
                        severity="info",
                        message=f"data file exists: {request.data_file}",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="data-file",
                        ok=False,
                        severity="error",
                        message=f"data file does not exist: {request.data_file}",
                    )
                )

        for input_file in request.input_files:
            if os.path.exists(input_file):
                checks.append(
                    ValidationCheck(
                        name="input-file",
                        ok=True,
                        severity="info",
                        message=f"input file exists: {input_file}",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="input-file",
                        ok=False,
                        severity="error",
                        message=f"input file does not exist: {input_file}",
                    )
                )

        ok = all(check.severity != "error" for check in checks)
        return SubmitValidationResult(ok=ok, args=args, checks=checks)

    def doctor(self, *, probe_server: bool = True) -> DoctorReport:
        checks: list[ValidationCheck] = []
        server_version: str | None = None
        server_capabilities: dict[str, bool] = {}
        connected_uri: str | None = None

        if self.config_path:
            if os.path.exists(self.config_path):
                checks.append(
                    ValidationCheck(
                        name="config-path",
                        ok=True,
                        severity="info",
                        message=f"config file is present: {self.config_path}",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="config-path",
                        ok=False,
                        severity="warning",
                        message=f"config file not found: {self.config_path}",
                    )
                )

        try:
            cfg = self._load_config()
            checks.append(
                ValidationCheck(
                    name="connection-config",
                    ok=True,
                    severity="info",
                    message=f"{len(cfg.listen_uris)} listen URIs configured",
                )
            )
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    name="connection-config",
                    ok=False,
                    severity="error",
                    message=str(exc),
                )
            )
            return DoctorReport(ok=False, checks=checks)

        tls_uris = 0
        for uri in cfg.listen_uris:
            try:
                parsed = parse_submit_uri(uri)
            except Exception as exc:
                checks.append(
                    ValidationCheck(
                        name="listen-uri",
                        ok=False,
                        severity="error",
                        message=f"invalid listen URI '{uri}': {exc}",
                    )
                )
                continue

            checks.append(
                ValidationCheck(
                    name="listen-uri",
                    ok=True,
                    severity="info",
                    message=f"parsed URI {uri} ({parsed.protocol})",
                )
            )
            if parsed.protocol == "tls":
                tls_uris += 1

        if tls_uris:
            if cfg.submit_ssl_ca and os.path.exists(cfg.submit_ssl_ca):
                checks.append(
                    ValidationCheck(
                        name="tls-ca",
                        ok=True,
                        severity="info",
                        message=f"TLS CA file is present: {cfg.submit_ssl_ca}",
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="tls-ca",
                        ok=False,
                        severity="error",
                        message=f"TLS CA file not found: {cfg.submit_ssl_ca}",
                    )
                )

        if os.access(os.getcwd(), os.W_OK):
            checks.append(
                ValidationCheck(
                    name="workdir-access",
                    ok=True,
                    severity="info",
                    message=f"working directory is writable: {os.getcwd()}",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="workdir-access",
                    ok=False,
                    severity="error",
                    message=f"working directory is not writable: {os.getcwd()}",
                )
            )

        try:
            creds = self._load_credentials()
            auth_modes: list[str] = []
            if creds.session_token:
                auth_modes.append("session-token")
            if creds.password:
                auth_modes.append("password")
            if creds.private_fingerprint:
                auth_modes.append("private-fingerprint")
            checks.append(
                ValidationCheck(
                    name="credentials",
                    ok=True,
                    severity="info",
                    message="available auth modes: " + ", ".join(auth_modes),
                )
            )
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    name="credentials",
                    ok=False,
                    severity="error",
                    message=str(exc),
                )
            )
            ok = all(check.severity != "error" for check in checks)
            return DoctorReport(ok=ok, checks=checks)

        if probe_server:
            try:
                probe_result = self._probe_server(cfg=cfg, creds=creds)
            except Exception as exc:
                checks.append(
                    ValidationCheck(
                        name="server-probe",
                        ok=False,
                        severity="error",
                        message=str(exc),
                    )
                )
            else:
                server_version = probe_result["server_version"]
                server_capabilities = probe_result["capabilities"]
                connected_uri = probe_result["connected_uri"]
                checks.append(
                    ValidationCheck(
                        name="server-probe",
                        ok=True,
                        severity="info",
                        message=(
                            "server auth successful"
                            + (f", version={server_version}" if server_version else "")
                        ),
                    )
                )

        ok = all(check.severity != "error" for check in checks)
        return DoctorReport(
            ok=ok,
            checks=checks,
            server_version=server_version,
            server_capabilities=server_capabilities,
            connected_uri=connected_uri,
        )

    def list_sessions(
        self,
        *,
        root: str | Path = ".",
        max_depth: int = 3,
        include_live_status: bool = False,
        limit: int | None = None,
    ) -> SessionsReport:
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")

        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise ValueError(f"sessions root does not exist: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"sessions root is not a directory: {root_path}")

        sessions: list[LocalSessionInfo] = []
        root_parts = len(root_path.parts)

        for dirpath, dirnames, filenames in os.walk(root_path):
            current = Path(dirpath)
            depth = len(current.parts) - root_parts
            if depth > max_depth:
                dirnames[:] = []
                continue

            marker_files = 0
            inferred_job_ids: set[int] = set()
            saw_start_marker = False
            saw_finish_marker = False
            has_parameter_file = "parameterCombinations.csv" in filenames

            for filename in filenames:
                match = _JOB_ID_MARKER.match(filename)
                if match:
                    marker_files += 1
                    inferred_job_ids.add(int(match.group(1)))
                    if filename.startswith(".__timestamp_start"):
                        saw_start_marker = True
                    if filename.startswith(".__timestamp_finish"):
                        saw_finish_marker = True

            if not has_parameter_file and marker_files == 0:
                continue

            status_counts: dict[str, int] = {}
            parameter_path = current / "parameterCombinations.csv"
            if has_parameter_file:
                status_counts = _parse_parameter_status_counts(parameter_path)

            inferred_state = _infer_session_state(
                status_counts=status_counts,
                saw_start_marker=saw_start_marker,
                saw_finish_marker=saw_finish_marker,
            )

            last_updated = current.stat().st_mtime
            for filename in filenames:
                file_path = current / filename
                try:
                    mtime = file_path.stat().st_mtime
                except OSError:
                    continue
                if mtime > last_updated:
                    last_updated = mtime

            sessions.append(
                LocalSessionInfo(
                    path=str(current),
                    run_name=current.name,
                    inferred_job_ids=sorted(inferred_job_ids),
                    inferred_state=inferred_state,
                    status_counts=status_counts,
                    marker_files=marker_files,
                    last_updated=last_updated,
                )
            )

        sessions.sort(key=lambda session: session.last_updated, reverse=True)
        if limit is not None and limit >= 0:
            sessions = sessions[:limit]

        inferred_job_ids = sorted(
            {job_id for session in sessions for job_id in session.inferred_job_ids}
        )

        live_probe: SessionStatusProbe | None = None
        if include_live_status:
            if inferred_job_ids:
                try:
                    status_result = self.status(inferred_job_ids)
                except Exception as exc:
                    live_probe = SessionStatusProbe(job_ids=inferred_job_ids, error=str(exc))
                else:
                    live_probe = SessionStatusProbe(
                        job_ids=inferred_job_ids,
                        returncode=status_result.returncode,
                        stdout=status_result.stdout,
                        stderr=status_result.stderr,
                    )
            else:
                live_probe = SessionStatusProbe(
                    job_ids=[],
                    returncode=0,
                    stdout="",
                    stderr="",
                    error=None,
                )

        return SessionsReport(
            sessions=sessions,
            inferred_job_ids=inferred_job_ids,
            live_probe=live_probe,
        )

    def _load_config(self) -> SubmitClientConfig:
        cfg = load_submit_client_config(self.config_path)
        if self.listen_uris_override is not None:
            cfg.listen_uris = list(self.listen_uris_override)
        if self.submit_ssl_ca_override is not None:
            cfg.submit_ssl_ca = self.submit_ssl_ca_override
        if self.maximum_connection_passes_override is not None:
            cfg.maximum_connection_passes = self.maximum_connection_passes_override
        if not cfg.listen_uris:
            raise CommandExecutionError(
                "no submit listen URIs configured; set listenURIs in config or pass listen_uris"
            )
        return cfg

    def _load_credentials(self) -> SignonCredentials:
        creds = load_signon_credentials(
            username=self.username,
            password=self.password,
            session_token=self.session_token,
            session_id=self.session_id,
            cache_hosts=self.cache_hosts,
            private_fingerprint=self.private_fingerprint,
            private_key_path=self.private_key_path,
        )
        if not creds.has_any_auth():
            raise AuthenticationError(
                "missing authentication credentials (session token, password, or private fingerprint)"
            )
        return creds

    def _probe_server(
        self, *, cfg: SubmitClientConfig, creds: SignonCredentials
    ) -> dict[str, Any]:
        conn = SubmitWireConnection(
            listen_uris=cfg.listen_uris,
            submit_ssl_ca=cfg.submit_ssl_ca,
            maximum_connection_passes=cfg.maximum_connection_passes,
            connect_timeout=self.connect_timeout,
        )

        signature_path = None
        server_id_hex = None
        server_version = None
        capabilities = {}
        connected_uri = cfg.listen_uris[0] if cfg.listen_uris else None
        client_id_hex = uuid.uuid4().hex

        try:
            self._verbose_log(
                "server probe connect attempt (uris=%s)" % ", ".join(cfg.listen_uris)
            )
            conn.connect()
            self._verbose_log("server probe connected")
            signature_path = self._send_execution_context(
                conn=conn,
                cfg=cfg,
                client_id_hex=client_id_hex,
                args=["--help"],
            )
            for message in creds.to_signon_messages():
                self._trace_message("->", message)
                conn.send_json(message)

            deadline = time.time() + max(5.0, self.connect_timeout + 2.0)
            while time.time() < deadline:
                message = conn.receive_json(timeout=self.idle_timeout)
                if message is None:
                    continue
                self._trace_message("<-", message)

                message_type = message.get("messageType")
                if message_type == "serverId":
                    server_id = message.get("serverId")
                    if isinstance(server_id, str):
                        try:
                            server_id_hex = uuid.UUID(server_id).hex
                        except ValueError:
                            server_id_hex = server_id
                elif message_type == "serverVersion":
                    version_value = message.get("version")
                    if isinstance(version_value, str):
                        server_version = version_value
                elif message_type == "serverReadyForSignon":
                    signon_message = {"messageType": "signon"}
                    encrypted_message = message.get("encryptedMessage")
                    if (
                        isinstance(encrypted_message, str)
                        and server_id_hex
                        and creds.private_key_path
                    ):
                        auth_hash = creds.build_authentication_hash(
                            client_id_hex=client_id_hex,
                            server_id_hex=server_id_hex,
                            encrypted_message=encrypted_message,
                        )
                        if auth_hash:
                            signon_message["authenticationHash"] = auth_hash
                    self._trace_message("->", signon_message)
                    conn.send_json(signon_message)
                elif message_type == "authz":
                    if not bool(message.get("success")):
                        raise AuthenticationError("session authentication failed")
                    capabilities = {
                        "hasDistributor": bool(message.get("hasDistributor")),
                        "hasHarvester": bool(message.get("hasHarvester")),
                        "hasJobStatus": bool(message.get("hasJobStatus")),
                        "hasJobKill": bool(message.get("hasJobKill")),
                        "hasVenueProbe": bool(message.get("hasVenueProbe")),
                    }
                    return {
                        "server_version": server_version,
                        "capabilities": capabilities,
                        "connected_uri": connected_uri,
                    }
        finally:
            conn.close()
            if signature_path and os.path.exists(signature_path):
                try:
                    os.remove(signature_path)
                except OSError:
                    pass

        raise CommandExecutionError("server probe timed out waiting for authz")

    def _run(
        self,
        *,
        action: str,
        args: list[str],
        local_execution: bool,
        operation_timeout: float | None = None,
    ) -> CommandResult:
        effective_timeout = (
            self.operation_timeout if operation_timeout is None else operation_timeout
        )
        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("operation_timeout must be positive when provided")

        cfg = self._load_config()
        creds = self._load_credentials()
        self._verbose_log(
            "starting action=%s args=%s"
            % (action, " ".join(args) if args else "<none>")
        )
        self._verbose_log("connect URIs: %s" % ", ".join(cfg.listen_uris))
        state = _SessionState(
            client_id_hex=uuid.uuid4().hex,
            action=action,
            local_execution=local_execution,
        )
        conn = SubmitWireConnection(
            listen_uris=cfg.listen_uris,
            submit_ssl_ca=cfg.submit_ssl_ca,
            maximum_connection_passes=cfg.maximum_connection_passes,
            connect_timeout=self.connect_timeout,
        )

        signature_path: str | None = None
        last_send_time = time.time()
        started_at = time.time()
        try:
            conn.connect()
            self._verbose_log("connected to submit server")
            signature_path = self._send_execution_context(
                conn=conn,
                cfg=cfg,
                client_id_hex=state.client_id_hex,
                args=args,
            )
            for message in creds.to_signon_messages():
                self._trace_message("->", message)
                conn.send_json(message)
                last_send_time = time.time()

            while state.exit_code is None:
                if (
                    effective_timeout is not None
                    and time.time() - started_at >= effective_timeout
                ):
                    raise CommandExecutionError(
                        "timed out waiting for %s response after %.1fs"
                        % (action, effective_timeout)
                    )

                server_message = conn.receive_json(timeout=self.idle_timeout)
                if server_message is None:
                    if time.time() - last_send_time >= self.keepalive_interval:
                        keepalive = {"messageType": "null"}
                        self._trace_message("->", keepalive)
                        conn.send_json(keepalive)
                        last_send_time = time.time()
                    continue

                self._trace_message("<-", server_message)
                state.server_messages.append(server_message)
                outbound_messages = self._process_server_message(
                    conn=conn,
                    state=state,
                    creds=creds,
                    message=server_message,
                )
                if outbound_messages:
                    for outbound in outbound_messages:
                        self._trace_message("->", outbound)
                        conn.send_json(outbound)
                        last_send_time = time.time()
        except AuthenticationError:
            raise
        except ConnectionClosedError:
            state.server_disconnected = True
            if state.exit_code is None:
                state.exit_code = 1
        except (OSError, WireProtocolError, ConnectionError) as exc:
            raise CommandExecutionError(str(exc)) from exc
        finally:
            conn.close()
            if signature_path and os.path.exists(signature_path):
                try:
                    os.remove(signature_path)
                except OSError:
                    pass

        result = CommandResult(
            args=["submit", *args],
            returncode=state.exit_code if state.exit_code is not None else 1,
            stdout="".join(state.stdout_chunks),
            stderr="".join(state.stderr_chunks),
            authenticated=state.authenticated,
            server_disconnected=state.server_disconnected,
            server_messages=state.server_messages,
            job_id=state.job_id,
            run_name=state.run_name,
            server_version=state.server_version,
        )
        self._verbose_log(
            "finished action=%s returncode=%d job_id=%s run_name=%s"
            % (action, result.returncode, result.job_id, result.run_name)
        )
        if self.check and not result.ok:
            raise CommandExecutionError(
                f"submit protocol exited with code {result.returncode}: {result.stderr.strip()}"
            )
        return result

    def _send_execution_context(
        self,
        *,
        conn: SubmitWireConnection,
        cfg: SubmitClientConfig,
        client_id_hex: str,
        args: list[str],
    ) -> str:
        work_directory = os.getcwd()
        mount_device = _discover_mount_device(work_directory)
        work_directory_properties = _file_properties(work_directory, mount_device)
        if work_directory_properties["inode"] == 0:
            raise CommandExecutionError("could not determine working directory inode")

        fd, signature_path = tempfile.mkstemp(prefix=".submitSignature_", dir=work_directory)
        os.close(fd)
        with open(signature_path, "w", encoding="utf-8") as fp:
            fp.write(str(int(time.time())))
        signature_properties = _file_properties(signature_path, mount_device)

        context_messages: list[dict[str, Any]] = [
            {"messageType": "clientId", "clientId": client_id_hex},
            {"messageType": "clientVersion", "version": self.client_version},
            {
                "messageType": "doubleDashTerminator",
                "doubleDashTerminator": cfg.double_dash_terminator,
            },
            {"messageType": "args", "args": ["nanohub-submit", *args]},
        ]

        env_vars = _collect_environment()
        if env_vars:
            context_messages.append({"messageType": "vars", "vars": env_vars})

        umask = os.umask(0)
        os.umask(umask)
        context_messages.extend(
            [
                {"messageType": "umask", "umask": umask},
                {
                    "messageType": "pwd",
                    "path": work_directory,
                    "properties": work_directory_properties,
                    "signature": signature_properties,
                },
                {
                    "messageType": "isClientTTY",
                    "isClientTTY": _is_stream_tty(sys.stdin) and _is_stream_tty(sys.stdout),
                },
                {"messageType": "pegasusVersion", "version": cfg.pegasus_version},
            ]
        )

        for message in context_messages:
            conn.send_json(message)

        return signature_path

    def _process_server_message(
        self,
        *,
        conn: SubmitWireConnection,
        state: _SessionState,
        creds: SignonCredentials,
        message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        message_type = message.get("messageType")
        if not isinstance(message_type, str):
            return []

        if message_type == "serverId":
            server_id = message.get("serverId")
            if isinstance(server_id, str):
                try:
                    state.server_id_hex = uuid.UUID(server_id).hex
                except ValueError:
                    state.server_id_hex = server_id
            return []

        if message_type == "serverVersion":
            version_value = message.get("version")
            if isinstance(version_value, str):
                state.server_version = version_value
            return []

        if message_type == "writeStdout":
            text = message.get("text")
            if isinstance(text, str):
                state.stdout_chunks.append(text)
            return []

        if message_type == "writeStderr":
            text = message.get("text")
            if isinstance(text, str):
                state.stderr_chunks.append(text)
            return []

        if message_type == "message":
            text = message.get("text")
            if isinstance(text, str):
                if text.endswith("\n"):
                    state.stderr_chunks.append(text)
                else:
                    state.stderr_chunks.append(text + "\n")
            return []

        if message_type == "jobId":
            job_id = message.get("jobId")
            if isinstance(job_id, int):
                state.job_id = job_id
            return []

        if message_type == "runName":
            run_name = message.get("runName")
            if isinstance(run_name, str):
                state.run_name = run_name
            job_id = message.get("jobId")
            if isinstance(job_id, int):
                state.job_id = job_id
            return []

        if message_type == "serverReadyForSignon":
            signon_message: dict[str, Any] = {"messageType": "signon"}
            encrypted_message = message.get("encryptedMessage")
            if (
                isinstance(encrypted_message, str)
                and state.server_id_hex
                and creds.private_key_path
            ):
                auth_hash = creds.build_authentication_hash(
                    client_id_hex=state.client_id_hex,
                    server_id_hex=state.server_id_hex,
                    encrypted_message=encrypted_message,
                )
                if auth_hash:
                    signon_message["authenticationHash"] = auth_hash
            return [signon_message]

        if message_type == "authz":
            success = bool(message.get("success"))
            if not success:
                raise AuthenticationError("session authentication failed")
            state.authenticated = True
            return [{"messageType": "submitCommandFileInodesSent"}]

        if message_type in {"noExportCommandFiles", "exportCommandFilesComplete"}:
            return [{"messageType": "parseArguments"}]

        if message_type == "argumentsParsed":
            if state.local_execution:
                return [{"messageType": "startLocal"}]
            return [{"messageType": "setupRemote"}]

        if message_type == "serverReadyForInputMapping":
            return [{"messageType": "inputFileInodesSent"}]

        if message_type in {"noExportFiles", "exportFilesComplete"}:
            return [{"messageType": "startRemote"}]

        if message_type == "serverReadyForIO":
            return [{"messageType": "serverReadyForIO"}]

        if message_type == "readyToDetach":
            return [{"messageType": "detach"}]

        if message_type == "attached":
            return [{"messageType": "clientVersion", "version": self.client_version}]

        if message_type == "wait":
            return [{"messageType": "exit"}]

        if message_type == "childHasExited":
            return [{"messageType": "clientReadyForIO"}]

        if message_type == "noImportFile":
            return [{"messageType": "importFilesComplete"}]

        if message_type == "importFileFailed":
            return [{"messageType": "importFilesComplete"}]

        if message_type == "importFile":
            import_file = message.get("file")
            if isinstance(import_file, str) and import_file:
                return [{"messageType": "importFileReady", "file": import_file}]
            return []

        if message_type in {"serverExit", "exit"}:
            exit_code = message.get("exitCode")
            if not isinstance(exit_code, int):
                exit_code = 1
            state.exit_code = exit_code
            return []

        if message_type == "null":
            return []

        return []
