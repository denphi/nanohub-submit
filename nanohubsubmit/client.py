from __future__ import annotations

"""High-level submit client implementation.

This module contains:
- low-level request/session helpers used by the submit wire protocol
- validation and diagnostics models (`doctor`, `preflight`, session discovery)
- the main `NanoHUBSubmitClient` API for submit/status/kill/catalog flows
"""

import csv
import hashlib
import itertools
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO, TYPE_CHECKING

try:
    from importlib.metadata import PackageNotFoundError, version as package_version
except ImportError:  # pragma: no cover - Python 3.7 compatibility
    from importlib_metadata import PackageNotFoundError, version as package_version

from .auth import SignonCredentials, load_signon_credentials
from .builder import CommandBuilder
from .config import DEFAULT_CONFIG_PATH, SubmitClientConfig, load_submit_client_config
from .models import ProgressMode, SubmitRequest
from .wire import (
    ConnectionClosedError,
    SubmitWireConnection,
    WireProtocolError,
    parse_submit_uri,
)

if TYPE_CHECKING:
    from .utils import SubmitCatalog

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
_PROGRESS_LINE_RE = re.compile(r"^=SUBMIT-PROGRESS=>\s+(.*)$")
_PROGRESS_KV_RE = re.compile(r"([A-Za-z_%][A-Za-z0-9_%]*)=([^\s]+)")


# ---------------------------------------------------------------------------
# Internal helper functions used by diagnostics, local execution, and protocol
# framing preparation.
# ---------------------------------------------------------------------------
def _resolve_client_version() -> str:
    """Resolve installed package version for protocol self-identification."""
    try:
        return package_version("nanohubsubmit")
    except PackageNotFoundError:
        return "0.0.0"


def _collect_environment() -> dict[str, str]:
    """Collect environment variables safe to forward to submit server."""
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
    """Return whether stream looks like an open TTY-like object."""
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False

    if bool(getattr(stream, "closed", False)):
        return False

    try:
        return bool(isatty())
    except Exception:
        return False


def _is_jupyter_environment() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False

    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False

    shell_name = shell.__class__.__name__
    if shell_name == "ZMQInteractiveShell":
        return True

    config = getattr(shell, "config", {})
    return "IPKernelApp" in config


def _format_submit_progress_line(
    *,
    finished: int,
    failed: int,
    aborted: int,
    executing: int,
    setup: int,
    setting_up: int,
    total: int,
) -> str:
    """Render one submit-style progress line for local fast-path sweeps."""
    completed = finished + failed + aborted
    percent_done = (100.0 * float(completed) / float(total)) if total > 0 else 100.0
    return (
        "=SUBMIT-PROGRESS=> "
        "aborted=%d finished=%d failed=%d executing=%d waiting=0 setup=%d setting_up=%d "
        "%%done=%.2f timestamp=%.1f"
        % (
            aborted,
            finished,
            failed,
            executing,
            setup,
            setting_up,
            percent_done,
            time.time(),
        )
    )


def _expand_local_parameter_sweep(
    request: SubmitRequest,
) -> list[dict[str, str]]:
    """Expand request parameters into cartesian product substitution maps."""
    if not request.parameters:
        return [{}]

    separator = "," if request.separator is None else str(request.separator)
    if not separator:
        raise ValueError("--separator cannot be empty")

    parameter_specs: list[tuple[str, list[str]]] = []
    for raw_parameter in request.parameters:
        spec = str(raw_parameter).strip()
        if not spec:
            raise ValueError("parameter entries cannot be empty")
        if "=" not in spec:
            raise ValueError("parameter must be in NAME=VALUE format: %s" % spec)
        name, raw_values = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("parameter name cannot be empty")
        values = [item.strip() for item in raw_values.split(separator)]
        values = [item for item in values if item != ""]
        if not values:
            raise ValueError("parameter %s must include at least one value" % name)
        parameter_specs.append((name, values))

    names = [entry[0] for entry in parameter_specs]
    value_lists = [entry[1] for entry in parameter_specs]
    combinations: list[dict[str, str]] = []
    for selected in itertools.product(*value_lists):
        combinations.append(
            {str(name): str(value) for name, value in zip(names, selected)}
        )
    return combinations


def _apply_substitutions(text: str, substitutions: dict[str, str]) -> str:
    """Apply simple token replacement (e.g., @@name) for one sweep instance."""
    rendered = str(text)
    for key, value in substitutions.items():
        rendered = rendered.replace(key, value)
    return rendered


def _write_text_file(path: str, text: str) -> None:
    """Write UTF-8 text to disk, replacing existing files."""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(text)


def _discover_mount_device(path: str) -> str:
    """Infer mount device identifier compatible with legacy submit metadata."""
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
    """Collect file metadata fields expected by submit protocol context frames."""
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
    """Parse per-state counts from `parameterCombinations.csv` if present."""
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


def _normalize_parameter_state(text: str) -> str:
    """Normalize parameterCombinations state values to snake-case."""
    return str(text).strip().lower().replace(" ", "_")


def _parse_parameter_instance_states(parameter_path: Path) -> dict[int, str]:
    """Parse per-instance states from `parameterCombinations.csv`."""
    states: dict[int, str] = {}
    if not parameter_path.exists():
        return states

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
                index_text = row[0].strip()
                if not index_text.isdigit():
                    continue
                state_text = _normalize_parameter_state(row[1])
                if not state_text:
                    continue
                states[int(index_text)] = state_text
    except OSError:
        return states
    return states


def _infer_session_state(
    *,
    status_counts: dict[str, int],
    saw_start_marker: bool,
    saw_finish_marker: bool,
) -> str:
    """Infer overall run state from markers and parameter status counts."""
    if any(status_counts.get(state, 0) > 0 for state in _STATUS_RUNNING_STATES):
        return "running"
    if status_counts:
        total = sum(
            status_counts.get(state, 0)
            for state in (_STATUS_RUNNING_STATES | _STATUS_DONE_STATES)
        )
        if total <= 0:
            return "unknown"
        done = sum(status_counts.get(state, 0) for state in _STATUS_DONE_STATES)
        if done == total:
            return "complete"
        return "mixed"
    if saw_start_marker and not saw_finish_marker:
        return "running_maybe"
    if saw_finish_marker:
        return "complete_maybe"
    return "unknown"


def _parse_submit_progress_lines(text: str) -> list[dict[str, Any]]:
    """Extract submit progress snapshots from stdout text."""
    updates: list[dict[str, Any]] = []
    if not text:
        return updates

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _PROGRESS_LINE_RE.match(line)
        if not match:
            continue
        kv_pairs = dict(_PROGRESS_KV_RE.findall(match.group(1)))

        update: dict[str, Any] = {
            "raw_line": line,
            "aborted": 0,
            "finished": 0,
            "failed": 0,
            "executing": 0,
            "waiting": 0,
            "setup": 0,
            "setting_up": 0,
        }
        for key in (
            "aborted",
            "finished",
            "failed",
            "executing",
            "waiting",
            "setup",
            "setting_up",
        ):
            raw_value = kv_pairs.get(key)
            if raw_value is None:
                continue
            try:
                update[key] = int(raw_value)
            except (TypeError, ValueError):
                continue

        if "%done" in kv_pairs:
            try:
                update["percent_done"] = float(kv_pairs["%done"])
            except (TypeError, ValueError):
                pass
        if "timestamp" in kv_pairs:
            try:
                update["timestamp"] = float(kv_pairs["timestamp"])
            except (TypeError, ValueError):
                pass

        updates.append(update)
    return updates


def _progress_from_status_counts(status_counts: dict[str, int]) -> dict[str, Any] | None:
    """Build a progress-like snapshot from parameter status counters."""
    if not status_counts:
        return None

    total = sum(
        status_counts.get(state, 0) for state in (_STATUS_RUNNING_STATES | _STATUS_DONE_STATES)
    )
    if total <= 0:
        return None

    finished = int(status_counts.get("finished", 0))
    failed = int(status_counts.get("failed", 0))
    aborted = int(status_counts.get("aborted", 0))
    executing = int(status_counts.get("executing", 0))
    waiting = int(status_counts.get("waiting", 0))
    setup = int(status_counts.get("setup", 0))
    setting_up = int(status_counts.get("setting_up", 0))
    done = finished + failed + aborted
    percent_done = (100.0 * float(done) / float(total)) if total > 0 else 100.0
    return {
        "aborted": aborted,
        "finished": finished,
        "failed": failed,
        "executing": executing,
        "waiting": waiting,
        "setup": setup,
        "setting_up": setting_up,
        "percent_done": percent_done,
        "source": "parameterCombinations.csv",
    }


def _merge_progress_snapshot(
    current: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Pick the more informative progress snapshot between two candidates."""
    if current is None:
        return dict(candidate) if candidate is not None else None
    if candidate is None:
        return dict(current)

    current_percent = current.get("percent_done")
    candidate_percent = candidate.get("percent_done")
    if isinstance(candidate_percent, (int, float)):
        if not isinstance(current_percent, (int, float)):
            return dict(candidate)
        if float(candidate_percent) >= float(current_percent):
            return dict(candidate)
    return dict(current)


def _extract_message_ids(payload: Any) -> list[int]:
    """Extract integer IDs from nested message payloads."""
    extracted: list[int] = []

    if isinstance(payload, bool):
        return extracted

    if isinstance(payload, int):
        return [payload]

    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.isdigit():
            try:
                return [int(stripped)]
            except ValueError:
                return []
        return []

    if isinstance(payload, list):
        for item in payload:
            extracted.extend(_extract_message_ids(item))
        return extracted

    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key).strip().lower()
            if (
                key_text.endswith("id")
                or key_text.endswith("ids")
                or key_text in {"sessions", "createdsessions", "jobs", "runs"}
            ):
                extracted.extend(_extract_message_ids(value))
            elif isinstance(value, (dict, list)):
                extracted.extend(_extract_message_ids(value))
        return extracted

    return extracted


def _unique_ints(values: list[int]) -> list[int]:
    """Return stable-order unique integer list."""
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _derive_tracked_run_state(
    *,
    run: "SimulationRunRecord",
    local_session: "LocalSessionInfo | None",
    latest_progress: dict[str, Any] | None,
) -> str:
    """Infer current tracked-run state from run metadata and progress/session info."""
    if local_session is not None and local_session.inferred_state != "unknown":
        session_state = local_session.inferred_state
        if session_state in {"running", "running_maybe"}:
            return "running"
        if session_state in {"complete", "complete_maybe"}:
            return "failed" if run.returncode != 0 else "complete"
        return session_state

    if latest_progress:
        executing = int(latest_progress.get("executing", 0))
        waiting = int(latest_progress.get("waiting", 0))
        setup = int(latest_progress.get("setup", 0))
        setting_up = int(latest_progress.get("setting_up", 0))
        if executing > 0 or waiting > 0 or setup > 0 or setting_up > 0:
            return "running"

        percent_done = latest_progress.get("percent_done")
        if isinstance(percent_done, (int, float)) and float(percent_done) >= 100.0:
            return "failed" if run.returncode != 0 else "complete"

    if run.returncode != 0:
        return "failed"
    if run.local:
        return "complete"
    return "submitted"


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
    process_ids: list[int] = field(default_factory=list)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True when `returncode == 0`."""
        return self.returncode == 0


@dataclass
class ValidationCheck:
    """One validation/doctor finding with severity and message."""

    name: str
    ok: bool
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dictionary."""
        return {
            "name": self.name,
            "ok": self.ok,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class SubmitValidationResult:
    """Aggregate validation result for a submit request."""

    ok: bool
    args: list[str]
    checks: list[ValidationCheck]

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation result and all checks."""
        return {
            "ok": self.ok,
            "args": list(self.args),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class DoctorReport:
    """Diagnostic report covering config, credentials, and optional probe."""

    ok: bool
    checks: list[ValidationCheck]
    server_version: str | None = None
    server_capabilities: dict[str, bool] = field(default_factory=dict)
    connected_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize doctor report to plain dictionaries/lists."""
        return {
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "server_version": self.server_version,
            "server_capabilities": dict(self.server_capabilities),
            "connected_uri": self.connected_uri,
        }


@dataclass
class SessionStatusProbe:
    """Optional live probe for inferred job IDs via `submit --status`."""

    job_ids: list[int]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when no probe error and return code is successful."""
        return self.error is None and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize probe data including computed `ok` state."""
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
    """Filesystem-derived session summary discovered under a root directory."""

    path: str
    run_name: str
    inferred_job_ids: list[int]
    inferred_state: str
    status_counts: dict[str, int]
    marker_files: int
    last_updated: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize local session summary."""
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
    """Collection of discovered sessions with optional live status probe."""

    sessions: list[LocalSessionInfo]
    inferred_job_ids: list[int]
    live_probe: SessionStatusProbe | None = None

    @property
    def ok(self) -> bool:
        """True when live probe is absent or succeeded."""
        return self.live_probe is None or self.live_probe.ok

    def to_dict(self) -> dict[str, Any]:
        """Serialize session report and nested session/probe payloads."""
        return {
            "ok": self.ok,
            "sessions": [session.to_dict() for session in self.sessions],
            "inferred_job_ids": list(self.inferred_job_ids),
            "live_probe": self.live_probe.to_dict() if self.live_probe else None,
        }


@dataclass
class SimulationRunRecord:
    """In-memory history entry for one client `submit(...)` call."""

    sequence: int
    started_at: float
    finished_at: float
    local: bool
    command: str
    command_arguments: list[str]
    venues: list[str]
    manager: str | None
    returncode: int
    job_id: int | None
    run_name: str | None
    submitted_from: str
    expected_run_path: str | None = None
    progress_updates: list[dict[str, Any]] = field(default_factory=list)
    latest_progress: dict[str, Any] | None = None
    status_counts: dict[str, int] = field(default_factory=dict)
    process_ids: list[int] = field(default_factory=list)
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize run tracking record."""
        return {
            "sequence": self.sequence,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "local": self.local,
            "command": self.command,
            "command_arguments": list(self.command_arguments),
            "venues": list(self.venues),
            "manager": self.manager,
            "returncode": self.returncode,
            "job_id": self.job_id,
            "run_name": self.run_name,
            "submitted_from": self.submitted_from,
            "expected_run_path": self.expected_run_path,
            "progress_updates": [dict(item) for item in self.progress_updates],
            "latest_progress": (
                dict(self.latest_progress) if self.latest_progress is not None else None
            ),
            "status_counts": dict(self.status_counts),
            "process_ids": list(self.process_ids),
            "timed_out": self.timed_out,
        }


@dataclass
class TrackedRunStatus:
    """Merged run tracking item combining run history and discovered session info."""

    run: SimulationRunRecord
    local_session: LocalSessionInfo | None = None
    latest_progress: dict[str, Any] | None = None
    derived_state: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize merged tracked-run status payload."""
        return {
            "run": self.run.to_dict(),
            "local_session": (
                self.local_session.to_dict() if self.local_session is not None else None
            ),
            "latest_progress": (
                dict(self.latest_progress) if self.latest_progress is not None else None
            ),
            "derived_state": self.derived_state,
        }


@dataclass
class TrackedRunsReport:
    """Collection of tracked runs with optional live status probe."""

    runs: list[TrackedRunStatus]
    job_ids: list[int]
    live_probe: SessionStatusProbe | None = None

    @property
    def ok(self) -> bool:
        """True when live probe is absent or succeeded."""
        return self.live_probe is None or self.live_probe.ok

    def to_dict(self) -> dict[str, Any]:
        """Serialize monitor report with nested tracking payloads."""
        return {
            "ok": self.ok,
            "runs": [run.to_dict() for run in self.runs],
            "job_ids": list(self.job_ids),
            "live_probe": self.live_probe.to_dict() if self.live_probe else None,
        }


class CommandExecutionError(RuntimeError):
    """Raised when submit protocol execution fails."""


class AuthenticationError(CommandExecutionError):
    """Raised when server rejects signon/authentication."""


@dataclass
class _SessionState:
    """Mutable protocol state accumulated during one wire session."""

    client_id_hex: str
    action: str
    local_execution: bool
    authenticated: bool = False
    server_disconnected: bool = False
    exit_code: int | None = None
    server_id_hex: str | None = None
    server_version: str | None = None
    job_id: int | None = None
    process_ids: list[int] = field(default_factory=list)
    run_name: str | None = None
    stdout_chunks: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)
    server_messages: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False


class _NotebookProgressMonitor:
    """Best-effort Jupyter progress renderer for submit sweeps.

    This monitor is optional and enabled only when:
    - running in a Jupyter kernel
    - `ipywidgets` is importable
    """

    def __init__(
        self,
        *,
        run_name: str | None,
        poll_interval: float,
        max_instance_bars: int = 128,
    ) -> None:
        self.enabled = False
        self.run_name = run_name
        self.poll_interval = poll_interval if poll_interval > 0 else 2.0
        self.max_instance_bars = max_instance_bars

        self._widgets: Any = None
        self._display: Any = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

        self._run_path: str | None = (
            os.path.abspath(run_name) if run_name else None
        )
        self._parameter_path: Path | None = (
            Path(self._run_path) / "parameterCombinations.csv"
            if self._run_path is not None
            else None
        )
        self._last_instance_states: dict[int, str] = {}

        self._title: Any = None
        self._summary_label: Any = None
        self._overall_bar: Any = None
        self._counts_label: Any = None
        self._instances_note: Any = None
        self._instance_box: Any = None
        self._container: Any = None
        self._instance_bars: dict[int, Any] = {}
        self._instance_labels: dict[int, Any] = {}
        self._instance_rows: dict[int, Any] = {}

        if not _is_jupyter_environment():
            return
        try:
            import ipywidgets as widgets  # type: ignore
            from IPython.display import display  # type: ignore
        except Exception:
            return

        self._widgets = widgets
        self._display = display
        self.enabled = True

    def start(self) -> None:
        """Render initial widgets and start background state polling."""
        if not self.enabled:
            return

        widgets = self._widgets
        self._title = widgets.HTML(value="<b>nanohubsubmit live progress</b>")
        self._summary_label = widgets.Label(value="starting submit session...")
        self._overall_bar = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=100.0,
            description="%done",
            bar_style="",
            layout=widgets.Layout(width="99%"),
        )
        self._counts_label = widgets.Label(value="waiting for progress frames...")
        self._instances_note = widgets.Label(value="")
        self._instance_box = widgets.VBox([])
        self._container = widgets.VBox(
            [
                self._title,
                self._summary_label,
                self._overall_bar,
                self._counts_label,
                self._instances_note,
                self._instance_box,
            ]
        )
        self._display(self._container)

        if self._run_path:
            self._summary_label.value = "tracking run directory: " + self._run_path

        if self._parameter_path is not None:
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def on_server_message(self, message: dict[str, Any]) -> None:
        """Update UI from live submit server messages."""
        if not self.enabled:
            return

        message_type = message.get("messageType")
        if not isinstance(message_type, str):
            return

        if message_type == "jobId":
            job_id = message.get("jobId")
            if isinstance(job_id, int):
                self._set_summary("submit launched (job_id=%d)" % job_id)
            return

        if message_type == "writeStdout":
            text = message.get("text")
            if not isinstance(text, str) or not text:
                return
            updates = _parse_submit_progress_lines(text)
            for update in updates:
                self._apply_progress_update(update)
            return

        if message_type == "serverExit":
            exit_code = message.get("exitCode")
            if isinstance(exit_code, int):
                self._set_summary("submit server exited with code %d" % exit_code)
            return

    def finish(
        self,
        *,
        result: CommandResult | None,
        error: Exception | None,
    ) -> None:
        """Finalize widget state after submit returns or raises."""
        if not self.enabled:
            return

        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=min(1.0, self.poll_interval + 0.1))

        self._refresh_from_parameter_file()

        if error is not None:
            self._set_summary("submit failed: %s" % error)
            return
        if result is None:
            self._set_summary("submit ended without result")
            return

        if result.timed_out:
            self._set_summary(
                "submit timed out (job_id=%s)" % (result.job_id if result.job_id else "n/a")
            )
            return
        if result.returncode == 0:
            self._set_summary(
                "submit completed (job_id=%s)" % (result.job_id if result.job_id else "n/a")
            )
            if self._overall_bar is not None:
                self._overall_bar.value = 100.0
                self._overall_bar.bar_style = "success"
            return

        self._set_summary("submit exited with returncode=%d" % result.returncode)

    def _set_summary(self, text: str) -> None:
        if self._summary_label is None:
            return
        self._summary_label.value = text

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._refresh_from_parameter_file()
            self._stop_event.wait(self.poll_interval)

    def _refresh_from_parameter_file(self) -> None:
        if self._parameter_path is None:
            return
        states = _parse_parameter_instance_states(self._parameter_path)
        if not states:
            return
        if states == self._last_instance_states:
            return
        self._last_instance_states = dict(states)
        self._apply_instance_states(states)

    def _apply_progress_update(self, update: dict[str, Any]) -> None:
        if self._overall_bar is None or self._counts_label is None:
            return

        aborted = int(update.get("aborted", 0))
        finished = int(update.get("finished", 0))
        failed = int(update.get("failed", 0))
        executing = int(update.get("executing", 0))
        waiting = int(update.get("waiting", 0))
        setup = int(update.get("setup", 0))
        setting_up = int(update.get("setting_up", 0))
        total = aborted + finished + failed + executing + waiting + setup + setting_up
        done = aborted + finished + failed
        percent_done = update.get("percent_done")
        if not isinstance(percent_done, (int, float)):
            percent_done = (100.0 * float(done) / float(total)) if total > 0 else 0.0

        with self._lock:
            self._overall_bar.value = max(0.0, min(float(percent_done), 100.0))
            if failed > 0 or aborted > 0:
                self._overall_bar.bar_style = "danger"
            elif float(percent_done) >= 100.0:
                self._overall_bar.bar_style = "success"
            elif executing > 0:
                self._overall_bar.bar_style = "info"
            else:
                self._overall_bar.bar_style = ""
            self._counts_label.value = (
                "finished=%d failed=%d aborted=%d executing=%d waiting=%d setup=%d setting_up=%d"
                % (finished, failed, aborted, executing, waiting, setup, setting_up)
            )

    def _apply_instance_states(self, states: dict[int, str]) -> None:
        if self._instance_box is None or self._instances_note is None:
            return

        state_counts: dict[str, int] = {}
        for state in states.values():
            state_counts[state] = state_counts.get(state, 0) + 1

        progress = _progress_from_status_counts(state_counts)
        if progress is not None:
            self._apply_progress_update(progress)

        if len(states) > self.max_instance_bars:
            self._instances_note.value = (
                "per-instance bars suppressed (count=%d > max=%d)"
                % (len(states), self.max_instance_bars)
            )
            if self._instance_box.children:
                self._instance_box.children = tuple()
            return

        self._instances_note.value = "per-instance status (%d)" % len(states)
        widgets = self._widgets
        rows: list[Any] = []

        for index in sorted(states):
            state = states[index]
            bar = self._instance_bars.get(index)
            label = self._instance_labels.get(index)
            row = self._instance_rows.get(index)

            if bar is None:
                bar = widgets.IntProgress(
                    value=0,
                    min=0,
                    max=100,
                    description=str(index),
                    bar_style="",
                    layout=widgets.Layout(width="70%"),
                )
                label = widgets.Label(value="")
                row = widgets.HBox([bar, label])
                self._instance_bars[index] = bar
                self._instance_labels[index] = label
                self._instance_rows[index] = row

            value, bar_style = self._instance_bar_style(state)
            bar.value = value
            bar.bar_style = bar_style
            label.value = state
            rows.append(row)

        self._instance_box.children = tuple(rows)

    @staticmethod
    def _instance_bar_style(state: str) -> tuple[int, str]:
        normalized = _normalize_parameter_state(state)
        if normalized in {"finished"}:
            return (100, "success")
        if normalized in {"failed", "aborted"}:
            return (100, "danger")
        if normalized in {"executing"}:
            return (70, "info")
        if normalized in {"setup", "setting_up"}:
            return (35, "warning")
        if normalized in {"waiting"}:
            return (10, "")
        return (0, "")


# ---------------------------------------------------------------------------
# Main public client API.
# ---------------------------------------------------------------------------
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
        jupyter_auto_progress: bool = True,
        jupyter_progress_poll_interval: float = 2.0,
    ) -> None:
        """Initialize a submit client with optional config/auth overrides.

        Args map closely to legacy submit behavior:
        - config/network settings control wire connection setup.
        - auth settings override discovered credentials.
        - timing options control connect/poll/keepalive behavior.
        - jupyter options enable live progress widgets for submit sweeps.
        """
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
        self.jupyter_auto_progress = jupyter_auto_progress
        self.jupyter_progress_poll_interval = jupyter_progress_poll_interval
        self.client_version = _resolve_client_version()
        self._catalog_cache: "SubmitCatalog | None" = None
        self._run_history: list[SimulationRunRecord] = []
        self._run_sequence = 0

    def _verbose_log(self, message: str) -> None:
        """Emit verbose diagnostic lines when `verbose=True`."""
        if not self.verbose:
            return
        stream = self.verbose_stream if self.verbose_stream is not None else sys.stderr
        try:
            stream.write(f"[nanohubsubmit] {message}\n")
            stream.flush()
        except Exception:
            pass

    def _trace_message(self, direction: str, message: dict[str, Any]) -> None:
        """Emit compact, human-friendly traces for protocol messages."""
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

    def _maybe_create_notebook_progress_monitor(
        self, request: SubmitRequest
    ) -> _NotebookProgressMonitor | None:
        """Create Jupyter monitor for submit progress when available/enabled."""
        if not self.jupyter_auto_progress:
            return None
        if request.progress != ProgressMode.SUBMIT:
            return None
        if not _is_jupyter_environment():
            return None

        monitor = _NotebookProgressMonitor(
            run_name=request.run_name,
            poll_interval=self.jupyter_progress_poll_interval,
        )
        if not monitor.enabled:
            return None
        return monitor

    def submit(
        self, request: SubmitRequest, *, operation_timeout: float | None = None
    ) -> CommandResult:
        """Submit a command request.

        Local requests run through the local fast path. Remote requests use the
        submit wire protocol.
        """
        if request.run_name:
            run_path = os.path.abspath(request.run_name)
            if os.path.exists(run_path):
                raise CommandExecutionError(
                    "run_name '%s' is already in use at %s" % (request.run_name, run_path)
                )

        monitor = self._maybe_create_notebook_progress_monitor(request)
        if monitor is not None:
            monitor.start()

        started_at = time.time()
        submitted_from = os.getcwd()
        result: CommandResult | None = None
        error: Exception | None = None
        try:
            if request.local:
                result = self._run_local_submit(
                    request, operation_timeout=operation_timeout
                )
            else:
                args = CommandBuilder.build_submit_args(request)
                result = self._run(
                    action="submit",
                    args=args,
                    local_execution=request.local,
                    operation_timeout=operation_timeout,
                    progress_callback=(
                        monitor.on_server_message if monitor is not None else None
                    ),
                )
        except Exception as exc:
            error = exc
            raise
        finally:
            if monitor is not None:
                monitor.finish(result=result, error=error)

        if result is None:
            raise CommandExecutionError("submit ended without a result")

        self._record_submit_run(
            request=request,
            result=result,
            started_at=started_at,
            finished_at=time.time(),
            submitted_from=submitted_from,
        )
        return result

    def status(
        self, job_ids: list[int], *, operation_timeout: float | None = None
    ) -> CommandResult:
        """Run `submit --status` equivalent for one or more job IDs."""
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
        """Run `submit --kill` equivalent for one or more job IDs."""
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
        """Run `submit --venueStatus` with optional venue filters."""
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
        """Execute raw submit-style args through the wire protocol."""
        if not args:
            raise ValueError("args cannot be empty")
        return self._run(
            action="raw",
            args=args,
            local_execution=False,
            operation_timeout=operation_timeout,
        )

    def get_catalog(
        self,
        *,
        refresh: bool = False,
        include_raw_help: bool = False,
        operation_timeout: float = 60.0,
    ) -> "SubmitCatalog":
        """Load catalog on-demand and cache it for this client instance."""
        if not refresh and self._catalog_cache is not None:
            if include_raw_help and not self._catalog_cache.raw_help:
                pass
            else:
                return self._catalog_cache

        from .utils import load_available_catalog

        catalog = load_available_catalog(
            self,
            include_raw_help=include_raw_help,
            operation_timeout=operation_timeout,
        )
        self._catalog_cache = catalog
        return catalog

    def invalidate_catalog_cache(self) -> None:
        """Clear cached catalog so next `get_catalog` reloads from server."""
        self._catalog_cache = None

    def list_tracked_runs(self) -> list[SimulationRunRecord]:
        """Return immutable copy of in-memory submit run history."""
        return list(self._run_history)

    def clear_tracked_runs(self) -> None:
        """Clear tracked submit run history for current client instance."""
        self._run_history = []
        self._run_sequence = 0

    def monitor_tracked_runs(
        self,
        *,
        root: str | Path = ".",
        max_depth: int = 3,
        include_live_status: bool = False,
        operation_timeout: float | None = None,
    ) -> TrackedRunsReport:
        """Merge tracked submit history with local session discovery.

        This helps monitor long-running parameter sweeps by combining:
        - in-memory submit calls made by this client
        - discovered run directories and `parameterCombinations.csv` status
        - optional live `submit --status` for tracked job IDs
        """
        sessions_report = self.list_sessions(
            root=root,
            max_depth=max_depth,
            include_live_status=False,
            limit=None,
        )
        sessions_by_path: dict[str, LocalSessionInfo] = {
            os.path.abspath(session.path): session for session in sessions_report.sessions
        }
        sessions_by_name: dict[str, list[LocalSessionInfo]] = {}
        for session in sessions_report.sessions:
            sessions_by_name.setdefault(session.run_name, []).append(session)
        for candidates in sessions_by_name.values():
            candidates.sort(key=lambda item: item.last_updated, reverse=True)

        runs: list[TrackedRunStatus] = []
        tracked_job_ids: set[int] = set()

        for run in self._run_history:
            local_session: LocalSessionInfo | None = None
            if run.expected_run_path:
                local_session = sessions_by_path.get(os.path.abspath(run.expected_run_path))
            if local_session is None and run.run_name:
                run_name_candidates = sessions_by_name.get(run.run_name, [])
                if run_name_candidates:
                    local_session = run_name_candidates[0]

            latest_progress = (
                dict(run.latest_progress) if run.latest_progress is not None else None
            )
            if local_session is not None:
                local_progress = _progress_from_status_counts(local_session.status_counts)
                if local_progress is not None:
                    local_progress["timestamp"] = local_session.last_updated
                latest_progress = _merge_progress_snapshot(latest_progress, local_progress)

            runs.append(
                TrackedRunStatus(
                    run=run,
                    local_session=local_session,
                    latest_progress=latest_progress,
                    derived_state=_derive_tracked_run_state(
                        run=run,
                        local_session=local_session,
                        latest_progress=latest_progress,
                    ),
                )
            )
            if run.process_ids:
                for process_id in run.process_ids:
                    tracked_job_ids.add(process_id)
            elif run.job_id is not None:
                tracked_job_ids.add(run.job_id)

        sorted_job_ids = sorted(tracked_job_ids)
        live_probe: SessionStatusProbe | None = None
        if include_live_status:
            if sorted_job_ids:
                try:
                    status_result = self.status(
                        sorted_job_ids,
                        operation_timeout=operation_timeout,
                    )
                except Exception as exc:
                    live_probe = SessionStatusProbe(
                        job_ids=sorted_job_ids,
                        error=str(exc),
                    )
                else:
                    live_probe = SessionStatusProbe(
                        job_ids=sorted_job_ids,
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

        return TrackedRunsReport(
            runs=runs,
            job_ids=sorted_job_ids,
            live_probe=live_probe,
        )

    def validate_submit_request(
        self,
        request: SubmitRequest,
        *,
        check_catalog: bool = False,
        operation_timeout: float = 60.0,
        extra_existing_paths: list[str] | None = None,
    ) -> SubmitValidationResult:
        """Run local preflight checks for a submit request.

        Checks include command shape, path existence, compatibility warnings,
        and optional server catalog lookups.
        """
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

            run_path = os.path.abspath(request.run_name)
            if os.path.exists(run_path):
                checks.append(
                    ValidationCheck(
                        name="run-name-availability",
                        ok=False,
                        severity="error",
                        message=(
                            "run_name is already in use (path exists): "
                            + run_path
                        ),
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="run-name-availability",
                        ok=True,
                        severity="info",
                        message="run_name path is available: " + run_path,
                    )
                )

        if request.local:
            command_path_like = (
                request.command.startswith(".")
                or request.command.startswith("~")
                or os.sep in request.command
                or (os.altsep is not None and os.altsep in request.command)
            )
            if command_path_like:
                expanded_command = os.path.expanduser(request.command)
                if os.path.exists(expanded_command):
                    checks.append(
                        ValidationCheck(
                            name="command-local-path",
                            ok=True,
                            severity="info",
                            message=f"local command path exists: {request.command}",
                        )
                    )
                else:
                    checks.append(
                        ValidationCheck(
                            name="command-local-path",
                            ok=False,
                            severity="error",
                            message=f"local command path does not exist: {request.command}",
                        )
                    )
            else:
                command_on_path = shutil.which(request.command) is not None
                checks.append(
                    ValidationCheck(
                        name="command-local-which",
                        ok=command_on_path,
                        severity="info" if command_on_path else "error",
                        message=(
                            f"local command is available on PATH: {request.command}"
                            if command_on_path
                            else f"local command not found on PATH: {request.command}"
                        ),
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
                if os.path.isdir(request.data_file):
                    checks.append(
                        ValidationCheck(
                            name="data-file-type",
                            ok=False,
                            severity="error",
                            message=f"data file must be a regular file: {request.data_file}",
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

        if extra_existing_paths:
            for check_path in extra_existing_paths:
                expanded_path = os.path.expanduser(check_path)
                if os.path.exists(expanded_path):
                    checks.append(
                        ValidationCheck(
                            name="extra-path",
                            ok=True,
                            severity="info",
                            message=f"extra path exists: {check_path}",
                        )
                    )
                else:
                    checks.append(
                        ValidationCheck(
                            name="extra-path",
                            ok=False,
                            severity="error",
                            message=f"extra path does not exist: {check_path}",
                        )
                    )

        if check_catalog:
            if request.local:
                checks.append(
                    ValidationCheck(
                        name="catalog-local-skip",
                        ok=True,
                        severity="info",
                        message="catalog validation skipped for local execution",
                    )
                )
            else:
                try:
                    catalog = self.get_catalog(
                        operation_timeout=operation_timeout
                    )
                except Exception as exc:
                    checks.append(
                        ValidationCheck(
                            name="catalog-load",
                            ok=False,
                            severity="error",
                            message=f"failed to load server catalog: {exc}",
                        )
                    )
                else:
                    if catalog.contains("tools", request.command):
                        checks.append(
                            ValidationCheck(
                                name="catalog-tool",
                                ok=True,
                                severity="info",
                                message=f"tool exists in catalog: {request.command}",
                            )
                        )
                    else:
                        tool_matches = catalog.filter(
                            request.command, details=["tools"], limit=5
                        )["tools"]
                        suggestion_text = (
                            ("; close matches: " + ", ".join(tool_matches))
                            if tool_matches
                            else ""
                        )
                        checks.append(
                            ValidationCheck(
                                name="catalog-tool",
                                ok=False,
                                severity="error",
                                message=(
                                    f"tool not found in catalog: {request.command}"
                                    + suggestion_text
                                ),
                            )
                        )

                    if request.manager:
                        manager_ok = catalog.contains("managers", request.manager)
                        checks.append(
                            ValidationCheck(
                                name="catalog-manager",
                                ok=manager_ok,
                                severity="info" if manager_ok else "error",
                                message=(
                                    f"manager exists in catalog: {request.manager}"
                                    if manager_ok
                                    else f"manager not found in catalog: {request.manager}"
                                ),
                            )
                        )

                    for venue in request.venues:
                        venue_ok = catalog.contains("venues", venue)
                        checks.append(
                            ValidationCheck(
                                name="catalog-venue",
                                ok=venue_ok,
                                severity="info" if venue_ok else "error",
                                message=(
                                    f"venue exists in catalog: {venue}"
                                    if venue_ok
                                    else f"venue not found in catalog: {venue}"
                                ),
                            )
                        )

        ok = all(check.severity != "error" for check in checks)
        return SubmitValidationResult(ok=ok, args=args, checks=checks)

    def preflight_submit_request(
        self,
        request: SubmitRequest,
        *,
        operation_timeout: float = 60.0,
        extra_existing_paths: list[str] | None = None,
    ) -> SubmitValidationResult:
        """Convenience wrapper over `validate_submit_request`.

        Catalog checks are enabled automatically for non-local requests.
        """
        return self.validate_submit_request(
            request,
            check_catalog=not request.local,
            operation_timeout=operation_timeout,
            extra_existing_paths=extra_existing_paths,
        )

    def _record_submit_run(
        self,
        *,
        request: SubmitRequest,
        result: CommandResult,
        started_at: float,
        finished_at: float,
        submitted_from: str,
    ) -> None:
        """Append one run-history record after every submit invocation."""
        self._run_sequence += 1
        expected_run_path: str | None = None
        if request.run_name:
            expected_run_path = os.path.abspath(
                os.path.join(submitted_from, request.run_name)
            )

        progress_updates = _parse_submit_progress_lines(result.stdout)
        latest_progress = progress_updates[-1] if progress_updates else None
        status_counts: dict[str, int] = {}
        process_ids: list[int] = []

        if expected_run_path:
            parameter_path = Path(expected_run_path) / "parameterCombinations.csv"
            status_counts = _parse_parameter_status_counts(parameter_path)
            file_progress = _progress_from_status_counts(status_counts)
            if file_progress is not None:
                file_progress["timestamp"] = finished_at
            latest_progress = _merge_progress_snapshot(latest_progress, file_progress)

        if result.process_ids:
            process_ids.extend(result.process_ids)
        if result.job_id is not None:
            process_ids.append(result.job_id)
        process_ids = _unique_ints(process_ids)

        self._run_history.append(
            SimulationRunRecord(
                sequence=self._run_sequence,
                started_at=started_at,
                finished_at=finished_at,
                local=request.local,
                command=request.command,
                command_arguments=list(request.command_arguments),
                venues=list(request.venues),
                manager=request.manager,
                returncode=result.returncode,
                job_id=result.job_id,
                run_name=result.run_name,
                submitted_from=submitted_from,
                expected_run_path=expected_run_path,
                progress_updates=[dict(item) for item in progress_updates],
                latest_progress=(
                    dict(latest_progress) if latest_progress is not None else None
                ),
                status_counts=status_counts,
                process_ids=process_ids,
                timed_out=result.timed_out,
            )
        )

    def doctor(self, *, probe_server: bool = True) -> DoctorReport:
        """Run local diagnostics and optional live server auth probe."""
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
        """Discover local run directories and infer session/job state.

        This scans a filesystem tree for known submit marker files and optional
        `parameterCombinations.csv` status content.
        """
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
        """Load and apply runtime config overrides."""
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
        """Load credentials and enforce at least one usable auth mode."""
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
        """Open lightweight probe session to validate auth/capabilities."""
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

    def _run_local_submit(
        self, request: SubmitRequest, *, operation_timeout: float | None = None
    ) -> CommandResult:
        """Execute local requests directly, including parameter sweeps.

        This path emulates submit's local behavior with optional progress
        rendering and run directory artifact generation.
        """
        parameter_combinations = _expand_local_parameter_sweep(request)
        effective_timeout = (
            self.operation_timeout if operation_timeout is None else operation_timeout
        )
        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("operation_timeout must be positive when provided")

        progress_submit = request.progress == ProgressMode.SUBMIT
        total_runs = len(parameter_combinations)
        finished = 0
        failed = 0
        aborted = 0
        started_runs = 0

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        last_returncode = 0
        started_at = time.time()
        run_name = request.run_name
        run_path: str | None = None
        per_instance = total_runs > 1
        parameter_names = (
            list(parameter_combinations[0].keys()) if parameter_combinations else []
        )

        if run_name:
            run_path = os.path.abspath(run_name)
            try:
                os.makedirs(run_path, exist_ok=True)
                synthetic_job_id = str(int(time.time()))
                metadata_lines = [
                    "run_name: %s" % run_name,
                    "job_id: %s" % synthetic_job_id,
                    "mode: local-fast-path",
                    "created_at: %.1f" % started_at,
                    "command: %s" % request.command,
                    "arguments: %s" % " ".join(list(request.command_arguments)),
                ]
                _write_text_file(
                    os.path.join(run_path, "%s.yml" % synthetic_job_id),
                    "\n".join(metadata_lines) + "\n",
                )
                if per_instance:
                    with open(
                        os.path.join(run_path, "parameterCombinations.csv"),
                        "w",
                        newline="",
                        encoding="utf-8",
                    ) as fp:
                        writer = csv.writer(fp)
                        writer.writerow(["instance"] + list(parameter_names))
                        for instance_index, substitutions in enumerate(
                            parameter_combinations, start=1
                        ):
                            writer.writerow(
                                [instance_index]
                                + [
                                    substitutions.get(parameter_name, "")
                                    for parameter_name in parameter_names
                                ]
                            )
            except OSError as exc:
                raise CommandExecutionError(str(exc)) from exc

        self._verbose_log(
            "starting local-fast-path command=%s sweeps=%d"
            % (
                " ".join([request.command, *list(request.command_arguments)]),
                total_runs,
            )
        )
        if progress_submit:
            stdout_chunks.append(
                _format_submit_progress_line(
                    finished=finished,
                    failed=failed,
                    aborted=aborted,
                    executing=0,
                    setup=max(total_runs - started_runs, 0),
                    setting_up=0,
                    total=total_runs,
                )
                + "\n"
            )

        for instance_index, substitutions in enumerate(parameter_combinations, start=1):
            command = [
                _apply_substitutions(request.command, substitutions),
                *[
                    _apply_substitutions(arg, substitutions)
                    for arg in list(request.command_arguments)
                ],
            ]
            env = os.environ.copy()
            for key, value in request.environment.items():
                if value is None:
                    continue
                env[str(key)] = _apply_substitutions(str(value), substitutions)

            stdout_file_path: str | None = None
            stderr_file_path: str | None = None
            if run_path:
                if per_instance:
                    n_digits = max(2, len(str(total_runs)))
                    instance_id = str(instance_index).zfill(n_digits)
                    instance_dir = os.path.join(run_path, instance_id)
                    try:
                        os.makedirs(instance_dir, exist_ok=True)
                    except OSError as exc:
                        raise CommandExecutionError(str(exc)) from exc
                    stdout_file_path = os.path.join(
                        instance_dir, "%s_%s.stdout" % (run_name, instance_id)
                    )
                    stderr_file_path = os.path.join(
                        instance_dir, "%s_%s.stderr" % (run_name, instance_id)
                    )
                else:
                    stdout_file_path = os.path.join(run_path, "%s.stdout" % run_name)
                    stderr_file_path = os.path.join(run_path, "%s.stderr" % run_name)

            started_runs += 1
            if progress_submit:
                stdout_chunks.append(
                    _format_submit_progress_line(
                        finished=finished,
                        failed=failed,
                        aborted=aborted,
                        executing=1,
                        setup=max(total_runs - started_runs, 0),
                        setting_up=0,
                        total=total_runs,
                    )
                    + "\n"
                )

            timeout_for_run: float | None = None
            if effective_timeout is not None:
                elapsed = time.time() - started_at
                remaining = effective_timeout - elapsed
                if remaining <= 0:
                    raise CommandExecutionError(
                        "timed out waiting for local command after %.1fs"
                        % effective_timeout
                    )
                timeout_for_run = remaining

            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=timeout_for_run,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                timeout = effective_timeout if effective_timeout is not None else 0.0
                raise CommandExecutionError(
                    "timed out waiting for local command after %.1fs" % timeout
                ) from exc
            except OSError as exc:
                raise CommandExecutionError(str(exc)) from exc

            if completed.stdout:
                stdout_chunks.append(completed.stdout)
            if completed.stderr:
                stderr_chunks.append(completed.stderr)

            if stdout_file_path:
                try:
                    _write_text_file(stdout_file_path, completed.stdout)
                except OSError as exc:
                    raise CommandExecutionError(str(exc)) from exc
            if stderr_file_path:
                try:
                    if completed.stderr:
                        _write_text_file(stderr_file_path, completed.stderr)
                    elif os.path.exists(stderr_file_path):
                        os.remove(stderr_file_path)
                except OSError as exc:
                    raise CommandExecutionError(str(exc)) from exc

            last_returncode = int(completed.returncode)
            if last_returncode == 0:
                finished += 1
            else:
                failed += 1

            if progress_submit:
                stdout_chunks.append(
                    _format_submit_progress_line(
                        finished=finished,
                        failed=failed,
                        aborted=aborted,
                        executing=0,
                        setup=max(total_runs - started_runs, 0),
                        setting_up=0,
                        total=total_runs,
                    )
                    + "\n"
                )

        result = CommandResult(
            args=["submit", "--local", request.command, *list(request.command_arguments)],
            returncode=0 if (failed == 0 and aborted == 0) else (last_returncode or 1),
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            authenticated=False,
            server_disconnected=False,
            server_messages=[],
            job_id=None,
            run_name=request.run_name,
            server_version=None,
        )
        self._verbose_log(
            "finished local-fast-path returncode=%d" % result.returncode
        )
        if self.check and not result.ok:
            raise CommandExecutionError(
                "local command exited with code %d: %s"
                % (result.returncode, result.stderr.strip())
            )
        return result

    def _run(
        self,
        *,
        action: str,
        args: list[str],
        local_execution: bool,
        operation_timeout: float | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> CommandResult:
        """Execute one full wire-protocol submit action and collect result."""
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
                    timeout_message = "timed out waiting for %s response after %.1fs" % (
                        action,
                        effective_timeout,
                    )
                    launched_submit = (
                        action == "submit"
                        and (
                            state.job_id is not None
                            or bool(state.process_ids)
                            or state.run_name is not None
                        )
                    )
                    if launched_submit:
                        state.timed_out = True
                        state.exit_code = 124
                        state.stderr_chunks.append(timeout_message + "\n")
                        state.server_messages.append(
                            {
                                "messageType": "timeout",
                                "text": timeout_message,
                            }
                        )
                        self._verbose_log(
                            "timeout reached after launch; returning partial submit result"
                        )
                        break
                    raise CommandExecutionError(timeout_message)

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
                if progress_callback is not None:
                    try:
                        progress_callback(server_message)
                    except Exception:
                        pass
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
            process_ids=_unique_ints(list(state.process_ids)),
            timed_out=state.timed_out,
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
        """Send mandatory client execution context frames to the server.

        Returns a temporary signature file path that must be cleaned up by the
        caller once the session ends.
        """
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
        """Process one server frame and return any immediate client responses."""
        message_type = message.get("messageType")
        if not isinstance(message_type, str):
            return []

        # Session metadata frames.
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

        # Stream forwarding frames.
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

        # Job identity frames.
        if message_type == "jobId":
            job_id = message.get("jobId")
            if isinstance(job_id, int):
                state.job_id = job_id
                state.process_ids = _unique_ints([*state.process_ids, job_id])
            return []

        if message_type == "runName":
            run_name = message.get("runName")
            if isinstance(run_name, str):
                state.run_name = run_name
            job_id = message.get("jobId")
            if isinstance(job_id, int):
                state.job_id = job_id
                state.process_ids = _unique_ints([*state.process_ids, job_id])
            return []

        if message_type == "createdSessions":
            state.process_ids = _unique_ints(
                [*state.process_ids, *_extract_message_ids(message)]
            )
            return []

        # Authentication handshake.
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

        # Execution setup pipeline.
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

        # Runtime I/O and lifecycle transitions.
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

        # Terminal frames.
        if message_type in {"serverExit", "exit"}:
            exit_code = message.get("exitCode")
            if not isinstance(exit_code, int):
                exit_code = 1
            state.exit_code = exit_code
            return []

        if message_type == "null":
            return []

        return []
