from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanohubsubmit import NanoHUBSubmitClient, ProgressMode, SubmitRequest
from nanohubsubmit.utils import explore_submit_server, load_available_catalog


def _live_enabled() -> bool:
    value = os.environ.get("NANOHUBSUBMIT_LIVE", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _live_config_path() -> str | None:
    value = os.environ.get("NANOHUBSUBMIT_CONFIG_PATH", "").strip()
    return value or None


def _make_live_client(local_fast_path: bool = True) -> NanoHUBSubmitClient:
    config_path = _live_config_path()
    if config_path:
        return NanoHUBSubmitClient(
            config_path=config_path,
            verbose=True,
            local_fast_path=local_fast_path,
        )
    return NanoHUBSubmitClient(verbose=True, local_fast_path=local_fast_path)


pytestmark = pytest.mark.skipif(
    not _live_enabled(),
    reason=(
        "live submit-server tests are disabled. "
        "Set NANOHUBSUBMIT_LIVE=1 to enable."
    ),
)


def test_live_doctor_probe_is_ok() -> None:
    client = _make_live_client()
    doctor = client.doctor(probe_server=True).to_dict()
    assert doctor["ok"] is True, doctor
    assert doctor["server_version"], doctor


def test_live_catalog_has_expected_lists() -> None:
    client = _make_live_client()
    catalog = load_available_catalog(client, operation_timeout=90.0)
    assert isinstance(catalog.tools, list)
    assert isinstance(catalog.venues, list)
    assert isinstance(catalog.managers, list)
    assert len(catalog.tools) > 0


def test_live_explore_collects_doctor_and_catalog() -> None:
    client = _make_live_client()
    payload = explore_submit_server(client, operation_timeout=90.0).to_dict()
    assert payload["doctor"]["ok"] is True, payload["doctor"]
    assert "catalog" in payload
    assert "tools" in payload["catalog"]


def test_live_submit_local_ls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = _make_live_client()
    result = client.submit(
        SubmitRequest(command="ls", local=True),
        operation_timeout=120.0,
    )
    assert result.returncode == 0, result.to_dict()


def test_live_submit_local_parameter_sweep_submit_progress() -> None:
    client = _make_live_client()
    result = client.submit(
        SubmitRequest(
            command="echo",
            command_arguments=["@@name"],
            local=True,
            run_name="echotest",
            separator=",",
            parameters=["@@name=hub1,hub2,hub3"],
            progress=ProgressMode.SUBMIT,
        ),
        operation_timeout=180.0,
    )
    assert result.returncode == 0, result.to_dict()
    progress_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("=SUBMIT-PROGRESS=>")
    ]
    assert progress_lines, result.to_dict()
    assert any("%done=100.00" in line for line in progress_lines), result.to_dict()
