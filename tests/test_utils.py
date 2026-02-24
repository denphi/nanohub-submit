from __future__ import annotations

from typing import Any, Dict, List

from nanohubsubmit.client import CommandResult
from nanohubsubmit.utils import (
    explore_submit_server,
    load_available_catalog,
    parse_help_items,
    parse_venue_status,
)


class _FakeDoctor:
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "server_version": "1.2.3",
            "server_capabilities": {"hasJobStatus": True, "hasVenueProbe": True},
        }


class _FakeClient:
    def __init__(self) -> None:
        self.raw_calls: List[List[str]] = []
        self.doctor_calls: List[bool] = []

    def raw(self, args: List[str]) -> CommandResult:
        self.raw_calls.append(list(args))
        key = tuple(args)
        payloads = {
            ("--help", "tools"): "Available tools:\n  abacus - desc\n  gem5 - desc\n",
            ("--help", "venues"): "Venues:\n  workspace : ready\n  community : ready\n",
            ("--help", "managers"): "Managers:\n  pegasus - default\n  local - local mode\n",
        }
        text = payloads.get(key, "")
        return CommandResult(
            args=["submit"] + list(args),
            returncode=0,
            stdout=text,
            stderr="",
            authenticated=True,
            server_disconnected=False,
        )

    def doctor(self, probe_server: bool = True) -> _FakeDoctor:
        self.doctor_calls.append(probe_server)
        return _FakeDoctor()

    def venue_status(self, venues=None) -> CommandResult:  # noqa: ANN001
        return CommandResult(
            args=["submit", "--venueStatus"],
            returncode=0,
            stdout="venue=workspace status=up\nvenue=community status=down\n",
            stderr="",
            authenticated=True,
            server_disconnected=False,
        )


def test_parse_help_items_extracts_names() -> None:
    text = """
usage: submit [options]

Available tools:
  abacus - Tool A
  gem5 - Tool B
options:
  --help
"""
    assert parse_help_items(text) == ["abacus", "gem5"]


def test_parse_venue_status_extracts_key_values() -> None:
    text = "venue=workspace status=up\nvenue=community status=down\n"
    assert parse_venue_status(text) == [
        {"venue": "workspace", "status": "up"},
        {"venue": "community", "status": "down"},
    ]


def test_load_available_catalog_queries_all_lists() -> None:
    client = _FakeClient()
    catalog = load_available_catalog(client, include_raw_help=False)
    assert catalog.tools == ["abacus", "gem5"]
    assert catalog.venues == ["workspace", "community"]
    assert catalog.managers == ["pegasus", "local"]
    assert client.raw_calls == [
        ["--help", "tools"],
        ["--help", "venues"],
        ["--help", "managers"],
    ]


def test_explore_submit_server_collects_doctor_catalog_and_venue_status() -> None:
    client = _FakeClient()
    exploration = explore_submit_server(
        client, include_raw_help=False, include_venue_status=True
    )
    payload = exploration.to_dict()
    assert payload["doctor"]["ok"] is True
    assert payload["catalog"]["tools"] == ["abacus", "gem5"]
    assert payload["venue_status"] == [
        {"venue": "workspace", "status": "up"},
        {"venue": "community", "status": "down"},
    ]
    assert client.doctor_calls == [True]
