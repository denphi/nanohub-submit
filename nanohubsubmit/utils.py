from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .client import CommandExecutionError, NanoHUBSubmitClient

_LIST_DETAIL_TO_ARGS = {
    "tools": ["--help", "tools"],
    "venues": ["--help", "venues"],
    "managers": ["--help", "managers"],
}

_ITEM_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+\-]*$")
_VENUE_KV_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")
_SECTION_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _/\-]{0,40}:$")

_LINE_STOPWORDS = {
    "usage",
    "options",
    "optional",
    "positional",
    "arguments",
    "available",
    "managers",
    "manager",
    "venues",
    "venue",
    "tools",
    "tool",
    "examples",
    "submit",
}


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _combined_text(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return stdout.rstrip("\n") + "\n" + stderr
    return stdout or stderr or ""


def _maybe_parse_json_items(text: str, detail: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []

    if isinstance(payload, list):
        return [str(entry) for entry in payload if isinstance(entry, (str, int, float))]
    if isinstance(payload, dict):
        raw_items = payload.get(detail)
        if isinstance(raw_items, list):
            return [str(entry) for entry in raw_items if isinstance(entry, (str, int, float))]
    return []


def parse_help_items(text: str) -> List[str]:
    """
    Parse manager/tool/venue names from free-form help output.

    The submit server output format can vary by deployment, so this parser uses
    resilient heuristics and keeps unknown lines untouched at call sites.
    """
    if not text:
        return []

    lines = text.splitlines()
    parsed: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith("usage:"):
            continue
        if lower.startswith("options:"):
            continue
        if lower.startswith("positional arguments:"):
            continue
        if lower.startswith("optional arguments:"):
            continue
        if _SECTION_LINE_RE.match(stripped):
            continue
        if stripped.startswith("--"):
            continue

        normalized = stripped.lstrip("-*+ ").strip()
        if not normalized:
            continue
        if normalized.startswith("--"):
            continue
        if normalized.startswith("error:"):
            continue

        token = normalized.split()[0].rstrip(":,;")
        if not token:
            continue

        token_lower = token.lower()
        if token_lower in _LINE_STOPWORDS:
            continue
        if not _ITEM_TOKEN_RE.match(token):
            continue
        parsed.append(token)

    return _dedupe_keep_order(parsed)


def parse_venue_status(text: str) -> List[Dict[str, str]]:
    """
    Parse venue status output into structured dictionaries.
    """
    entries: List[Dict[str, str]] = []
    if not text:
        return entries

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("usage:"):
            continue

        kv_pairs = dict(_VENUE_KV_RE.findall(stripped))
        if kv_pairs:
            entries.append({str(k): str(v) for k, v in kv_pairs.items()})
            continue

        if ":" in stripped:
            left, right = stripped.split(":", 1)
            venue = left.strip()
            status = right.strip()
            if venue and status:
                entries.append({"venue": venue, "status": status})
    return entries


def load_available_list(
    client: NanoHUBSubmitClient, detail: str, operation_timeout: float = 20.0
) -> List[str]:
    """
    Query submit server for one list detail: tools, venues, or managers.
    """
    if detail not in _LIST_DETAIL_TO_ARGS:
        raise ValueError("detail must be one of: tools, venues, managers")

    result = client.raw(_LIST_DETAIL_TO_ARGS[detail], operation_timeout=operation_timeout)
    text = _combined_text(result.stdout, result.stderr)
    if result.returncode != 0 and not text.strip():
        raise CommandExecutionError(
            "failed to load %s from submit server (exit=%d)"
            % (detail, result.returncode)
        )

    json_items = _maybe_parse_json_items(text, detail)
    if json_items:
        return _dedupe_keep_order(json_items)
    return parse_help_items(text)


def load_available_tools(
    client: NanoHUBSubmitClient, operation_timeout: float = 20.0
) -> List[str]:
    return load_available_list(client, "tools", operation_timeout=operation_timeout)


def load_available_venues(
    client: NanoHUBSubmitClient, operation_timeout: float = 20.0
) -> List[str]:
    return load_available_list(client, "venues", operation_timeout=operation_timeout)


def load_available_managers(
    client: NanoHUBSubmitClient, operation_timeout: float = 20.0
) -> List[str]:
    return load_available_list(client, "managers", operation_timeout=operation_timeout)


@dataclass
class SubmitCatalog:
    tools: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    managers: List[str] = field(default_factory=list)
    raw_help: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tools": list(self.tools),
            "venues": list(self.venues),
            "managers": list(self.managers),
            "raw_help": dict(self.raw_help),
        }


def load_available_catalog(
    client: NanoHUBSubmitClient,
    include_raw_help: bool = False,
    operation_timeout: float = 20.0,
) -> SubmitCatalog:
    raw_help: Dict[str, str] = {}

    def _load(detail: str) -> List[str]:
        args = _LIST_DETAIL_TO_ARGS[detail]
        result = client.raw(args, operation_timeout=operation_timeout)
        text = _combined_text(result.stdout, result.stderr)
        if include_raw_help:
            raw_help[detail] = text
        json_items = _maybe_parse_json_items(text, detail)
        if json_items:
            return _dedupe_keep_order(json_items)
        return parse_help_items(text)

    return SubmitCatalog(
        tools=_load("tools"),
        venues=_load("venues"),
        managers=_load("managers"),
        raw_help=raw_help,
    )


@dataclass
class SubmitServerExploration:
    doctor: Dict[str, Any]
    catalog: SubmitCatalog
    venue_status_raw: str
    venue_status: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doctor": dict(self.doctor),
            "catalog": self.catalog.to_dict(),
            "venue_status_raw": self.venue_status_raw,
            "venue_status": list(self.venue_status),
        }


def explore_submit_server(
    client: NanoHUBSubmitClient,
    *,
    include_raw_help: bool = False,
    include_venue_status: bool = True,
    operation_timeout: float = 20.0,
) -> SubmitServerExploration:
    """
    Collect as much introspection data as possible from submit server.
    """
    doctor_report = client.doctor(probe_server=True).to_dict()
    catalog = load_available_catalog(
        client,
        include_raw_help=include_raw_help,
        operation_timeout=operation_timeout,
    )

    venue_status_raw = ""
    venue_status: List[Dict[str, str]] = []
    if include_venue_status:
        venue_result = client.venue_status(operation_timeout=operation_timeout)
        venue_status_raw = _combined_text(venue_result.stdout, venue_result.stderr)
        venue_status = parse_venue_status(venue_status_raw)

    return SubmitServerExploration(
        doctor=doctor_report,
        catalog=catalog,
        venue_status_raw=venue_status_raw,
        venue_status=venue_status,
    )
