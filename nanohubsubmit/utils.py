from __future__ import annotations

"""Utility helpers for submit metadata discovery and parsing.

These functions provide high-level catalog/exploration primitives built on top
of `NanoHUBSubmitClient.raw(...)` responses.
"""

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

_PROTOCOL_NOISE_SUBSTRINGS = (
    "successfully authenticated",
    "authentication failed",
)


_CATALOG_DETAILS = ("tools", "venues", "managers")


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    """Deduplicate while preserving original item order."""
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _combined_text(stdout: str, stderr: str) -> str:
    """Combine stdout/stderr streams into a single parse target."""
    if stdout and stderr:
        return stdout.rstrip("\n") + "\n" + stderr
    return stdout or stderr or ""


def _maybe_parse_json_items(text: str, detail: str) -> List[str]:
    """Parse list payloads when server/help output is JSON formatted."""
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
        if any(noise in lower for noise in _PROTOCOL_NOISE_SUBSTRINGS):
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


def _filter_names(
    values: Iterable[str],
    query: str,
    *,
    exact: bool = False,
    case_sensitive: bool = False,
    use_regex: bool = False,
    limit: int | None = None,
) -> List[str]:
    """Filter names using substring/exact/regex matching semantics."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    query = query or ""

    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(query, flags)
    else:
        regex = None
        query_cmp = query if case_sensitive else query.lower()

    filtered: List[str] = []
    for value in values:
        text = str(value)
        text_cmp = text if case_sensitive else text.lower()
        matched = False
        if regex is not None:
            matched = bool(regex.search(text))
        elif exact:
            matched = text_cmp == query_cmp
        else:
            matched = query_cmp in text_cmp

        if matched:
            filtered.append(text)
            if limit is not None and len(filtered) >= limit:
                break
    return filtered


def _normalize_details(details: Iterable[str] | None) -> List[str]:
    """Normalize and validate requested catalog detail groups."""
    if details is None:
        return list(_CATALOG_DETAILS)

    normalized: List[str] = []
    for detail in details:
        lowered = str(detail).strip().lower()
        if lowered not in _CATALOG_DETAILS:
            raise ValueError("detail must be one of: tools, venues, managers")
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


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
    client: NanoHUBSubmitClient, detail: str, operation_timeout: float = 60.0
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
    client: NanoHUBSubmitClient, operation_timeout: float = 60.0
) -> List[str]:
    """Load available tool names from submit server help output."""
    return load_available_list(client, "tools", operation_timeout=operation_timeout)


def load_available_venues(
    client: NanoHUBSubmitClient, operation_timeout: float = 60.0
) -> List[str]:
    """Load available venue names from submit server help output."""
    return load_available_list(client, "venues", operation_timeout=operation_timeout)


def load_available_managers(
    client: NanoHUBSubmitClient, operation_timeout: float = 60.0
) -> List[str]:
    """Load available manager names from submit server help output."""
    return load_available_list(client, "managers", operation_timeout=operation_timeout)


@dataclass
class SubmitCatalog:
    """Structured server catalog containing tools, venues, and managers."""

    tools: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    managers: List[str] = field(default_factory=list)
    raw_help: Dict[str, str] = field(default_factory=dict)

    def entries(self, detail: str) -> List[str]:
        """Return entries for one detail group (`tools|venues|managers`)."""
        lowered = detail.strip().lower()
        if lowered == "tools":
            return list(self.tools)
        if lowered == "venues":
            return list(self.venues)
        if lowered == "managers":
            return list(self.managers)
        raise ValueError("detail must be one of: tools, venues, managers")

    def contains(
        self, detail: str, name: str, *, case_sensitive: bool = False
    ) -> bool:
        """Check whether a name exists in the selected detail group."""
        values = self.entries(detail)
        if case_sensitive:
            return name in values
        target = name.lower()
        return any(value.lower() == target for value in values)

    def filter(
        self,
        query: str,
        *,
        details: Iterable[str] | None = None,
        exact: bool = False,
        case_sensitive: bool = False,
        use_regex: bool = False,
        limit: int | None = None,
    ) -> Dict[str, List[str]]:
        """Filter names across one or more detail groups."""
        selected = _normalize_details(details)
        return {
            detail: _filter_names(
                self.entries(detail),
                query,
                exact=exact,
                case_sensitive=case_sensitive,
                use_regex=use_regex,
                limit=limit,
            )
            for detail in selected
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize catalog and raw help payloads to plain dictionaries."""
        return {
            "tools": list(self.tools),
            "venues": list(self.venues),
            "managers": list(self.managers),
            "raw_help": dict(self.raw_help),
        }


def load_available_catalog(
    client: NanoHUBSubmitClient,
    include_raw_help: bool = False,
    operation_timeout: float = 60.0,
) -> SubmitCatalog:
    """Load tools, venues, and managers in one request sequence."""
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


def filter_catalog(
    catalog: SubmitCatalog,
    query: str,
    *,
    details: Iterable[str] | None = None,
    exact: bool = False,
    case_sensitive: bool = False,
    use_regex: bool = False,
    limit: int | None = None,
) -> Dict[str, List[str]]:
    """Filter an already loaded `SubmitCatalog` by query."""
    return catalog.filter(
        query,
        details=details,
        exact=exact,
        case_sensitive=case_sensitive,
        use_regex=use_regex,
        limit=limit,
    )


def find_catalog_entries(
    client: NanoHUBSubmitClient,
    query: str,
    *,
    details: Iterable[str] | None = None,
    exact: bool = False,
    case_sensitive: bool = False,
    use_regex: bool = False,
    limit: int | None = 25,
    operation_timeout: float = 60.0,
) -> Dict[str, List[str]]:
    """Load (or reuse cached) catalog and return filtered matches."""
    get_catalog = getattr(client, "get_catalog", None)
    if callable(get_catalog):
        catalog = get_catalog(operation_timeout=operation_timeout)
    else:
        catalog = load_available_catalog(client, operation_timeout=operation_timeout)
    return filter_catalog(
        catalog,
        query,
        details=details,
        exact=exact,
        case_sensitive=case_sensitive,
        use_regex=use_regex,
        limit=limit,
    )


@dataclass
class SubmitServerExploration:
    """Composite introspection payload returned by `explore_submit_server`."""

    doctor: Dict[str, Any]
    catalog: SubmitCatalog
    venue_status_raw: str
    venue_status: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exploration payload to JSON-friendly structures."""
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
    operation_timeout: float = 60.0,
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
