from __future__ import annotations

"""Configuration parsing for legacy submit-client.conf files.

Only the subset needed by this modern client is parsed from the `[client]`
section.
"""

import os
import re
from dataclasses import dataclass, field

DEFAULT_CONFIG_PATH = os.path.join(os.sep, "etc", "submit", "submit-client.conf")


@dataclass
class SubmitClientConfig:
    """Connection and protocol settings loaded from submit-client.conf."""

    listen_uris: list[str] = field(default_factory=list)
    maximum_connection_passes: int = 15
    submit_ssl_ca: str = os.path.join(os.sep, "etc", "submit", "submit_server_ca.crt")
    pegasus_version: str = "4.8.1"
    double_dash_terminator: bool = False


def _parse_bool(value: str) -> bool:
    """Parse legacy submit booleans where only literal 'true' enables a flag."""
    return value.strip().lower() == "true"


def _strip_comments(record: str) -> str:
    """Drop inline comments while preserving leading assignment content."""
    hash_pos = record.find("#")
    if hash_pos == -1:
        return record
    return record[:hash_pos]


def load_submit_client_config(path: str = DEFAULT_CONFIG_PATH) -> SubmitClientConfig:
    """
    Parse submit client configuration from disk.

    Unknown keys are ignored to preserve compatibility with richer config files
    used by legacy submit implementations.
    """

    cfg = SubmitClientConfig()
    if not path:
        return cfg
    if not os.path.exists(path):
        return cfg

    section_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    kv_re = re.compile(r"^\s*(\w+)\s*=\s*(.*?)\s*$")
    in_client_section = False

    with open(path, "r", encoding="utf-8") as fp:
        for raw_record in fp:
            record = _strip_comments(raw_record).strip()
            if not record:
                continue

            section_match = section_re.match(record)
            if section_match:
                in_client_section = section_match.group(1) == "client"
                continue

            if not in_client_section:
                continue

            kv_match = kv_re.match(record)
            if not kv_match:
                continue

            key, value = kv_match.group(1), kv_match.group(2)
            if key == "listenURIs":
                cfg.listen_uris = [entry.strip() for entry in value.split(",") if entry.strip()]
            elif key == "maximumConnectionPasses":
                cfg.maximum_connection_passes = int(value)
            elif key == "submitSSLCA":
                cfg.submit_ssl_ca = value
            elif key == "pegasusVersion":
                cfg.pegasus_version = value
            elif key == "doubleDashTerminator":
                cfg.double_dash_terminator = _parse_bool(value)

    # Legacy behavior allows forcing this via env.
    forced_double_dash = os.environ.get("SUBMIT_DOUBLEDASH")
    if forced_double_dash is not None:
        cfg.double_dash_terminator = _parse_bool(forced_double_dash)

    return cfg
