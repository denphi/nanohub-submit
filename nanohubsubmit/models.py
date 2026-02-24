from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ProgressMode(str, Enum):
    """Supported progress renderers."""

    AUTO = "auto"
    CURSES = "curses"
    SUBMIT = "submit"
    TEXT = "text"
    PEGASUS = "pegasus"
    SILENT = "silent"


@dataclass
class SubmitRequest:
    """Typed request data for a job submission command."""

    command: str
    command_arguments: list[str] = field(default_factory=list)

    debug: bool = False
    local: bool = False
    asynchronous: bool = False

    venues: list[str] = field(default_factory=list)
    input_files: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    separator: str | None = None
    data_file: str | None = None

    n_cpus: int | None = None
    ppn: int | None = None
    n_gpus: int | None = None
    gpn: int | None = None
    stripes: int | None = None
    memory_mb: int | None = None
    wall_time: int | str | None = None

    environment: Mapping[str, str | None] = field(default_factory=dict)

    run_name: str | None = None
    manager: str | None = None
    redundancy: int | None = None
    report_metrics: bool = False

    detach: bool = False
    attach_id: str | None = None
    wait: bool = False
    quota: bool = True

    tail_stdout: int | None = None
    tail_stderr: int | None = None
    tail_files: list[str] = field(default_factory=list)
    progress: ProgressMode | None = None

    save_json: str | None = None
    show: bool = False

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("command must be a non-empty string")
