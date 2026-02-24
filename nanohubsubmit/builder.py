from __future__ import annotations

"""Argument builders for translating typed requests to submit CLI flags."""

from collections.abc import Iterable, Sequence
import re

from .models import SubmitRequest


def _append_many(args: list[str], option: str, values: Iterable[str]) -> None:
    """Append a repeatable option for each value in order."""
    for value in values:
        args.extend([option, value])


def _append_int(args: list[str], option: str, value: int | None) -> None:
    """Append an integer option if provided, validating non-negative values."""
    if value is None:
        return
    if value < 0:
        raise ValueError(f"{option} cannot be negative")
    args.extend([option, str(value)])


_WALL_TIME_HHMMSS = re.compile(r"^\d+:\d{2}:\d{2}$")


def _append_wall_time(args: list[str], value: int | str | None) -> None:
    """Append --wallTime supporting minutes or HH:MM:SS legacy format."""
    if value is None:
        return

    if isinstance(value, int):
        if value < 0:
            raise ValueError("--wallTime cannot be negative")
        args.extend(["--wallTime", str(value)])
        return

    wall_time = str(value).strip()
    if not wall_time:
        raise ValueError("--wallTime cannot be empty")

    if wall_time.isdigit():
        if int(wall_time) < 0:
            raise ValueError("--wallTime cannot be negative")
        args.extend(["--wallTime", wall_time])
        return

    if _WALL_TIME_HHMMSS.match(wall_time):
        args.extend(["--wallTime", wall_time])
        return

    raise ValueError("--wallTime must be minutes or hh:mm:ss")


def _append_separator(args: list[str], value: str | None) -> None:
    """Append parameter separator option used by sweep value splitting."""
    if value is None:
        return
    separator = str(value)
    if not separator:
        raise ValueError("--separator cannot be empty")
    args.extend(["--separator", separator])


class CommandBuilder:
    """Converts typed request data to submit CLI arguments."""

    @staticmethod
    def build_submit_args(request: SubmitRequest) -> list[str]:
        """Build a submit command argument vector from a SubmitRequest."""
        args: list[str] = []

        if request.local and request.asynchronous:
            raise ValueError("local and asynchronous cannot both be true")

        if request.debug:
            args.append("--debug")
        if request.local:
            args.append("--local")
        elif request.asynchronous:
            args.append("--asynchronous")

        _append_many(args, "--venue", request.venues)
        _append_many(args, "--inputfile", request.input_files)
        _append_separator(args, request.separator)
        _append_many(args, "--parameters", request.parameters)

        if request.data_file:
            args.extend(["--data", request.data_file])

        _append_int(args, "--nCpus", request.n_cpus)
        _append_int(args, "--ppn", request.ppn)
        _append_int(args, "--nGpus", request.n_gpus)
        _append_int(args, "--gpn", request.gpn)
        _append_int(args, "--stripes", request.stripes)
        _append_int(args, "--memory", request.memory_mb)
        _append_wall_time(args, request.wall_time)

        for key, value in request.environment.items():
            if not key:
                raise ValueError("environment variable key cannot be empty")
            if value is None:
                args.extend(["--env", key])
            else:
                args.extend(["--env", f"{key}={value}"])

        if request.run_name:
            args.extend(["--runName", request.run_name])
        if request.manager:
            args.extend(["--manager", request.manager])
        _append_int(args, "--redundancy", request.redundancy)

        if request.report_metrics:
            args.append("--metrics")
        if request.detach:
            args.append("--detach")
        if request.attach_id:
            args.extend(["--attach", request.attach_id])
        if request.wait:
            args.append("--wait")
        if not request.quota:
            args.append("--noquota")

        _append_int(args, "--tailStdout", request.tail_stdout)
        _append_int(args, "--tailStderr", request.tail_stderr)
        _append_many(args, "--tail", request.tail_files)

        if request.progress:
            args.extend(["--progress", request.progress.value])
        if request.save_json:
            args.extend(["--savejson", request.save_json])
        if request.show:
            args.append("--show")

        args.append(request.command)
        args.extend(request.command_arguments)
        return args

    @staticmethod
    def build_status_args(job_ids: Sequence[int]) -> list[str]:
        """Build repeated --status arguments for one or more remote job IDs."""
        if not job_ids:
            raise ValueError("job_ids cannot be empty")
        args: list[str] = []
        for job_id in job_ids:
            if job_id < 0:
                raise ValueError("job_ids cannot contain negative values")
            args.extend(["--status", str(job_id)])
        return args

    @staticmethod
    def build_kill_args(job_ids: Sequence[int]) -> list[str]:
        """Build repeated --kill arguments for one or more remote job IDs."""
        if not job_ids:
            raise ValueError("job_ids cannot be empty")
        args: list[str] = []
        for job_id in job_ids:
            if job_id < 0:
                raise ValueError("job_ids cannot contain negative values")
            args.extend(["--kill", str(job_id)])
        return args

    @staticmethod
    def build_venue_status_args(venues: Sequence[str] | None = None) -> list[str]:
        """Build a --venueStatus request with optional venue filters."""
        args = ["--venueStatus"]
        if venues:
            args.extend(venues)
        return args
