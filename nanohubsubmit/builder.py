from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import SubmitRequest


def _append_many(args: list[str], option: str, values: Iterable[str]) -> None:
    for value in values:
        args.extend([option, value])


def _append_int(args: list[str], option: str, value: int | None) -> None:
    if value is None:
        return
    if value < 0:
        raise ValueError(f"{option} cannot be negative")
    args.extend([option, str(value)])


class CommandBuilder:
    """Converts typed request data to submit CLI arguments."""

    @staticmethod
    def build_submit_args(request: SubmitRequest) -> list[str]:
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
        _append_many(args, "--parameters", request.parameters)

        if request.data_file:
            args.extend(["--data", request.data_file])

        _append_int(args, "--nCpus", request.n_cpus)
        _append_int(args, "--ppn", request.ppn)
        _append_int(args, "--nGpus", request.n_gpus)
        _append_int(args, "--gpn", request.gpn)
        _append_int(args, "--stripes", request.stripes)
        _append_int(args, "--memory", request.memory_mb)
        _append_int(args, "--wallTime", request.wall_time)

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
        args = ["--venueStatus"]
        if venues:
            args.extend(venues)
        return args
