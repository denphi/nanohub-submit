import pytest

from nanohubsubmit.builder import CommandBuilder
from nanohubsubmit.models import ProgressMode, SubmitRequest


def test_build_submit_args_full() -> None:
    request = SubmitRequest(
        command="python3",
        command_arguments=["run.py", "--size", "32"],
        debug=True,
        asynchronous=True,
        venues=["workspace", "community"],
        input_files=["input.dat"],
        parameters=["alpha=1", "beta=2"],
        data_file="data.csv",
        n_cpus=8,
        ppn=4,
        n_gpus=1,
        gpn=1,
        memory_mb=4096,
        wall_time=1800,
        environment={"OMP_NUM_THREADS": "8", "KEEP_ME": None},
        run_name="example",
        manager="pegasus",
        redundancy=2,
        report_metrics=True,
        wait=True,
        quota=False,
        tail_stdout=20,
        tail_stderr=10,
        tail_files=["stderr.log"],
        progress=ProgressMode.TEXT,
        save_json="submit.json",
        show=True,
    )

    args = CommandBuilder.build_submit_args(request)

    assert args == [
        "--debug",
        "--asynchronous",
        "--venue",
        "workspace",
        "--venue",
        "community",
        "--inputfile",
        "input.dat",
        "--parameters",
        "alpha=1",
        "--parameters",
        "beta=2",
        "--data",
        "data.csv",
        "--nCpus",
        "8",
        "--ppn",
        "4",
        "--nGpus",
        "1",
        "--gpn",
        "1",
        "--memory",
        "4096",
        "--wallTime",
        "1800",
        "--env",
        "OMP_NUM_THREADS=8",
        "--env",
        "KEEP_ME",
        "--runName",
        "example",
        "--manager",
        "pegasus",
        "--redundancy",
        "2",
        "--metrics",
        "--wait",
        "--noquota",
        "--tailStdout",
        "20",
        "--tailStderr",
        "10",
        "--tail",
        "stderr.log",
        "--progress",
        "text",
        "--savejson",
        "submit.json",
        "--show",
        "python3",
        "run.py",
        "--size",
        "32",
    ]


def test_build_submit_args_rejects_conflicting_mode() -> None:
    request = SubmitRequest(command="echo", local=True, asynchronous=True)
    with pytest.raises(ValueError, match="local and asynchronous"):
        CommandBuilder.build_submit_args(request)


def test_build_status_and_kill_args() -> None:
    assert CommandBuilder.build_status_args([101, 202]) == [
        "--status",
        "101",
        "--status",
        "202",
    ]
    assert CommandBuilder.build_kill_args([1]) == ["--kill", "1"]


def test_build_submit_args_supports_wall_time_hhmmss() -> None:
    request = SubmitRequest(
        command="echo",
        command_arguments=["hello"],
        wall_time="01:30:00",
        progress=ProgressMode.AUTO,
    )
    args = CommandBuilder.build_submit_args(request)
    assert "--wallTime" in args
    assert "01:30:00" in args
    assert "--progress" in args
    assert "auto" in args


def test_build_submit_args_rejects_invalid_wall_time_format() -> None:
    request = SubmitRequest(command="echo", command_arguments=["hello"], wall_time="1h")
    with pytest.raises(ValueError, match="--wallTime must be minutes or hh:mm:ss"):
        CommandBuilder.build_submit_args(request)
