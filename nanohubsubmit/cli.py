from __future__ import annotations

import argparse
import json
import shlex
import sys

from .client import CommandExecutionError, NanoHUBSubmitClient
from .config import DEFAULT_CONFIG_PATH
from .models import ProgressMode, SubmitRequest
from .utils import explore_submit_server, load_available_catalog


def _parse_env(values: list[str]) -> dict[str, str | None]:
    env: dict[str, str | None] = {}
    for item in values:
        if "=" in item:
            key, value = item.split("=", 1)
            if not key:
                raise ValueError("invalid --env entry: key cannot be empty")
            env[key] = value
        else:
            if not item:
                raise ValueError("invalid --env entry: key cannot be empty")
            env[item] = None
    return env


def _resolve_command(
    command: str | None, command_line: list[str]
) -> tuple[str, list[str]]:
    command_line = list(command_line)
    if command_line and command_line[0] == "--":
        command_line = command_line[1:]

    if command_line:
        return command_line[0], command_line[1:]

    if command:
        parts = shlex.split(command)
        if not parts:
            raise ValueError("--command did not contain an executable")
        return parts[0], parts[1:]

    raise ValueError("submit requires a command via --command or trailing command")


def _emit_result(stdout: str, stderr: str) -> None:
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)


def _emit_catalog_text(catalog: dict) -> None:
    for key in ("tools", "venues", "managers"):
        values = catalog.get(key, [])
        sys.stdout.write("%s:\n" % key)
        if values:
            for value in values:
                sys.stdout.write("  %s\n" % value)
        else:
            sys.stdout.write("  (none)\n")


def _add_verbose_argument(
    parser: argparse.ArgumentParser, *, default: object = False
) -> None:
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default,
        help="Enable verbose protocol tracing",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanohub-submit",
        description="Modern standalone client for NanoHUB submit.",
    )
    _add_verbose_argument(parser, default=False)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to submit client config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--listen-uri",
        action="append",
        default=None,
        help="Server URI override, repeatable (e.g. tls://host:port)",
    )
    parser.add_argument(
        "--submit-ssl-ca",
        default=None,
        help="CA cert path override for TLS URIs",
    )
    parser.add_argument(
        "--maximum-connection-passes",
        type=int,
        default=None,
        help="Connection retry passes before failing",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Signon username override",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Signon password override",
    )
    parser.add_argument(
        "--session-token",
        default=None,
        help="Session token override",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID override",
    )
    parser.add_argument(
        "--cache-hosts",
        default=None,
        help="cache_hosts signon attribute override",
    )
    parser.add_argument(
        "--private-fingerprint",
        default=None,
        help="privateFingerPrint signon attribute override",
    )
    parser.add_argument(
        "--private-key-path",
        default=None,
        help="Private key path for encrypted signon challenge",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=1.0,
        help="Idle poll timeout in seconds while waiting for server messages",
    )
    parser.add_argument(
        "--keepalive-interval",
        type=float,
        default=15.0,
        help="Interval in seconds between keepalive null messages",
    )

    subparsers = parser.add_subparsers(dest="action", required=True)

    submit = subparsers.add_parser("submit", help="Submit a command")
    _add_verbose_argument(submit, default=argparse.SUPPRESS)
    submit.add_argument("--debug", action="store_true")
    submit.add_argument("--local", action="store_true")
    submit.add_argument("--asynchronous", action="store_true")
    submit.add_argument("--venue", action="append", default=[])
    submit.add_argument("--input-file", action="append", default=[])
    submit.add_argument("--separator")
    submit.add_argument("--parameter", action="append", default=[])
    submit.add_argument("--data-file")
    submit.add_argument("--n-cpus", type=int)
    submit.add_argument("--ppn", type=int)
    submit.add_argument("--n-gpus", type=int)
    submit.add_argument("--gpn", type=int)
    submit.add_argument("--stripes", type=int)
    submit.add_argument("--memory-mb", type=int)
    submit.add_argument("--wall-time")
    submit.add_argument("--env", action="append", default=[])
    submit.add_argument("--run-name")
    submit.add_argument("--manager")
    submit.add_argument("--redundancy", type=int)
    submit.add_argument("--metrics", action="store_true")
    submit.add_argument("--detach", action="store_true")
    submit.add_argument("--attach-id")
    submit.add_argument("--wait", action="store_true")
    quota_group = submit.add_mutually_exclusive_group()
    quota_group.add_argument(
        "--noquota",
        action="store_true",
        help="Disable quota checks",
    )
    quota_group.add_argument(
        "--quota",
        action="store_true",
        help="Explicitly enable quota checks",
    )
    submit.add_argument("--tail-stdout", type=int)
    submit.add_argument("--tail-stderr", type=int)
    submit.add_argument("--tail-file", action="append", default=[])
    submit.add_argument(
        "--progress",
        choices=[mode.value for mode in ProgressMode],
    )
    submit.add_argument("--save-json")
    submit.add_argument("--show", action="store_true")
    submit.add_argument(
        "--command",
        help="Command string to run, e.g. \"python3 run.py --size 10\"",
    )
    submit.add_argument(
        "command_line",
        nargs=argparse.REMAINDER,
        help="Alternative command form: -- <command> [args...]",
    )

    status = subparsers.add_parser("status", help="Query job status")
    _add_verbose_argument(status, default=argparse.SUPPRESS)
    status.add_argument("job_ids", nargs="+", type=int)

    kill = subparsers.add_parser("kill", help="Kill jobs")
    _add_verbose_argument(kill, default=argparse.SUPPRESS)
    kill.add_argument("job_ids", nargs="+", type=int)

    venue_status = subparsers.add_parser("venue-status", help="Query venue status")
    _add_verbose_argument(venue_status, default=argparse.SUPPRESS)
    venue_status.add_argument("venues", nargs="*")

    catalog = subparsers.add_parser(
        "catalog", help="Load available tools, venues, and managers"
    )
    _add_verbose_argument(catalog, default=argparse.SUPPRESS)
    catalog.add_argument(
        "--raw-help",
        action="store_true",
        help="Include raw help text in JSON output",
    )
    catalog.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    explore = subparsers.add_parser(
        "explore", help="Explore server capabilities and available metadata"
    )
    _add_verbose_argument(explore, default=argparse.SUPPRESS)
    explore.add_argument(
        "--raw-help",
        action="store_true",
        help="Include raw help text from server in output",
    )
    explore.add_argument(
        "--no-venue-status",
        action="store_true",
        help="Skip venue status query",
    )
    explore.add_argument(
        "--format",
        choices=["text", "json"],
        default="json",
        help="Output format",
    )

    raw = subparsers.add_parser("raw", help="Pass through raw submit args")
    _add_verbose_argument(raw, default=argparse.SUPPRESS)
    raw.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    verbose = bool(getattr(ns, "verbose", False))

    client = NanoHUBSubmitClient(
        config_path=ns.config,
        listen_uris=ns.listen_uri,
        submit_ssl_ca=ns.submit_ssl_ca,
        maximum_connection_passes=ns.maximum_connection_passes,
        username=ns.username,
        password=ns.password,
        session_token=ns.session_token,
        session_id=ns.session_id,
        cache_hosts=ns.cache_hosts,
        private_fingerprint=ns.private_fingerprint,
        private_key_path=ns.private_key_path,
        connect_timeout=ns.connect_timeout,
        idle_timeout=ns.idle_timeout,
        keepalive_interval=ns.keepalive_interval,
        verbose=verbose,
    )

    try:
        if ns.action == "submit":
            command, command_arguments = _resolve_command(ns.command, ns.command_line)
            request = SubmitRequest(
                command=command,
                command_arguments=command_arguments,
                debug=ns.debug,
                local=ns.local,
                asynchronous=ns.asynchronous,
                venues=ns.venue,
                input_files=ns.input_file,
                separator=ns.separator,
                parameters=ns.parameter,
                data_file=ns.data_file,
                n_cpus=ns.n_cpus,
                ppn=ns.ppn,
                n_gpus=ns.n_gpus,
                gpn=ns.gpn,
                stripes=ns.stripes,
                memory_mb=ns.memory_mb,
                wall_time=ns.wall_time,
                environment=_parse_env(ns.env),
                run_name=ns.run_name,
                manager=ns.manager,
                redundancy=ns.redundancy,
                report_metrics=ns.metrics,
                detach=ns.detach,
                attach_id=ns.attach_id,
                wait=ns.wait,
                quota=bool(ns.quota) or not bool(ns.noquota),
                tail_stdout=ns.tail_stdout,
                tail_stderr=ns.tail_stderr,
                tail_files=ns.tail_file,
                progress=ProgressMode(ns.progress) if ns.progress else None,
                save_json=ns.save_json,
                show=ns.show,
            )
            result = client.submit(request)
        elif ns.action == "status":
            result = client.status(ns.job_ids)
        elif ns.action == "kill":
            result = client.kill(ns.job_ids)
        elif ns.action == "venue-status":
            result = client.venue_status(ns.venues)
        elif ns.action == "catalog":
            catalog = load_available_catalog(
                client, include_raw_help=bool(ns.raw_help)
            ).to_dict()
            if ns.format == "json":
                sys.stdout.write(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
            else:
                _emit_catalog_text(catalog)
            return 0
        elif ns.action == "explore":
            exploration = explore_submit_server(
                client,
                include_raw_help=bool(ns.raw_help),
                include_venue_status=not bool(ns.no_venue_status),
            ).to_dict()
            if ns.format == "json":
                sys.stdout.write(
                    json.dumps(exploration, indent=2, sort_keys=True) + "\n"
                )
            else:
                doctor = exploration.get("doctor", {})
                sys.stdout.write("doctor.ok: %s\n" % doctor.get("ok"))
                if doctor.get("server_version"):
                    sys.stdout.write(
                        "doctor.server_version: %s\n" % doctor.get("server_version")
                    )
                capabilities = doctor.get("server_capabilities", {})
                if capabilities:
                    sys.stdout.write("capabilities:\n")
                    for key in sorted(capabilities.keys()):
                        sys.stdout.write("  %s=%s\n" % (key, capabilities[key]))
                _emit_catalog_text(exploration.get("catalog", {}))
                venue_status = exploration.get("venue_status", [])
                if venue_status:
                    sys.stdout.write("venue_status:\n")
                    for entry in venue_status:
                        sys.stdout.write("  %s\n" % entry)
            return 0
        else:
            raw_args = ns.args
            if raw_args and raw_args[0] == "--":
                raw_args = raw_args[1:]
            result = client.raw(raw_args)
    except (ValueError, CommandExecutionError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    _emit_result(result.stdout, result.stderr)
    return result.returncode
