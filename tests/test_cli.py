from __future__ import annotations

from nanohubsubmit.cli import build_parser


def test_verbose_flag_before_subcommand() -> None:
    parser = build_parser()
    ns = parser.parse_args(["--verbose", "status", "1"])
    assert ns.verbose is True
    assert ns.action == "status"


def test_verbose_flag_after_subcommand() -> None:
    parser = build_parser()
    ns = parser.parse_args(["status", "--verbose", "1"])
    assert ns.verbose is True
    assert ns.action == "status"


def test_all_subcommands_accept_verbose() -> None:
    parser = build_parser()
    command_vectors = [
        ["submit", "--verbose", "--", "echo"],
        ["status", "--verbose", "1"],
        ["kill", "--verbose", "1"],
        ["venue-status", "--verbose"],
        ["catalog", "--verbose"],
        ["explore", "--verbose"],
        ["raw", "--verbose", "--", "--help", "tools"],
    ]
    for argv in command_vectors:
        ns = parser.parse_args(argv)
        assert ns.verbose is True
