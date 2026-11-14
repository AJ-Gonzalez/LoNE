"""The macOS gate is about Core ML, not about the command surface."""

import io
import json
import os
import sys

import pytest
import typer

from lone import cli
from lone.cli_options import ComputeUnits


def test_help_inspect_and_endpoint_need_no_coreml():
    assert not cli._needs_coreml([])
    assert not cli._needs_coreml(["--help"])
    assert not cli._needs_coreml(["harness", "--help"])
    assert not cli._needs_coreml(["inspect", "--bundle", "b"])
    assert not cli._needs_coreml(["harness", "--endpoint", "http://h:8082"])
    assert not cli._needs_coreml(["harness", "--endpoint=http://h:8082"])


def test_model_commands_need_macos():
    assert cli._needs_coreml(["harness", "--bundle", "b"])
    assert cli._needs_coreml(["run", "--bundle", "b"])
    assert cli._needs_coreml(["serve", "--bundle", "b"])
    assert cli._needs_coreml(["chat"])
    assert cli._needs_coreml(["verify"])


def test_off_darwin_exit_is_clean(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["lone", "harness", "--bundle", "x"])
    with pytest.raises(SystemExit) as exit:
        cli.main()
    assert exit.value.code == 1
    captured = capsys.readouterr()
    assert "lone targets the Apple Neural Engine and needs macOS" in captured.err
    assert "Traceback" not in captured.err


def test_help_works_off_darwin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["lone", "--help"])
    with pytest.raises(SystemExit) as exit:
        cli.main()
    assert exit.value.code == 0
    assert "harness" in capsys.readouterr().out


def _harness(endpoint=None, bundle=None):
    return cli.harness(
        bundle=bundle,
        endpoint=endpoint,
        system=None,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=None,
        thinking=True,
        cpu_only=False,
        compute_units=ComputeUnits.cpu_and_ne,
        wire_caches=False,
    )


def test_harness_endpoint_mode_writes_the_envelope(monkeypatch, capsys, canned_server):
    canned_server.payload = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({
        "input": [[{"type": "text", "text": "Say hi."}]],
        "tools": [],
        "resources": [],
    })))
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    _harness(endpoint=canned_server.url)
    assert exits == [0]
    assert json.loads(capsys.readouterr().out.strip()) == {
        "eval": [{"type": "assistant_text_content", "text": "hi"}],
        "stdout": "",
    }


def test_harness_needs_exactly_one_source():
    with pytest.raises(typer.BadParameter):
        _harness()
    with pytest.raises(typer.BadParameter):
        _harness(bundle="b", endpoint="http://h:8082")
