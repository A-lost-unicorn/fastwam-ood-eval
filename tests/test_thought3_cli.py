from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from fastwam_ood_eval.cli import build_parser, main
from thought3_test_utils import write_thought3_config


THOUGHT3_COMMANDS = (
    "thought3-audit",
    "thought3-smoke-real",
    "thought3-cache-real-smoke",
    "thought3-plan-cache",
    "thought3-build-cache",
    "thought3-validate-cache",
    "thought3-train",
    "thought3-counterfactual",
    "thought3-evaluate",
    "thought3-aggregate",
    "thought3-report",
)


@pytest.mark.parametrize("command", THOUGHT3_COMMANDS)
def test_every_thought3_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as captured:
        build_parser().parse_args([command, "--help"])
    assert captured.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--config", "--dry-run", "--resume", "--device"):
        assert flag in help_text


@pytest.mark.parametrize("command", THOUGHT3_COMMANDS)
def test_dry_run_never_loads_checkpoint_or_writes(
    command,
    tmp_path,
    monkeypatch,
    capsys,
):
    config = write_thought3_config(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    assert main([command, "--config", str(config), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["would_load_checkpoint"] is False
    assert payload["would_load_fastwam"] is False
    assert payload["would_write"] is False
    assert not (tmp_path / "thought3").exists()


def test_cpu_mock_cli_pipeline_is_complete_and_isolated(tmp_path, capsys):
    config = write_thought3_config(
        tmp_path,
        sample_count=8,
        max_steps=6,
        overrides={"training": {"checkpoint_interval": 6}},
    )
    commands = (
        "thought3-plan-cache",
        "thought3-build-cache",
        "thought3-validate-cache",
        "thought3-train",
        "thought3-counterfactual",
        "thought3-evaluate",
        "thought3-aggregate",
        "thought3-report",
    )
    payloads = {}
    for command in commands:
        assert main([command, "--config", str(config)]) == 0
        payloads[command] = json.loads(capsys.readouterr().out)
    assert payloads["thought3-validate-cache"]["status"] == "valid"
    assert payloads["thought3-counterfactual"]["action_metrics_status"] == "mock_complete"
    assert payloads["thought3-evaluate"]["online_cache_read"] is False
    assert payloads["thought3-report"]["status"] == "written"


def test_dry_run_process_does_not_import_torch_or_safetensors(tmp_path):
    config = write_thought3_config(tmp_path)
    script = (
        "import sys\n"
        "from fastwam_ood_eval.cli import main\n"
        f"code=main(['thought3-train','--config',{str(config)!r},'--dry-run'])\n"
        "assert code == 0\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'safetensors' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
