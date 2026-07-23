from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path("scripts/run_thought1_single_gpu_full.sh")


def test_single_gpu_full_script_exposes_safe_usage():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "[all|clean|ood]" in result.stdout
    assert "CONFIRM_FULL_EVAL" in result.stdout
    assert "6,771 OOD" in result.stdout


def test_single_gpu_full_script_requires_explicit_confirmation():
    environment = dict(os.environ)
    environment.pop("CONFIRM_FULL_EVAL", None)

    result = subprocess.run(
        ["bash", str(SCRIPT), "all"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "Formal evaluation was not started" in result.stderr
