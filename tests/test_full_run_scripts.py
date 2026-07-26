from __future__ import annotations

import os
import subprocess
from pathlib import Path


SINGLE_GPU_SCRIPT = Path("scripts/run_thought1_single_gpu_full.sh")
THREE_GPU_SCRIPT = Path("scripts/run_thought1_3gpu_full.sh")
THOUGHT2_FIVE_CATEGORY_SCRIPT = Path(
    "scripts/run_thought2_five_category_full.sh"
)


def test_single_gpu_full_script_exposes_safe_usage():
    result = subprocess.run(
        ["bash", str(SINGLE_GPU_SCRIPT), "--help"],
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
        ["bash", str(SINGLE_GPU_SCRIPT), "all"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "Formal evaluation was not started" in result.stderr


def test_three_gpu_full_script_exposes_safe_usage():
    result = subprocess.run(
        ["bash", str(THREE_GPU_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "GPU_IDS=0,1,2" in result.stdout
    assert "[all|clean|ood]" in result.stdout
    assert "6,771 OOD" in result.stdout


def test_three_gpu_full_script_requires_explicit_confirmation():
    environment = dict(os.environ)
    environment.pop("CONFIRM_FULL_EVAL", None)

    result = subprocess.run(
        ["bash", str(THREE_GPU_SCRIPT), "all"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "Formal evaluation was not started" in result.stderr


def test_thought2_five_category_script_exposes_safe_usage():
    result = subprocess.run(
        ["bash", str(THOUGHT2_FIVE_CATEGORY_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "[all|calibrate|diagnose|aggregate]" in result.stdout
    assert "CONFIRM_PHASE2_FIVE_CATEGORY" in result.stdout
    assert "200 Clean + 532 OOD" in result.stdout
    assert "--background" in result.stdout


def test_thought2_five_category_script_requires_confirmation():
    environment = dict(os.environ)
    environment.pop("CONFIRM_PHASE2_FIVE_CATEGORY", None)
    environment.pop("ACCEPT_STATIC_THRESHOLD", None)

    result = subprocess.run(
        ["bash", str(THOUGHT2_FIVE_CATEGORY_SCRIPT), "all"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "five-category run was not started" in result.stderr
