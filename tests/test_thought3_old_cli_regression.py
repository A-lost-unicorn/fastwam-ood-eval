from __future__ import annotations

import hashlib

from conftest import write_config
from fastwam_ood_eval.cli import main


FROZEN_FILES = {
    "outputs/thought1/fastwam/combined/experiment_manifest.json": (
        "57dd93f51a2491423f1b14f0d90523f219218698e231a133dcef114caca132ee"
    ),
    "outputs/thought2/five_category_formal_v1/run_status.txt": (
        "32128801f41bfad982645fb2a8358df40bee638206623805bc2dedf6d13be718"
    ),
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_old_plan_command_still_uses_existing_config_and_schema(tmp_path, capsys):
    config = write_config(tmp_path, episodes=2)
    assert main(["plan", "--config", str(config)]) == 0
    assert "job_manifest" in capsys.readouterr().out
    assert (tmp_path / "mock_eval" / "job_manifest.jsonl").is_file()


def test_frozen_thought1_thought2_sentinel_hashes_are_unchanged():
    from pathlib import Path

    for raw_path, expected in FROZEN_FILES.items():
        path = Path(raw_path)
        assert path.is_file()
        assert _sha(path) == expected
