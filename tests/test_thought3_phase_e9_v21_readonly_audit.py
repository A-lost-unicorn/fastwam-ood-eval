from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from fastwam_ood_eval.cli import main
from fastwam_ood_eval.thought3.io_utils import load_jsonl
from fastwam_ood_eval.thought3.phase_e9_v21_readonly_audit import (
    INVALID_OUTCOME,
    VALID_OUTCOME,
    _CpuSchedulerReconstruction,
    _audit_probe_grid,
    audit_dry_run_payload,
    derive_flow_identity,
    load_e9_v21_audit_config,
)


CONFIG = Path(
    "configs/thought3/audits/phase_e9_v2_1_readonly_audit.yaml"
)
PARENT = Path(
    "outputs/thought3/phase_e9_sample_tail_mitigation_v2"
)


def test_audit_config_freezes_disjoint_parent_and_output():
    cfg = load_e9_v21_audit_config(CONFIG)
    assert cfg.parent_root == PARENT
    assert cfg.output_root != cfg.parent_root
    assert cfg.heldout_flows == tuple(range(75, 107))
    assert cfg.expected_zero_weight_positions == ((1, 80), (7, 93))
    assert {VALID_OUTCOME, INVALID_OUTCOME} == {
        "audit_valid_scientific_failed",
        "audit_invalid_identity_unrecoverable",
    }
    dry = audit_dry_run_payload(cfg)
    assert dry["scope"]["forward"] is False
    assert dry["scope"]["parent_write"] is False
    assert dry["would_load_checkpoint"] is False
    assert dry["would_load_fastwam"] is False
    assert dry["would_write"] is False


def test_legacy_identity_reconstruction_matches_observed_training_row():
    cfg = load_e9_v21_audit_config(CONFIG)
    path = PARENT / "tracks/raw/a0/train_objective_metrics.jsonl"
    observed = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    derived = derive_flow_identity(
        base_sample_id=observed["base_sample_id"],
        train_seed=cfg.train_seed,
        flow_step=observed["flow_step"],
        noise_namespace=cfg.noise_namespace,
        timestep_namespace=cfg.timestep_namespace,
    )
    assert derived["action_noise_seed"] == observed["action_noise_seed"]
    assert (
        derived["action_timestep_seed"]
        == observed["action_timestep_seed"]
    )
    assert (
        derived["flow_objective_sha256"]
        == observed["flow_objective_sha256"]
    )


def test_parent_probe_grid_is_reconstructable_without_parent_fields():
    cfg = load_e9_v21_audit_config(CONFIG)
    path = PARENT / "tracks/raw/a0/heldout_multiflow_metrics.jsonl"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    probes = load_jsonl(path)
    result, rows = _audit_probe_grid(
        cfg,
        track_key="raw/A0",
        probes=probes,
        scheduler=_CpuSchedulerReconstruction(),
    )
    assert all(result["checks"].values()), result
    assert result["stored_objective_rows"] == 512
    assert result["timestep_mismatch_count"] == 0
    assert result["weight_mismatch_count"] == 0
    assert (
        result["max_cpu_gpu_weight_abs_difference"]
        <= cfg.weight_abs_tolerance
    )
    assert len(rows) == 256
    assert all(
        row["original_parent_fields_observed"] is False for row in rows
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_audit_cli_dry_run_does_not_import_torch_or_write():
    script = (
        "import sys\n"
        "from fastwam_ood_eval.cli import main\n"
        f"code=main(['thought3-audit-e9-v2-artifacts','--config',{str(CONFIG)!r},'--dry-run'])\n"
        "assert code == 0\n"
        "assert 'torch' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["would_write"] is False


def test_audit_cli_rejects_device_override(capsys):
    assert (
        main(
            [
                "thought3-audit-e9-v2-artifacts",
                "--config",
                str(CONFIG),
                "--dry-run",
                "--device",
                "cuda:0",
            ]
        )
        == 2
    )
    assert (
        "forbids config overrides, device, and ranks"
        in capsys.readouterr().err
    )
