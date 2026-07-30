from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from fastwam_ood_eval.cli import main
from fastwam_ood_eval.thought3.injection import (
    ActionEncoderFutureInjector,
    FutureInjectionError,
)
from fastwam_ood_eval.thought3.online_counterfactual import (
    ONLINE_CF_CLASSIFICATIONS,
    OnlineCounterfactualError,
    action_pair_metrics,
    build_episode_derangement,
    classify_online_action_sensitivity,
    compute_replay_floor,
    delta_direction_cosine,
    load_k1_online_counterfactual_config,
    stable_online_seed,
)
from fastwam_ood_eval.thought3.phase1_k1_online_counterfactual import (
    Phase1OnlineCounterfactualError,
    _artifact_manifest,
    _preflight,
    _verify_artifact_manifest,
    online_counterfactual_dry_run_payload,
    run_k1_online_counterfactual,
)


CONFIG = Path(
    "configs/thought3/online/phase1_k1_action_counterfactual.yaml"
)
RUNNER = Path("scripts/run_thought3_k1_online_counterfactual.sh")


class _CountingAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden, future, mask):
        del future, mask
        self.calls += 1
        return hidden + 1


def _rows(
    *,
    correct_null: float,
    correct_shuffle: float,
    shuffle_hash_change: bool,
    b0_null: float = 0.0,
) -> list[dict[str, object]]:
    result = []
    for index in range(8):
        result.append(
            {
                "actions": {
                    "correct": {"sha256": f"correct-{index}"},
                    "shuffle": {
                        "sha256": (
                            f"shuffle-{index}"
                            if shuffle_hash_change
                            else f"correct-{index}"
                        )
                    },
                },
                "correct_null_vs_correct_shuffle_delta_cosine": 0.5,
                "pairs": {
                    "b0_null": {"linf": b0_null},
                    "correct_null": {"l2": correct_null},
                    "correct_shuffle": {"l2": correct_shuffle},
                },
            }
        )
    return result


def test_config_freezes_checkpoint_cohort_and_derangement():
    cfg = load_k1_online_counterfactual_config(CONFIG)
    mapping = build_episode_derangement(cfg)
    assert cfg.e6_checkpoint_dir.name == "step_00000200"
    assert cfg.e6_adapter_sha256 == (
        "aa55622c03aafea05c1bfedcb8548df398b0912dcecba397"
        "741c190c6b01b78f"
    )
    assert len(cfg.cohort) == 8
    assert len({sample.episode_id for sample in cfg.cohort}) == 8
    assert mapping["fingerprint"] == cfg.expected_shuffle_mapping_sha256
    assert {
        row["target_base_sample_id"] for row in mapping["mapping"]
    } == {
        row["donor_base_sample_id"] for row in mapping["mapping"]
    }
    assert all(
        row["target_base_sample_id"] != row["donor_base_sample_id"]
        for row in mapping["mapping"]
    )


def test_seed_identity_is_stable_and_sample_specific():
    first = stable_online_seed("namespace", 3407, "sample-a")
    assert first == stable_online_seed("namespace", 3407, "sample-a")
    assert first != stable_online_seed("namespace", 3407, "sample-b")
    assert first != stable_online_seed("other", 3407, "sample-a")
    assert 0 <= first < 2**63


def test_formal_null_is_exact_identity_and_never_calls_adapter():
    encoder = nn.Linear(3, 4)
    adapter = _CountingAdapter()
    injector = ActionEncoderFutureInjector(encoder, adapter)
    value = torch.randn(2, 3)
    baseline = encoder(value)
    with injector.activate_null(expected_calls=2):
        first = encoder(value)
        second = encoder(value)
    assert torch.equal(first, baseline)
    assert torch.equal(second, baseline)
    assert adapter.calls == 0
    assert not injector.has_active_context
    injector.close()


def test_formal_null_fails_closed_on_call_count_and_nesting():
    encoder = nn.Linear(3, 4)
    injector = ActionEncoderFutureInjector(encoder, _CountingAdapter())
    with pytest.raises(FutureInjectionError, match="call mismatch"):
        with injector.activate_null(expected_calls=2):
            encoder(torch.randn(1, 3))
    with injector.activate_null():
        with pytest.raises(FutureInjectionError, match="nested"):
            with injector.activate_null():
                pass
        encoder(torch.randn(1, 3))
    assert not injector.has_active_context
    injector.close()


def test_action_metrics_cover_components_timesteps_and_unavailable_eef():
    reference = torch.zeros(32, 7)
    intervention = torch.ones(32, 7)
    metrics = action_pair_metrics(reference, intervention)
    assert metrics["finite"] is True
    assert metrics["l1"] == pytest.approx(1.0)
    assert metrics["l2"] == pytest.approx(1.0)
    assert metrics["linf"] == pytest.approx(1.0)
    assert metrics["translation_difference"] == pytest.approx(3**0.5)
    assert metrics["rotation_difference"] == pytest.approx(3**0.5)
    assert metrics["gripper_difference"] == pytest.approx(1.0)
    assert len(metrics["per_timestep_l2"]) == 32
    assert metrics["per_timestep_l2"][0] == pytest.approx(7**0.5)
    assert metrics["eef_trajectory_difference"]["status"] == "unavailable"
    assert (
        delta_direction_cosine(
            correct=intervention,
            null=reference,
            shuffle=-intervention,
        )
        == pytest.approx(1.0)
    )


def test_replay_floor_is_derived_only_from_two_b0_repeats():
    cfg = load_k1_online_counterfactual_config(CONFIG)
    replay = [
        {
            "metrics": {
                "finite": True,
                "l2": float(index + 1) * 1e-10,
                "linf": float(index + 1) * 1e-10,
            }
        }
        for index in range(8)
    ]
    floor = compute_replay_floor(replay, cfg)
    assert floor["hard_passed"] is True
    assert floor["definition_frozen_before_interventions"] is True
    assert floor["material_l2_threshold"] == pytest.approx(
        cfg.replay_absolute_l2_floor
    )


@pytest.mark.parametrize(
    ("correct_null", "correct_shuffle", "hash_change", "expected"),
    (
        (0.2, 0.2, True, ONLINE_CF_CLASSIFICATIONS[0]),
        (0.2, 0.0, False, ONLINE_CF_CLASSIFICATIONS[1]),
        (0.0, 0.0, False, ONLINE_CF_CLASSIFICATIONS[2]),
    ),
)
def test_frozen_three_way_decision_rule(
    correct_null,
    correct_shuffle,
    hash_change,
    expected,
):
    cfg = load_k1_online_counterfactual_config(CONFIG)
    decision = classify_online_action_sensitivity(
        _rows(
            correct_null=correct_null,
            correct_shuffle=correct_shuffle,
            shuffle_hash_change=hash_change,
        ),
        replay_floor={
            "hard_passed": True,
            "material_l2_threshold": 0.1,
        },
        cfg=cfg,
    )
    assert decision["classification"] == expected


def test_null_parity_failure_forbids_sensitivity_classification():
    cfg = load_k1_online_counterfactual_config(CONFIG)
    with pytest.raises(OnlineCounterfactualError, match="null failed B0"):
        classify_online_action_sensitivity(
            _rows(
                correct_null=0.2,
                correct_shuffle=0.2,
                shuffle_hash_change=True,
                b0_null=cfg.replay_hard_max_linf * 2,
            ),
            replay_floor={
                "hard_passed": True,
                "material_l2_threshold": 0.1,
            },
            cfg=cfg,
        )


def test_readonly_preflight_resolves_exact_e6_checkpoint(monkeypatch):
    cfg = load_k1_online_counterfactual_config(CONFIG)
    monkeypatch.setattr(
        "fastwam_ood_eval.thought3.phase1_k1_online_counterfactual._git_status",
        lambda path: "",
    )
    base, report = _preflight(cfg)
    assert base.fingerprint == cfg.thought3_config_fingerprint
    assert report["main_checkpoint"]["global_step"] == 200
    assert report["main_checkpoint"]["variant"] == "A1"
    assert report["scope"]["training_future_cache_read"] is False


def test_dry_run_imports_no_torch_or_safetensors_and_writes_nothing():
    script = (
        "import json,sys\n"
        "from fastwam_ood_eval.cli import main\n"
        f"code=main(['thought3-k1-online-counterfactual','--config',{str(CONFIG)!r},'--dry-run'])\n"
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
    payload = json.loads(completed.stdout)
    assert payload == online_counterfactual_dry_run_payload(
        load_k1_online_counterfactual_config(CONFIG)
    )
    assert payload["would_write"] is False
    assert payload["would_start_rollout"] is False


def test_cli_rejects_runtime_overrides(capsys):
    assert (
        main(
            [
                "thought3-k1-online-counterfactual",
                "--config",
                str(CONFIG),
                "--dry-run",
                "--device",
                "cuda:0",
            ]
        )
        == 2
    )
    assert "forbids config overrides" in capsys.readouterr().err


def test_real_entrypoint_requires_explicit_confirmation(monkeypatch):
    cfg = load_k1_online_counterfactual_config(CONFIG)
    monkeypatch.delenv("CONFIRM_THOUGHT3_K1_ONLINE_CF", raising=False)
    with pytest.raises(
        Phase1OnlineCounterfactualError,
        match="CONFIRM_THOUGHT3_K1_ONLINE_CF",
    ):
        run_k1_online_counterfactual(cfg, resume=False)


def test_shell_runner_rejects_multi_gpu_id_before_nvidia_smi():
    environment = dict(os.environ)
    environment.update(
        {
            "CONFIRM_THOUGHT3_K1_ONLINE_CF": "YES",
            "THOUGHT3_GPU_ID": "1,2",
        }
    )
    completed = subprocess.run(
        ["bash", str(RUNNER)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "exactly one physical GPU integer" in completed.stderr


def test_completed_artifact_manifest_detects_corruption(tmp_path):
    output = tmp_path / "outputs" / "thought3" / "online"
    output.mkdir(parents=True)
    payload = output / "sample_results.jsonl"
    payload.write_text('{"sample":1}\n', encoding="utf-8")
    manifest = _artifact_manifest(output)
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert _verify_artifact_manifest(output)["file_count"] == 1
    payload.write_text('{"sample":2}\n', encoding="utf-8")
    with pytest.raises(
        Phase1OnlineCounterfactualError,
        match="checksum failed",
    ):
        _verify_artifact_manifest(output)


def test_online_runner_source_forbids_training_cache_and_rgb_decode():
    source = Path(
        "src/fastwam_ood_eval/thought3/"
        "phase1_k1_online_counterfactual.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "FutureCacheReader",
        "load_cache_plan",
        "_decode_latents",
        "run_action_counterfactuals",
    ):
        assert forbidden not in source
    assert "model.infer_action(" in source
    assert "sampler.sample(" in source
    assert "activate_null(" in source
