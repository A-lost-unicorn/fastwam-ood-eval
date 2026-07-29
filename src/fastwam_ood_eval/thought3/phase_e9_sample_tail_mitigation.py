"""Gate E.9a: matched single-variable sample-tail mitigation.

E.9a is a result-conditioned sequential engineering experiment.  Four tracks
share the E.6 cohort, a new training-flow namespace, and a new held-out flow
panel: raw A0/A1 controls and fixed sample-normalized A0/A1 treatments.  The
only treatment variable is the frozen per-sample loss weight.  Train-order
positions 17--28 are identity-frozen here but are never decoded or trained.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import statistics
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.evaluation.evaluator import git_commit, git_dirty
from fastwam_ood_eval.thought3.cache_planner import load_cache_plan
from fastwam_ood_eval.thought3.config import (
    Thought3Config,
    load_thought3_config,
    validate_config,
)
from fastwam_ood_eval.thought3.io_utils import (
    atomic_write_json,
    load_json,
    load_jsonl,
    sha256_file,
)
from fastwam_ood_eval.thought3.objective_aggregation_training import (
    OBJECTIVE_AGGREGATION_UPDATES,
    OBJECTIVES_PER_UPDATE,
    ObjectiveAggregationProtocol,
    objective_aggregation_flow_slot,
    objective_aggregation_identity_schedule_sha256,
    objective_aggregation_schedule_sha256,
    run_full_cohort_objective_aggregation,
    sample_loss_weights_sha256,
)
from fastwam_ood_eval.thought3.phase_c_smoke import _load_upstream_model
from fastwam_ood_eval.thought3.phase_e2_eight_sample import (
    _matched_recipe_payload,
)
from fastwam_ood_eval.thought3.phase_e5_objective_aggregation import (
    _initial_probe_signature,
)
from fastwam_ood_eval.thought3.phase_e6_fresh_cohort_replication import (
    PHASE_E6_CONFIG,
    PHASE_E6_FROZEN_COHORT,
    PHASE_E6_LEARNING_RATE,
    PHASE_E6_ROOT,
    paired_superiority_checks,
    replication_performance_checks,
    verify_frozen_fresh_cohort,
    verify_frozen_phase_e5,
)
from fastwam_ood_eval.thought3.phase_e7_checkpoint_trajectory import (
    PHASE_E6_SAMPLE_PAYLOAD_SHA256,
    PHASE_E7_FASTWAM_COMMIT,
    verify_frozen_phase_e6,
)
from fastwam_ood_eval.thought3.phase_e8_a0_flow_variance_replication import (
    PHASE_E8_ROOT,
)
from fastwam_ood_eval.thought3.phase_e_training_smoke import (
    _verify_phase_d_gate,
)
from fastwam_ood_eval.thought3.real_training import (
    _flow_objective_identity,
    _training_order_key,
    multiflow_subset_outcome,
    prepare_real_training_data,
)
from fastwam_ood_eval.thought3.safety import ensure_thought3_output_path


PHASE_E9_SCHEMA = "thought3.phase_e9.sample_tail_mitigation.v1"
PHASE_E9_CONFIG = Path(
    "configs/thought3/phase_e9_sample_tail_mitigation.yaml"
)
PHASE_E9_EXPERIMENT_NAME = "thought3_phase_e9_sample_tail_mitigation"
PHASE_E9_ROOT = Path(
    "outputs/thought3/phase_e9_sample_tail_mitigation_v1"
)
PHASE_E9_CONFIG_FINGERPRINT = (
    "d4828d22224ce50c1c4d51b70bc9ce8c1c2a88a13b8d69560a4037de5fff512c"
)
PHASE_E9_FLOW_SLOT_OFFSET = 40_000
PHASE_E9_TRAIN_FLOW_START = 40_001
PHASE_E9_TRAIN_FLOW_END = 41_600
PHASE_E9_HELDOUT_BLOCK_A = tuple(range(75, 91))
PHASE_E9_HELDOUT_BLOCK_B = tuple(range(91, 107))
PHASE_E9_HELDOUT_FLOW_STEPS = (
    PHASE_E9_HELDOUT_BLOCK_A + PHASE_E9_HELDOUT_BLOCK_B
)
PHASE_E9_IDENTITY_SCHEDULE_SHA256 = (
    "4c5c66f977e6f75dfaf3bb9db398a13c8a2807d6c065ae19307b19435440d64e"
)
PHASE_E9_HELDOUT_IDENTITY_SHA256 = (
    "76e96cb5be832908aff1510256bc058fa5023c8b71e51b57dfe6b3f277d899fb"
)
PHASE_E9_EXPECTED_ZERO_WEIGHT_SLOTS = (
    (16, 4, 40_124),
    (16, 8, 40_128),
    (27, 4, 40_212),
    (39, 7, 40_311),
    (57, 5, 40_453),
    (58, 4, 40_460),
    (62, 5, 40_493),
    (69, 8, 40_552),
    (82, 5, 40_653),
    (89, 1, 40_705),
    (91, 8, 40_728),
    (104, 3, 40_827),
    (107, 5, 40_853),
    (108, 3, 40_859),
    (109, 1, 40_865),
    (119, 8, 40_952),
    (124, 7, 40_991),
    (135, 5, 41_077),
    (142, 5, 41_133),
    (167, 4, 41_332),
    (183, 1, 41_457),
    (183, 6, 41_462),
)
PHASE_E9_HELDOUT_ZERO_WEIGHT_POSITIONS = ((1, 80), (7, 93))
PHASE_E9_INITIAL_SAMPLE_LOSSES = {
    "9610d2aed3a6ddf382c514715ead977c9f9a25b56265b2705a9146ac28f6c0cc": (
        0.005343552826218456
    ),
    "75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b": (
        0.003564936602231228
    ),
    "5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44": (
        0.002473537944609916
    ),
    "8f34793be5e051e0d62c0397b83cc341f17b626bd73660968f48ff1f6339d1b9": (
        0.005682245363914262
    ),
    "8c00174e915504c49a3c69057f9c199af1654a6ecef414070c1657316b1e4418": (
        0.004042625069416772
    ),
    "461a673f2745ab243d99d617f4514a737644d44ba2fc5fdece8b45f347e51564": (
        0.007539614804613848
    ),
    "739baab482230ba4ee1ae9c0cccf5886268db9ee37c895435af6c6891d22c3b0": (
        0.004545348813849159
    ),
    "81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b": (
        0.0033364327465505994
    ),
}
PHASE_E9_SAMPLE_LOSS_WEIGHTS = {
    "9610d2aed3a6ddf382c514715ead977c9f9a25b56265b2705a9146ac28f6c0cc": (
        0.7686897293811428
    ),
    "75359438f810e6921754de327beda8bd974343f5e89fb54d7ac8852f79c89c9b": (
        1.152203989644269
    ),
    "5f82a5db9be7a61f969fd32f5bca19dbb19a65106fb49d5357705be2d03def44": (
        1.6605907278966263
    ),
    "8f34793be5e051e0d62c0397b83cc341f17b626bd73660968f48ff1f6339d1b9": (
        0.7228716665431351
    ),
    "8c00174e915504c49a3c69057f9c199af1654a6ecef414070c1657316b1e4418": (
        1.0160561777034391
    ),
    "461a673f2745ab243d99d617f4514a737644d44ba2fc5fdece8b45f347e51564": (
        0.5447936376545538
    ),
    "739baab482230ba4ee1ae9c0cccf5886268db9ee37c895435af6c6891d22c3b0": (
        0.9036785391265281
    ),
    "81363feff988d3f3faaeeb66191e7ff9c4fd40c85d7b3b7cd0bda84cd41e3b9b": (
        1.231115532050306
    ),
}
PHASE_E9_CALIBRATION_PAYLOAD_SHA256 = (
    "edfb31e3fe1d6a8067a607ed20803ded33ba98f860c2a679067e70aa21105d70"
)
PHASE_E9_SAMPLE_LOSS_WEIGHTS_SHA256 = (
    "3e65b4f76f6cdee7176c49c9befd12bcd416fe9f60f2f719446a2896b05719f6"
)
PHASE_E9_BOOTSTRAP_REPLICATES = 20_000
PHASE_E9_BOOTSTRAP_SEED = 20_260_729_090
PHASE_E9_FAMILYWISE_ALPHA = 0.05
PHASE_E9_FAMILYWISE_COMPARISONS = 32
PHASE_E9_TRACKS = (
    ("raw", "A0"),
    ("raw", "A1"),
    ("normalized", "A0"),
    ("normalized", "A1"),
)
PHASE_E8_FROZEN_ARTIFACTS = {
    "gate_e8_result.json": (
        "e3809eedaadc4eb7ce4c681151214f01304e08b0a45cd3bccf926ed003c989e1"
    ),
    "run_status.json": (
        "03e9039b078ef5cd34c2a97d55b5d25fec29937959aff29c4dd322956ce8f53a"
    ),
    "pre_validation_result.json": (
        "1a46e92af902e1613a87a4644912326f184b1517c289ea04d0d0becab8d6bc04"
    ),
    "data_preparation.json": (
        "abdb800855e3bdedc5f8e9e267e5c7e1cef030050b88a32cd58ecdf81c983828"
    ),
    "logs/phase_e8.log": (
        "68eda4a7b131a9cb82209df2c56ac67877ffc3dff564682877026d0abdc9743c"
    ),
}
PHASE_E9_RESERVED_COHORT = (
    ("1fc95daceda870a85bb86922ab9616fbafbe855cf8ba4087a9e24a4fba0ff15c", "episode_000005", 5),
    ("8905a37f8ae459be86fc1b32038978b31e2e76705c61d8376b6197047eb0650e", "episode_000008", 8),
    ("a10b86b1ab484588bd9dc3123b453bba8e32d2e1a299ec70119b6eafc96d6d63", "episode_000015", 15),
    ("0f4df424468f65f5d811a534f66667239f6e5491a54e9b4dfbe3d4155fa54456", "episode_000009", 9),
    ("3adec4471f56081985baeb57d428088594897e574c9dc932cad3950a909ab702", "episode_000016", 16),
    ("8f192e8bb4efccda60df55a6144aec7c4be8d4a1a3486757de88c7f094a69361", "episode_000022", 22),
    ("011cd4c8d0b8733b64f3bb6972d3e9cf729624fd86410ed93e4beadb3782f7f5", "episode_000006", 6),
    ("7b6f6128910d00fa642e1558255a6d109870cb691ea056ba1bae08537bd3a6ab", "episode_000039", 39),
    ("a57634c75a6ff93a7c9c403cac92a165dfd17c989f3c8af796c487b360717bb9", "episode_000020", 20),
    ("79f40b100893a2f47bc0fc20dfef740e60732710e2b2de8cc366b01ec41c6835", "episode_000017", 17),
    ("12bbc8a48340d1ea1d4f144c34c5cd1896321587038259cbf231824fa4bc4255", "episode_000018", 18),
    ("201476e51f22ba7a3cd26d3eb56013f4fa1fefe87a7c956e7a7bfdc820072613", "episode_000040", 40),
)
PHASE_E9_RESERVED_COHORT_SHA256 = (
    "0218d90eb6455d3297857423bfd34109469f308db9f69d5adeee02146ee42324"
)
PHASE_E9B_FLOW_BLOCK_A = tuple(range(107, 123))
PHASE_E9B_FLOW_BLOCK_B = tuple(range(123, 139))
PHASE_E9B_FLOW_STEPS = PHASE_E9B_FLOW_BLOCK_A + PHASE_E9B_FLOW_BLOCK_B
PHASE_E9B_IDENTITY_SCHEDULE_SHA256 = (
    "d5aeb3df50bbf11940ba545318327fd08df7f1e83dc27d7e3026ff6ed70b4f64"
)
PHASE_E9B_ZERO_WEIGHT_POSITIONS = (
    (1, 113),
    (2, 113),
    (3, 120),
    (3, 121),
    (8, 133),
    (9, 131),
)


PHASE_E9_RAW_PROTOCOL = ObjectiveAggregationProtocol(
    gate_label="Gate E.9a raw control",
    checkpoint_marker_key="gate_e9a_raw_control",
    flow_slot_offset=PHASE_E9_FLOW_SLOT_OFFSET,
    expected_zero_weight_slots=PHASE_E9_EXPECTED_ZERO_WEIGHT_SLOTS,
    heldout_flow_steps=PHASE_E9_HELDOUT_FLOW_STEPS,
)
PHASE_E9_NORMALIZED_PROTOCOL = ObjectiveAggregationProtocol(
    gate_label="Gate E.9a sample-normalized treatment",
    checkpoint_marker_key="gate_e9a_sample_normalized",
    flow_slot_offset=PHASE_E9_FLOW_SLOT_OFFSET,
    expected_zero_weight_slots=PHASE_E9_EXPECTED_ZERO_WEIGHT_SLOTS,
    heldout_flow_steps=PHASE_E9_HELDOUT_FLOW_STEPS,
    gradient_reduction="fixed_sample_normalized_mean",
    sample_loss_weights_sha256=PHASE_E9_SAMPLE_LOSS_WEIGHTS_SHA256,
)


class PhaseE9GateError(RuntimeError):
    """Raised when the frozen E.9 protocol is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    stage: str,
    values: Mapping[str, Any] | None = None,
    **extra: Any,
) -> None:
    import json

    payload = dict(values or {})
    payload.update(extra)
    print(
        json.dumps(
            {
                "phase": "E.9a",
                "stage": stage,
                "time": _utc_now(),
                **payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _assert_phase_e9_scope(cfg: Thought3Config) -> None:
    e6 = load_thought3_config(PHASE_E6_CONFIG)
    expected = replace(
        e6,
        experiment=replace(
            e6.experiment,
            name=PHASE_E9_EXPERIMENT_NAME,
            output_dir=PHASE_E9_ROOT,
        ),
    )
    observed_payload = cfg.to_dict()
    expected_payload = expected.to_dict()
    observed_payload.pop("source_path")
    expected_payload.pop("source_path")
    if observed_payload != expected_payload:
        raise PhaseE9GateError(
            "Gate E.9a changes more than experiment name/output"
        )
    if (
        cfg.fingerprint != PHASE_E9_CONFIG_FINGERPRINT
        or cfg.experiment.name != PHASE_E9_EXPERIMENT_NAME
        or cfg.experiment.output_dir != PHASE_E9_ROOT
        or cfg.training.learning_rate != PHASE_E6_LEARNING_RATE
    ):
        raise PhaseE9GateError("Gate E.9a frozen config identity changed")


def _require_phase_e9_confirmation() -> None:
    if os.environ.get("CONFIRM_THOUGHT3_PHASE_E9A") != "YES":
        raise PhaseE9GateError(
            "set CONFIRM_THOUGHT3_PHASE_E9A=YES for real E.9a training"
        )


def _verify_execution_repository() -> dict[str, Any]:
    provenance = {
        "fastwam_commit": git_commit(Path("third_party/FastWAM")),
        "fastwam_dirty": git_dirty(Path("third_party/FastWAM")),
        "project_commit": git_commit(Path.cwd()),
        "project_dirty": git_dirty(Path.cwd()),
    }
    if (
        provenance["project_commit"] is None
        or provenance["project_dirty"] is not False
        or provenance["fastwam_commit"] != PHASE_E7_FASTWAM_COMMIT
        or provenance["fastwam_dirty"] is not False
    ):
        raise PhaseE9GateError(
            "Gate E.9a requires clean project/FastWAM repositories and "
            "the frozen FastWAM commit"
        )
    return provenance


def probe_identity_schedule_sha256(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
    flow_steps: Sequence[int],
) -> str:
    """Hash a complete sample × held-out-flow identity grid."""

    normalized_ids = tuple(str(value) for value in sample_ids)
    normalized_steps = tuple(int(value) for value in flow_steps)
    if (
        not normalized_ids
        or len(set(normalized_ids)) != len(normalized_ids)
        or not normalized_steps
        or len(set(normalized_steps)) != len(normalized_steps)
        or any(value < 1 for value in normalized_steps)
    ):
        raise PhaseE9GateError("invalid E.9 probe identity grid")
    rows = []
    for sample_index, base_sample_id in enumerate(
        normalized_ids,
        start=0,
    ):
        for flow_step in normalized_steps:
            identity = _flow_objective_identity(
                base_sample_id=base_sample_id,
                train_seed=train_seed,
                flow_step=flow_step,
            )
            rows.append(
                "\0".join(
                    (
                        str(sample_index),
                        base_sample_id,
                        str(flow_step),
                        str(identity["action_noise_seed"]),
                        str(identity["action_timestep_seed"]),
                        str(identity["flow_objective_sha256"]),
                    )
                )
            )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _calibration_payload_sha256(sample_ids: Sequence[str]) -> str:
    payload = "\n".join(
        "\0".join(
            (
                sample_id,
                repr(PHASE_E9_INITIAL_SAMPLE_LOSSES[sample_id]),
                repr(PHASE_E9_SAMPLE_LOSS_WEIGHTS[sample_id]),
            )
        )
        for sample_id in sample_ids
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_phase_e8() -> dict[str, Any]:
    """Freeze the result-conditioned E.8 parent and its E.6 chain."""

    artifact_sha256 = {}
    for name, expected in PHASE_E8_FROZEN_ARTIFACTS.items():
        path = PHASE_E8_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise PhaseE9GateError(
                f"frozen Gate E.8 artifact changed/missing: {path}"
            )
        artifact_sha256[str(path)] = expected
    result = load_json(PHASE_E8_ROOT / "gate_e8_result.json")
    status = load_json(PHASE_E8_ROOT / "run_status.json")
    e6 = verify_frozen_phase_e6()
    sample_ids = list(e6["sample_ids"])
    observed_losses = {
        str(row["base_sample_id"]): float(row["action_loss"])
        for row in result["initial_probe"]["per_sample"]
    }
    if (
        result.get("schema_version")
        != "thought3.phase_e8.a0_flow_variance_replication.v1"
        or result.get("engineering_passed") is not True
        or result.get("status") != "complete"
        or status.get("status") != "complete"
        or status.get("gate_e8_passed") is not True
        or result["diagnostic_classification"]["classification"]
        != "mixed_or_inconclusive"
        or result["scope"]["probe_objectives"] != 1_536
        or result["scope"]["optimizer_steps"] != 0
        or result["scope"]["training_objectives"] != 0
        or result["steps"]["200"]["panels"]["full"]["outcome"][
            "loss_reduction_fraction"
        ]
        != 0.037282523392622224
        or result["steps"]["200"]["panels"]["full"]["outcome"][
            "non_worsened_sample_count"
        ]
        != 4
        or observed_losses != PHASE_E9_INITIAL_SAMPLE_LOSSES
        or list(result["initial_probe"]["sample_ids"]) != sample_ids
        or _calibration_payload_sha256(sample_ids)
        != PHASE_E9_CALIBRATION_PAYLOAD_SHA256
        or sample_loss_weights_sha256(
            sample_ids,
            PHASE_E9_SAMPLE_LOSS_WEIGHTS,
        )
        != PHASE_E9_SAMPLE_LOSS_WEIGHTS_SHA256
    ):
        raise PhaseE9GateError("Gate E.8 parent/calibration identity changed")
    return {
        "artifact_sha256": artifact_sha256,
        "classification": "mixed_or_inconclusive",
        "e6": e6,
        "known_before_e9": {
            "all_e8_results_read": True,
            "mitigation_selected_after_e8": True,
            "not_independent_confirmatory": True,
        },
        "sample_ids": sample_ids,
        "sample_initial_losses": dict(PHASE_E9_INITIAL_SAMPLE_LOSSES),
        "sample_loss_weights": dict(PHASE_E9_SAMPLE_LOSS_WEIGHTS),
    }


def verify_reserved_replication_cohort(
    cfg: Thought3Config,
    *,
    used_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze positions 17--28 without decoding or training them."""

    entries, plan = load_cache_plan(cfg.cache.root)
    train = sorted(
        (entry for entry in entries if entry.k == 1 and entry.split == "train"),
        key=lambda entry: _training_order_key(
            entry.identity.base_sample_id,
            seed=cfg.training.train_seed,
        ),
    )
    development_ids = {
        entry.identity.base_sample_id
        for entry in entries
        if entry.k == 1 and entry.split == "development"
    }
    selected = train[16:28]
    rows = tuple(
        (
            entry.identity.base_sample_id,
            entry.identity.demonstration_id,
            entry.identity.episode_index,
        )
        for entry in selected
    )
    cohort_payload = "\n".join(
        "\0".join(
            (
                str(index),
                sample_id,
                demonstration_id,
                str(episode_index),
                "train",
            )
        )
        for index, (
            sample_id,
            demonstration_id,
            episode_index,
        ) in enumerate(rows, start=1)
    )
    cohort_sha256 = hashlib.sha256(
        cohort_payload.encode("utf-8")
    ).hexdigest()
    sample_ids = [row[0] for row in rows]
    identity_sha256 = probe_identity_schedule_sha256(
        sample_ids,
        train_seed=cfg.training.train_seed,
        flow_steps=PHASE_E9B_FLOW_STEPS,
    )
    if (
        rows != PHASE_E9_RESERVED_COHORT
        or cohort_sha256 != PHASE_E9_RESERVED_COHORT_SHA256
        or identity_sha256 != PHASE_E9B_IDENTITY_SCHEDULE_SHA256
        or set(sample_ids) & set(used_sample_ids)
        or set(sample_ids) & development_ids
        or plan["split_fingerprint"]
        != "ea5402955023ccd48d790d821a73f98549b31d1ace8af035a90ceae2ad3951eb"
    ):
        raise PhaseE9GateError("Gate E.9b reserved cohort identity changed")
    return {
        "cohort_sha256": cohort_sha256,
        "decoded_or_trained_by_e9a": False,
        "heldout_flow_block_a": list(PHASE_E9B_FLOW_BLOCK_A),
        "heldout_flow_block_b": list(PHASE_E9B_FLOW_BLOCK_B),
        "identity_schedule_sha256": identity_sha256,
        "sample_ids": sample_ids,
        "samples": [
            {
                "base_sample_id": sample_id,
                "demonstration_id": demonstration_id,
                "episode_index": episode_index,
                "split": "train",
            }
            for sample_id, demonstration_id, episode_index in rows
        ],
        "selection": "train order positions 17-28 (1-based)",
        "zero_weight_positions": [
            list(value) for value in PHASE_E9B_ZERO_WEIGHT_POSITIONS
        ],
    }


def _assert_frozen_design(
    sample_ids: Sequence[str],
    *,
    train_seed: int,
) -> dict[str, str]:
    train_identity = objective_aggregation_identity_schedule_sha256(
        sample_ids,
        train_seed=train_seed,
        flow_slot_offset=PHASE_E9_FLOW_SLOT_OFFSET,
    )
    heldout_identity = probe_identity_schedule_sha256(
        sample_ids,
        train_seed=train_seed,
        flow_steps=PHASE_E9_HELDOUT_FLOW_STEPS,
    )
    if (
        tuple(row["base_sample_id"] for row in PHASE_E6_FROZEN_COHORT)
        != tuple(sample_ids)
        or PHASE_E9_HELDOUT_BLOCK_A != tuple(range(75, 91))
        or PHASE_E9_HELDOUT_BLOCK_B != tuple(range(91, 107))
        or PHASE_E9_TRAIN_FLOW_START != 40_001
        or PHASE_E9_TRAIN_FLOW_END != 41_600
        or train_identity != PHASE_E9_IDENTITY_SCHEDULE_SHA256
        or heldout_identity != PHASE_E9_HELDOUT_IDENTITY_SHA256
        or PHASE_E9_BOOTSTRAP_REPLICATES != 20_000
        or PHASE_E9_FAMILYWISE_COMPARISONS != 32
        or PHASE_E9_TRACKS
        != (
            ("raw", "A0"),
            ("raw", "A1"),
            ("normalized", "A0"),
            ("normalized", "A1"),
        )
    ):
        raise PhaseE9GateError("Gate E.9a frozen design changed")
    return {
        "heldout_identity_schedule_sha256": heldout_identity,
        "train_identity_schedule_sha256": train_identity,
    }


def derive_e9_track_config(
    cfg: Thought3Config,
    *,
    recipe: str,
    variant: str,
) -> Thought3Config:
    if recipe not in {"raw", "normalized"} or variant not in {"A0", "A1"}:
        raise PhaseE9GateError(
            f"unsupported Gate E.9a track: {recipe}/{variant}"
        )
    derived = replace(
        cfg,
        variant=variant,
        experiment=replace(
            cfg.experiment,
            name=f"thought3_phase_e9a_{recipe}_{variant.lower()}",
            output_dir=(
                cfg.experiment.output_dir
                / "tracks"
                / recipe
                / variant.lower()
            ),
        ),
        sampler=replace(
            cfg.sampler,
            active_k=0 if variant == "A0" else 1,
        ),
    )
    validate_config(derived)
    return derived


def _protocol(recipe: str) -> ObjectiveAggregationProtocol:
    if recipe == "raw":
        return PHASE_E9_RAW_PROTOCOL
    if recipe == "normalized":
        return PHASE_E9_NORMALIZED_PROTOCOL
    raise PhaseE9GateError(f"unsupported Gate E.9a recipe: {recipe}")


def paired_tail_bootstrap(
    initial_probe: Mapping[str, Any],
    final_probe: Mapping[str, Any],
    *,
    track_key: str,
    bootstrap_replicates: int = PHASE_E9_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = PHASE_E9_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Confirm within-sample harm using the frozen 32-flow panel."""

    import numpy as np

    if (
        track_key not in {
            f"{recipe}/{variant}" for recipe, variant in PHASE_E9_TRACKS
        }
        or bootstrap_replicates != PHASE_E9_BOOTSTRAP_REPLICATES
    ):
        raise PhaseE9GateError("Gate E.9a bootstrap design changed")
    sample_ids = [str(value) for value in initial_probe["sample_ids"]]
    if sample_ids != [str(value) for value in final_probe["sample_ids"]]:
        raise PhaseE9GateError("Gate E.9a bootstrap sample order changed")
    initial_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): float(
            row["action_loss"]
        )
        for row in initial_probe["per_objective"]
    }
    final_rows = {
        (str(row["base_sample_id"]), int(row["flow_step"])): float(
            row["action_loss"]
        )
        for row in final_probe["per_objective"]
    }
    expected = {
        (sample_id, flow_step)
        for sample_id in sample_ids
        for flow_step in PHASE_E9_HELDOUT_FLOW_STEPS
    }
    if set(initial_rows) != expected or set(final_rows) != expected:
        raise PhaseE9GateError("Gate E.9a bootstrap objective grid changed")
    quantile = PHASE_E9_FAMILYWISE_ALPHA / PHASE_E9_FAMILYWISE_COMPARISONS
    per_sample = []
    for sample_index, sample_id in enumerate(sample_ids):
        initial = np.asarray(
            [
                initial_rows[(sample_id, flow)]
                for flow in PHASE_E9_HELDOUT_FLOW_STEPS
            ],
            dtype=np.float64,
        )
        final = np.asarray(
            [
                final_rows[(sample_id, flow)]
                for flow in PHASE_E9_HELDOUT_FLOW_STEPS
            ],
            dtype=np.float64,
        )
        if (
            float(initial.mean()) <= 0
            or float(initial[:16].mean()) <= 0
            or float(initial[16:].mean()) <= 0
        ):
            raise PhaseE9GateError(
                "Gate E.9a bootstrap initial means must be positive"
            )
        rng = np.random.default_rng(bootstrap_seed + sample_index)
        indices = rng.integers(
            0,
            len(PHASE_E9_HELDOUT_FLOW_STEPS),
            size=(
                bootstrap_replicates,
                len(PHASE_E9_HELDOUT_FLOW_STEPS),
            ),
        )
        boot_initial = initial[indices].mean(axis=1)
        boot_final = final[indices].mean(axis=1)
        if bool(np.any(boot_initial <= 0)):
            raise PhaseE9GateError(
                "Gate E.9a bootstrap produced non-positive denominator"
            )
        boot_change = (boot_final - boot_initial) / boot_initial
        full_change = float(
            (final.mean() - initial.mean()) / initial.mean()
        )
        block_a_change = float(
            (final[:16].mean() - initial[:16].mean())
            / initial[:16].mean()
        )
        block_b_change = float(
            (final[16:].mean() - initial[16:].mean())
            / initial[16:].mean()
        )
        lower = float(
            np.quantile(boot_change, quantile, method="linear")
        )
        upper = float(
            np.quantile(
                boot_change,
                1.0 - quantile,
                method="linear",
            )
        )
        confirmed_worsened = (
            full_change > 0
            and block_a_change > 0
            and block_b_change > 0
            and lower > 0
        )
        per_sample.append(
            {
                "base_sample_id": sample_id,
                "block_a_relative_change": block_a_change,
                "block_b_relative_change": block_b_change,
                "bonferroni_one_sided_lower": lower,
                "bonferroni_one_sided_upper": upper,
                "confirmed_worsened": confirmed_worsened,
                "flow_worsened_fraction": float(
                    np.mean(final > initial)
                ),
                "full_relative_change": full_change,
                "material_two_percent_worsening": (
                    confirmed_worsened and full_change >= 0.02
                ),
            }
        )
    confirmed = [
        str(row["base_sample_id"])
        for row in per_sample
        if bool(row["confirmed_worsened"])
    ]
    return {
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "confirmed_worsened_sample_count": len(confirmed),
        "confirmed_worsened_sample_ids": confirmed,
        "familywise_alpha": PHASE_E9_FAMILYWISE_ALPHA,
        "familywise_comparisons": PHASE_E9_FAMILYWISE_COMPARISONS,
        "flow_steps": list(PHASE_E9_HELDOUT_FLOW_STEPS),
        "one_sided_quantile": quantile,
        "per_sample": per_sample,
        "resampling_unit": "paired_flow_within_sample",
        "track_key": track_key,
    }


def classify_sample_tail_mitigation(
    performance: Mapping[str, Mapping[str, bool]],
    paired: Mapping[str, bool],
    tail: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen E.9a candidate and tail-contrast hierarchy."""

    expected = {
        f"{recipe}/{variant}" for recipe, variant in PHASE_E9_TRACKS
    }
    if set(performance) != expected or set(tail) != expected:
        raise PhaseE9GateError("Gate E.9a classification track set changed")
    raw_harms = sum(
        int(tail[f"raw/{variant}"]["confirmed_worsened_sample_count"])
        for variant in ("A0", "A1")
    )
    normalized_harms = sum(
        int(
            tail[f"normalized/{variant}"][
                "confirmed_worsened_sample_count"
            ]
        )
        for variant in ("A0", "A1")
    )
    normalized_absolute = all(
        all(performance[f"normalized/{variant}"].values())
        for variant in ("A0", "A1")
    )
    normalized_tail_safe = normalized_harms == 0
    normalized_candidate = (
        normalized_absolute
        and all(paired.values())
        and normalized_tail_safe
    )
    if normalized_candidate and normalized_harms < raw_harms:
        classification = "tail_mitigation_candidate_supported"
        independent_replication_candidate = True
    elif normalized_candidate:
        classification = (
            "stable_normalized_candidate_without_tail_contrast"
        )
        independent_replication_candidate = True
    else:
        classification = "sample_tail_mitigation_not_supported"
        independent_replication_candidate = False
    return {
        "classification": classification,
        "independent_replication_candidate": (
            independent_replication_candidate
        ),
        "normalized_absolute_gates_passed": normalized_absolute,
        "normalized_confirmed_harm_count": normalized_harms,
        "normalized_paired_gate_passed": all(paired.values()),
        "normalized_tail_safe": normalized_tail_safe,
        "raw_confirmed_harm_count": raw_harms,
        "strict_tail_harm_reduction": normalized_harms < raw_harms,
    }


def _track_checks(
    cfg: Thought3Config,
    result: Mapping[str, Any],
    *,
    recipe: str,
    protocol: ObjectiveAggregationProtocol,
) -> tuple[dict[str, bool], dict[str, Any]]:
    objective_rows = load_jsonl(Path(str(result["objective_metrics"])))
    update_rows = load_jsonl(Path(str(result["update_metrics"])))
    probe_rows = load_jsonl(Path(str(result["probe_metrics"])))
    sample_ids = list(result["sample_ids"])
    expected_weights = (
        {sample_id: 1.0 for sample_id in sample_ids}
        if recipe == "raw"
        else PHASE_E9_SAMPLE_LOSS_WEIGHTS
    )
    observed_schedule = objective_aggregation_schedule_sha256(
        objective_rows,
        flow_slot_offset=PHASE_E9_FLOW_SLOT_OFFSET,
    )
    recomputed = multiflow_subset_outcome(
        result["initial_probe"],
        result["final_probe"],
    )
    zero_slots = tuple(
        (
            int(row["optimizer_update"]),
            int(row["micro_index"]),
            int(row["training_flow_slot"]),
        )
        for row in objective_rows
        if float(row["action_weight"]) == 0
    )

    def heldout_probe_exact(probe: Mapping[str, Any]) -> bool:
        rows = list(probe["per_objective"])
        expected_grid = {
            (sample_id, flow_step)
            for sample_id in sample_ids
            for flow_step in PHASE_E9_HELDOUT_FLOW_STEPS
        }
        observed_zero = tuple(
            (
                sample_ids.index(str(row["base_sample_id"])),
                int(row["flow_step"]),
            )
            for row in rows
            if float(row["action_weight"]) == 0
        )
        return (
            len(rows) == 256
            and {
                (str(row["base_sample_id"]), int(row["flow_step"]))
                for row in rows
            }
            == expected_grid
            and all(
                all(
                    row.get(field)
                    == _flow_objective_identity(
                        base_sample_id=str(row["base_sample_id"]),
                        train_seed=cfg.training.train_seed,
                        flow_step=int(row["flow_step"]),
                    )[field]
                    for field in (
                        "action_noise_seed",
                        "action_timestep_seed",
                        "flow_objective_sha256",
                    )
                )
                for row in rows
            )
            and observed_zero
            == PHASE_E9_HELDOUT_ZERO_WEIGHT_POSITIONS
            and all(
                float(row["action_weight"]) != 0
                or float(row["action_loss"]) == 0
                for row in rows
            )
        )

    rows_exact = all(
        int(row["objective_index"]) == index
        and int(row["optimizer_update"]) == (index - 1) // 8 + 1
        and int(row["micro_index"]) == (index - 1) % 8 + 1
        and str(row["base_sample_id"])
        == sample_ids[(index - 1) % 8]
        and int(row["training_flow_slot"]) == 40_000 + index
        and float(row["sample_loss_weight"])
        == expected_weights[str(row["base_sample_id"])]
        and row["gradient_reduction"] == protocol.gradient_reduction
        and math.isclose(
            float(row["mean_scaled_backward_loss"]),
            float(row["action_loss"])
            * float(row["sample_loss_weight"])
            / 8,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for index, row in enumerate(objective_rows, start=1)
    )
    execution = {
        "complete_200_updates_1600_objectives": (
            result.get("status") == "complete"
            and int(result["completed_steps"]) == 200
            and int(result["completed_objectives"]) == 1_600
            and len(update_rows) == 200
            and len(objective_rows) == 1_600
        ),
        "exact_single_variable_weight_recipe": (
            rows_exact
            and result["gradient_reduction"]
            == protocol.gradient_reduction
            and result.get("sample_loss_weights_sha256")
            == protocol.sample_loss_weights_sha256
        ),
        "frozen_train_identity_schedule": (
            result["identity_schedule_sha256"]
            == PHASE_E9_IDENTITY_SCHEDULE_SHA256
            and observed_schedule
            == result["train_flow_schedule_sha256"]
        ),
        "new_training_flow_namespace_exact": (
            [int(row["training_flow_slot"]) for row in objective_rows]
            == list(range(40_001, 41_601))
        ),
        "heldout_probe_schedule_exact": (
            [int(row["global_step"]) for row in probe_rows] == [0, 200]
            and all(
                row["flow_steps"]
                == list(PHASE_E9_HELDOUT_FLOW_STEPS)
                and int(row["flow_objective_count"]) == 256
                and row["sample_ids"] == sample_ids
                for row in probe_rows
            )
        ),
        "heldout_rng_and_zero_weight_identity_exact": (
            heldout_probe_exact(result["initial_probe"])
            and heldout_probe_exact(result["final_probe"])
            and probe_identity_schedule_sha256(
                sample_ids,
                train_seed=cfg.training.train_seed,
                flow_steps=PHASE_E9_HELDOUT_FLOW_STEPS,
            )
            == PHASE_E9_HELDOUT_IDENTITY_SHA256
        ),
        "initial_zero_gate_identity": all(
            float(row["gated_delta_norm"]) == 0
            for row in result["initial_probe"]["per_objective"]
        ),
        "outcome_recomputes_exactly": dict(result["outcome"]) == recomputed,
        "zero_weight_training_slots_exact": (
            zero_slots == PHASE_E9_EXPECTED_ZERO_WEIGHT_SLOTS
        ),
        "finite_and_memory_bounded": (
            all(
                not bool(row["nan_or_inf"])
                and math.isfinite(float(row["action_loss"]))
                and math.isfinite(float(row["mean_scaled_backward_loss"]))
                for row in objective_rows
            )
            and all(
                not bool(row["nan_or_inf"])
                and math.isfinite(float(row["gate_gradient"]))
                and float(row["peak_memory_mib"]) < 43 * 1024
                for row in update_rows
            )
            and float(result["max_peak_memory_mib"]) < 43 * 1024
        ),
        "gate_then_non_gate_gradient_contract": (
            int(result["first_non_gate_nonzero_gradient_update"]) == 2
            and int(result["first_projector_nonzero_gradient_update"]) == 2
            and int(result["first_attention_nonzero_gradient_update"]) == 2
        ),
        "adapter_only_checkpoint_roundtrip": (
            result["optimizer_parameter_scope"] == "adapter_only"
            and result["checkpoint_roundtrip"]["state_equal"] is True
            and int(result["checkpoint_roundtrip"]["global_step"]) == 200
        ),
        "no_development_ood_success_or_future_rgb": (
            result["uses_development_outcomes"] is False
            and result["uses_ood_or_success_outcomes"] is False
            and result["uses_ground_truth_future_input"] is False
        ),
    }
    artifacts = {
        "identity_schedule_sha256": result["identity_schedule_sha256"],
        "observed_schedule_sha256": observed_schedule,
        "output_dir": str(cfg.experiment.output_dir),
        "sample_loss_weights_sha256": (
            protocol.sample_loss_weights_sha256
        ),
    }
    return execution, artifacts


def _run_phase_e9(
    cfg: Thought3Config,
    *,
    resume: bool,
    execution_repository: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from fastwam_ood_eval.thought3.model_wrapper import (
        parameter_state_sha256,
    )

    _assert_phase_e9_scope(cfg)
    _require_phase_e9_confirmation()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PhaseE9GateError("Gate E.9a requires one CUDA-visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise PhaseE9GateError(
            "Gate E.9a requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.cuda.set_device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)

    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    e8 = verify_frozen_phase_e8()
    e5 = verify_frozen_phase_e5()
    cohort = verify_frozen_fresh_cohort(
        cfg,
        e5_sample_ids=e5["sample_ids"],
    )
    if cohort["sample_ids"] != e8["sample_ids"]:
        raise PhaseE9GateError("Gate E.9a cohort differs from E.8")
    design = _assert_frozen_design(
        cohort["sample_ids"],
        train_seed=cfg.training.train_seed,
    )
    reserved = verify_reserved_replication_cohort(
        cfg,
        used_sample_ids=cohort["sample_ids"],
    )
    phase_d = _verify_phase_d_gate(cfg)
    tracks = {
        f"{recipe}/{variant}": derive_e9_track_config(
            cfg,
            recipe=recipe,
            variant=variant,
        )
        for recipe, variant in PHASE_E9_TRACKS
    }
    if any(
        _matched_recipe_payload(tracks[f"{recipe}/A0"])
        != _matched_recipe_payload(tracks[f"{recipe}/A1"])
        for recipe in ("raw", "normalized")
    ):
        raise PhaseE9GateError("Gate E.9a A0/A1 recipes are not matched")
    _progress(
        "frozen_inputs_verified",
        e8_sha256=PHASE_E8_FROZEN_ARTIFACTS["gate_e8_result.json"],
        heldout_identity_sha256=design[
            "heldout_identity_schedule_sha256"
        ],
        reserved_cohort_sha256=PHASE_E9_RESERVED_COHORT_SHA256,
        train_identity_sha256=design[
            "train_identity_schedule_sha256"
        ],
    )
    _progress("model_load_started", device="cuda:0")
    model, upstream_cfg, model_report = _load_upstream_model(cfg)
    torch.cuda.synchronize("cuda:0")
    model_report["load_peak_mib"] = (
        int(torch.cuda.max_memory_allocated("cuda:0")) / 2**20
    )
    _progress(
        "model_loaded",
        load_peak_mib=model_report["load_peak_mib"],
    )
    prepared = prepare_real_training_data(
        cfg,
        model=model,
        upstream_cfg=upstream_cfg,
        device="cuda:0",
        progress=_progress,
        train_only_limit=8,
        train_only_offset=8,
    )
    data_report = dict(prepared.report)
    atomic_write_json(output / "data_preparation.json", data_report)
    source = data_report["current_source"]
    prepared_sample_ids = [
        sample.base_sample_id for sample in prepared.samples
    ]
    if (
        source["actual_future_read"] is not False
        or int(source["future_rgb_frames_decoded"]) != 0
        or data_report["future_rgb_used_as_input"] is not False
        or data_report["split_counts"] != {"train": 8, "development": 0}
        or data_report["available_split_counts"]
        != {"train": 28, "development": 4}
        or data_report["selection_mode"] != "ordered_train_window"
        or int(data_report["train_only_offset"]) != 8
        or data_report["sample_payload_sha256"]
        != PHASE_E6_SAMPLE_PAYLOAD_SHA256
        or prepared_sample_ids != cohort["sample_ids"]
    ):
        raise PhaseE9GateError("Gate E.9a data/cohort isolation failed")

    frozen_before = parameter_state_sha256(iter(model.named_parameters()))
    results: dict[str, Mapping[str, Any]] = {}
    execution_error: BaseException | None = None
    execution_traceback: str | None = None
    try:
        for recipe, variant in PHASE_E9_TRACKS:
            key = f"{recipe}/{variant}"
            protocol = _protocol(recipe)
            _progress("track_started", recipe=recipe, variant=variant)
            track_result = run_full_cohort_objective_aggregation(
                tracks[key],
                model=model,
                prepared=prepared,
                frozen_parameter_sha256=frozen_before,
                resume=resume,
                device="cuda:0",
                progress=_progress,
                protocol=protocol,
                sample_loss_weights=(
                    PHASE_E9_SAMPLE_LOSS_WEIGHTS
                    if recipe == "normalized"
                    else None
                ),
            )
            results[key] = track_result
            _progress(
                "track_complete",
                loss_reduction_fraction=track_result["outcome"][
                    "loss_reduction_fraction"
                ],
                non_worsened=track_result["outcome"][
                    "non_worsened_sample_count"
                ],
                recipe=recipe,
                variant=variant,
            )
    except BaseException as exc:
        execution_error = exc
        execution_traceback = traceback.format_exc()

    frozen_after = parameter_state_sha256(iter(model.named_parameters()))
    prevalidation = {
        "captured_at": _utc_now(),
        "data_preparation": data_report,
        "e8_frozen": e8,
        "execution_error": (
            None
            if execution_error is None
            else f"{type(execution_error).__name__}: {execution_error}"
        ),
        "execution_traceback": execution_traceback,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "model_load": model_report,
        "phase_d_frozen": phase_d,
        "reserved_e9b_cohort": reserved,
        "schema_version": PHASE_E9_SCHEMA,
        "tracks": {key: dict(value) for key, value in results.items()},
    }
    atomic_write_json(output / "pre_validation_result.json", prevalidation)
    _progress("frozen_hash_after", sha256=frozen_after)
    if execution_error is not None:
        del prepared, upstream_cfg, model
        gc.collect()
        torch.cuda.empty_cache()
        raise PhaseE9GateError(
            "Gate E.9a track execution failed after frozen hash capture"
        ) from execution_error

    execution_checks = {}
    artifacts = {}
    performance = {}
    tail = {}
    for recipe, variant in PHASE_E9_TRACKS:
        key = f"{recipe}/{variant}"
        execution_checks[key], artifacts[key] = _track_checks(
            tracks[key],
            results[key],
            recipe=recipe,
            protocol=_protocol(recipe),
        )
        performance[key] = replication_performance_checks(
            variant,
            results[key],
        )
        tail[key] = paired_tail_bootstrap(
            results[key]["initial_probe"],
            results[key]["final_probe"],
            track_key=key,
        )
    normalized_paired, normalized_paired_values = (
        paired_superiority_checks(
            results["normalized/A0"],
            results["normalized/A1"],
        )
    )
    raw_paired, raw_paired_values = paired_superiority_checks(
        results["raw/A0"],
        results["raw/A1"],
    )
    schedule_hashes = {
        result["train_flow_schedule_sha256"]
        for result in results.values()
    }
    initial_signatures = {
        _initial_probe_signature(result) for result in results.values()
    }
    paired_checks = {
        "all_four_same_initial_adapter": (
            len(
                {
                    result["initial_adapter_sha256"]
                    for result in results.values()
                }
            )
            == 1
        ),
        "all_four_same_initial_probe": len(initial_signatures) == 1,
        "all_four_same_observed_schedule": len(schedule_hashes) == 1,
        "all_four_same_sample_ids": (
            len(
                {
                    tuple(result["sample_ids"])
                    for result in results.values()
                }
            )
            == 1
        ),
        "all_four_same_training_budget": all(
            int(result["completed_steps"]) == 200
            and int(result["completed_objectives"]) == 1_600
            for result in results.values()
        ),
        "only_gradient_weighting_differs_between_recipe_arms": (
            all(
                _matched_recipe_payload(tracks[f"raw/{variant}"])
                == _matched_recipe_payload(
                    tracks[f"normalized/{variant}"]
                )
                for variant in ("A0", "A1")
            )
            and results["raw/A0"]["gradient_reduction"]
            == results["raw/A1"]["gradient_reduction"]
            == "arithmetic_mean"
            and results["normalized/A0"]["gradient_reduction"]
            == results["normalized/A1"]["gradient_reduction"]
            == "fixed_sample_normalized_mean"
        ),
    }
    classification = classify_sample_tail_mitigation(
        performance,
        normalized_paired,
        tail,
    )
    repository_after = _verify_execution_repository()
    cross_checks = {
        "all_track_execution_checks_passed": all(
            all(values.values()) for values in execution_checks.values()
        ),
        "e8_artifacts_unchanged": all(
            sha256_file(PHASE_E8_ROOT / name) == expected
            for name, expected in PHASE_E8_FROZEN_ARTIFACTS.items()
        ),
        "frozen_fastwam_has_no_grad": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "frozen_fastwam_not_trainable": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "frozen_fastwam_unchanged": frozen_before == frozen_after,
        "paired_track_contracts_passed": all(paired_checks.values()),
        "repository_provenance_unchanged": (
            repository_after == dict(execution_repository)
        ),
        "reserved_cohort_not_decoded_or_trained": (
            reserved["decoded_or_trained_by_e9a"] is False
            and not set(reserved["sample_ids"])
            & set(prepared_sample_ids)
        ),
    }
    engineering_passed = all(cross_checks.values())
    result = {
        "completed_at": _utc_now(),
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "cross_checks": cross_checks,
        "data_preparation": data_report,
        "design_identity": design,
        "engineering_passed": engineering_passed,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameter_sha256_before": frozen_before,
        "gate_e9a_passed": engineering_passed,
        "mitigation_classification": classification,
        "model_load": model_report,
        "normalized_paired_checks": normalized_paired,
        "normalized_paired_values": normalized_paired_values,
        "paired_checks": paired_checks,
        "phase_d_frozen": phase_d,
        "phase_e8_frozen": e8,
        "post_selection_disclosure": {
            "e8_results_known_before_recipe_choice": True,
            "independent_confirmatory_test": False,
            "learning_rate_chosen_after_e5": True,
            "mitigation_chosen_after_e8": True,
            "weights_use_only_e8_zero_gate_initial_losses": True,
        },
        "preregistered_gate": {
            "a0_min_loss_reduction_fraction": 0.0,
            "a1_min_loss_reduction_fraction": 0.1,
            "a1_vs_a0_min_non_higher_samples": 6,
            "a1_vs_a0_min_relative_mean_improvement": 0.1,
            "bootstrap_replicates": PHASE_E9_BOOTSTRAP_REPLICATES,
            "familywise_alpha": PHASE_E9_FAMILYWISE_ALPHA,
            "familywise_comparisons": (
                PHASE_E9_FAMILYWISE_COMPARISONS
            ),
            "max_catastrophic_samples": 0,
            "max_confirmed_harmed_samples_per_normalized_track": 0,
            "max_median_delta_hidden_ratio": 0.5,
            "max_sample_delta_hidden_ratio": 1.0,
            "min_non_worsened_samples_per_variant": 6,
            "optimizer_updates_per_track": 200,
            "target_checkpoint_step": 200,
        },
        "raw_paired_checks": raw_paired,
        "raw_paired_values": raw_paired_values,
        "reserved_e9b_replication": reserved,
        "sample_loss_calibration": {
            "calibration_payload_sha256": (
                PHASE_E9_CALIBRATION_PAYLOAD_SHA256
            ),
            "formula": (
                "w_i=(1/L_i)/(mean_j(1/L_j)); fixed before E.9a; "
                "sum_i(w_i)=8"
            ),
            "initial_losses": dict(PHASE_E9_INITIAL_SAMPLE_LOSSES),
            "sample_loss_weights": dict(PHASE_E9_SAMPLE_LOSS_WEIGHTS),
            "sample_loss_weights_sha256": (
                PHASE_E9_SAMPLE_LOSS_WEIGHTS_SHA256
            ),
        },
        "schema_version": PHASE_E9_SCHEMA,
        "scope": {
            "backward_calls": 6_400,
            "development_outcomes_read": False,
            "future_rgb_frames_read": 0,
            "heldout_probe_objectives": 2_048,
            "new_training_samples_consumed": 0,
            "ood_outcomes_read": False,
            "optimizer_steps": 800,
            "reserved_replication_samples_decoded": 0,
            "rollout_started": False,
            "success_outcomes_read": False,
            "track_count": 4,
            "training_objectives": 6_400,
            "training_samples": 8,
            "uses_ground_truth_future": False,
        },
        "status": "complete" if engineering_passed else "invalid",
        "tail_bootstrap": tail,
        "tracks": {
            key: {
                "artifacts": artifacts[key],
                "execution_checks": execution_checks[key],
                "performance_checks": performance[key],
                "result": dict(results[key]),
            }
            for key in results
        },
    }
    del prepared, upstream_cfg, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_phase_e9_sample_tail_mitigation(
    cfg: Thought3Config,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run E.9a once; negative scientific classifications remain valid."""

    _assert_phase_e9_scope(cfg)
    _require_phase_e9_confirmation()
    output = ensure_thought3_output_path(cfg.experiment.output_dir)
    result_path = output / "gate_e9a_result.json"
    status_path = output / "run_status.json"
    if result_path.is_file():
        existing = load_json(result_path)
        if resume and existing.get("engineering_passed") is True:
            return existing
        if resume:
            raise PhaseE9GateError(
                "existing Gate E.9a result is invalid; preserve this Run ID"
            )
        raise FileExistsError(
            f"Gate E.9a result exists; pass --resume: {result_path}"
        )
    if status_path.is_file() and not resume:
        raise PhaseE9GateError(
            "existing partial Gate E.9a requires --resume or a new Run ID"
        )
    execution_repository = _verify_execution_repository()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        status_path,
        {
            "schema_version": PHASE_E9_SCHEMA,
            "started_at": _utc_now(),
            "status": "running",
        },
    )
    started = time.perf_counter()
    try:
        result = _run_phase_e9(
            cfg,
            resume=resume,
            execution_repository=execution_repository,
        )
        result["gate_wall_s"] = time.perf_counter() - started
        atomic_write_json(result_path, result)
        if result["engineering_passed"] is not True:
            raise PhaseE9GateError(
                "Gate E.9a engineering checks failed; inspect result"
            )
    except BaseException as exc:
        atomic_write_json(
            status_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": _utc_now(),
                "gate_e9a_passed": False,
                "result": (
                    str(result_path.resolve())
                    if result_path.is_file()
                    else None
                ),
                "schema_version": PHASE_E9_SCHEMA,
                "status": "failed",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    atomic_write_json(
        status_path,
        {
            "finished_at": _utc_now(),
            "gate_e9a_passed": True,
            "mitigation_classification": result[
                "mitigation_classification"
            ]["classification"],
            "result": str(result_path.resolve()),
            "schema_version": PHASE_E9_SCHEMA,
            "status": "complete",
        },
    )
    return result
