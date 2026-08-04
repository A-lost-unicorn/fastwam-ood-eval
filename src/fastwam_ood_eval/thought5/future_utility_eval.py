"""Phase 5-C matched A0/A1/AS sensitivity, utility, and specificity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from fastwam_ood_eval.thought5.statistics import (
    grouped_bootstrap_mean,
    task_cluster_bootstrap,
)


@dataclass(frozen=True)
class FutureUtilityRecord:
    backbone: str
    adapter_variant: str
    task_id: str
    episode_id: str
    condition: str
    flow_slot: int
    action_noise_seed: int
    action_timestep_seed: int
    denoise_schedule_sha256: str
    loss: float
    action_sha256: str
    action_rms: float
    translation_rms: float | None = None
    rotation_rms: float | None = None
    gripper_rms: float | None = None

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.task_id, self.episode_id, self.condition, self.flow_slot


def validate_matched_counterfactuals(records: Sequence[FutureUtilityRecord]) -> None:
    groups: dict[
        tuple[str, tuple[str, str, str, int]], list[FutureUtilityRecord]
    ] = defaultdict(list)
    for row in records:
        if row.backbone not in {"B1", "G3", "G4"} or row.adapter_variant not in {"A0", "A1", "AS"}:
            raise ValueError("invalid future-utility backbone/adapter variant")
        if row.condition not in {"clean", "camera"}:
            raise ValueError("future utility is frozen to clean/camera")
        if not np.isfinite(row.loss) or row.loss < 0 or not np.isfinite(row.action_rms):
            raise ValueError("future-utility record is non-finite")
        groups[(row.backbone, row.key)].append(row)
    for (_backbone, _key), values in groups.items():
        if {row.adapter_variant for row in values} != {"A0", "A1", "AS"} or len(values) != 3:
            raise ValueError("each future counterfactual needs exactly A0/A1/AS")
        identities = {
            (
                row.action_noise_seed,
                row.action_timestep_seed,
                row.denoise_schedule_sha256,
            )
            for row in values
        }
        if len(identities) != 1:
            raise ValueError("A0/A1/AS action noise, timestep, or schedule differs")


def evaluate_future_utility(
    records: Sequence[FutureUtilityRecord],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 5577,
    g4_equivalence_fraction: float = 0.8,
) -> dict[str, Any]:
    validate_matched_counterfactuals(records)
    groups: dict[
        tuple[str, tuple[str, str, str, int]], dict[str, FutureUtilityRecord]
    ] = defaultdict(dict)
    for row in records:
        groups[(row.backbone, row.key)][row.adapter_variant] = row
    rows: list[dict[str, Any]] = []
    by_backbone: dict[
        str, dict[tuple[str, str, str, int], dict[str, float]]
    ] = defaultdict(dict)
    for (backbone, key), values in sorted(groups.items()):
        losses = {variant: values[variant].loss for variant in ("A0", "A1", "AS")}
        utility = losses["A0"] - losses["A1"]
        specificity = losses["AS"] - losses["A1"]
        relative = utility / max(losses["A0"], 1e-12)
        sensitivity = {
            "correct_null_action_changed": values["A1"].action_sha256
            != values["A0"].action_sha256,
            "correct_shuffle_action_changed": values["A1"].action_sha256
            != values["AS"].action_sha256,
            "correct_null_rms_delta": abs(values["A1"].action_rms - values["A0"].action_rms),
            "correct_shuffle_rms_delta": abs(values["A1"].action_rms - values["AS"].action_rms),
            "translation_rms": values["A1"].translation_rms,
            "rotation_rms": values["A1"].rotation_rms,
            "gripper_rms": values["A1"].gripper_rms,
        }
        rows.append(
            {
                "backbone": backbone,
                "task_id": key[0],
                "episode_id": key[1],
                "condition": key[2],
                "flow_slot": key[3],
                "losses": losses,
                "utility": utility,
                "relative_utility": relative,
                "specificity": specificity,
                "sensitivity": sensitivity,
            }
        )
        by_backbone[backbone][key] = {
            "utility": utility,
            "specificity": specificity,
            "a0": losses["A0"],
        }
    if not {"B1", "G3"}.issubset(by_backbone) or set(
        by_backbone["B1"]
    ) != set(by_backbone["G3"]):
        raise ValueError("B1/G3 future-utility panels are not matched")
    keys = set(by_backbone["B1"])
    utility_gain = {
        "/".join(map(str, key)): [
            by_backbone["G3"][key]["utility"] - by_backbone["B1"][key]["utility"]
        ]
        for key in keys
    }
    interval = grouped_bootstrap_mean(
        utility_gain,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    task_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key in keys:
        task, episode, condition, _flow_slot = key
        task_values[task][f"{episode}/{condition}"].append(
            by_backbone["G3"][key]["utility"]
            - by_backbone["B1"][key]["utility"]
        )
    formal_multitask = len(task_values) >= 2
    if formal_multitask:
        task_interval: dict[str, Any] = asdict(
            task_cluster_bootstrap(
                task_values,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 1,
            )
        )
    else:
        task_interval = {
            "status": "PILOT_SINGLE_TASK_NOT_INFERENTIAL",
            "task_count": len(task_values),
            "estimate": float(
                np.mean(
                    [
                        by_backbone["G3"][key]["utility"]
                        - by_backbone["B1"][key]["utility"]
                        for key in keys
                    ]
                )
            ),
            "lower": None,
            "upper": None,
        }
    g3 = by_backbone["G3"]
    g3_utility_groups = {
        "/".join(map(str, key)): [values["utility"]]
        for key, values in g3.items()
    }
    g3_specificity_groups = {
        "/".join(map(str, key)): [values["specificity"]]
        for key, values in g3.items()
    }
    g3_utility_interval = grouped_bootstrap_mean(
        g3_utility_groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 2,
    )
    g3_specificity_interval = grouped_bootstrap_mean(
        g3_specificity_groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 3,
    )
    b1_a0 = np.mean([values["a0"] for values in by_backbone["B1"].values()])
    g3_a0 = np.mean([values["a0"] for values in g3.values()])
    g4_matches: bool | None = None
    if "G4" in by_backbone:
        if set(by_backbone["G4"]) != keys:
            raise ValueError("B1/G4 future-utility panels are not matched")
        b1_mean_utility = float(
            np.mean([value["utility"] for value in by_backbone["B1"].values()])
        )
        g3_mean_utility = float(
            np.mean([value["utility"] for value in by_backbone["G3"].values()])
        )
        g4_mean_utility = float(
            np.mean([value["utility"] for value in by_backbone["G4"].values()])
        )
        g3_gain = g3_mean_utility - b1_mean_utility
        g4_gain = g4_mean_utility - b1_mean_utility
        g4_matches = bool(
            g3_gain <= 0
            or g4_gain >= g4_equivalence_fraction * g3_gain
        )
    pilot_direction = bool(
        g3_utility_interval.lower > 0
        and g3_specificity_interval.lower > 0
        and interval.lower > 0
        and g3_a0 <= b1_a0 * 1.05
        and g4_matches is not True
    )
    h2 = bool(
        formal_multitask
        and pilot_direction
        and float(task_interval["lower"]) > 0
    )
    return {
        "schema_version": "thought5.phase5.future_utility_results.v1",
        "status": "complete",
        "adapter_recipe": {
            "architecture": "Thought3 FutureToActionAdapter unchanged",
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "steps": 200,
            "seed": 3407,
            "checkpoint_rule": "fixed_step_200",
        },
        "rows": rows,
        "utility_g3_minus_b1_grouped_bootstrap": asdict(interval),
        "utility_g3_minus_b1_task_cluster_bootstrap": task_interval,
        "g3_correct_minus_null_utility_grouped_bootstrap": asdict(
            g3_utility_interval
        ),
        "g3_correct_minus_shuffle_specificity_grouped_bootstrap": asdict(
            g3_specificity_interval
        ),
        "a0_b1_mean_loss": float(b1_a0),
        "a0_g3_mean_loss": float(g3_a0),
        "a0_g3_not_abnormally_worse": bool(g3_a0 <= b1_a0 * 1.05),
        "pilot_direction_observed": pilot_direction,
        "shuffled_control_matches_gain": g4_matches,
        "g4_equivalence_fraction": g4_equivalence_fraction,
        "formal_multitask_inference": formal_multitask,
        "h2_supported": h2,
    }


def not_run_future_utility_result() -> dict[str, Any]:
    return {
        "schema_version": "thought5.phase5.future_utility_results.v1",
        "status": "NOT RUN",
        "adapter_variants": ["A0", "A1", "AS"],
        "h2_supported": None,
    }
