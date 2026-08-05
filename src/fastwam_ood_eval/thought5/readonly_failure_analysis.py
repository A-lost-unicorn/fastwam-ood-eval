"""Post-hoc, read-only failure decomposition for the sealed Thought5 pilot v4.

The analyzer deliberately consumes only artifacts that already exist.  It does
not load Fast-WAM, use a GPU, train a parameter, render an environment, or run a
rollout.  Derived files are written to a sibling namespace so the sealed pilot
directory is never modified.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fastwam_ood_eval.thought5.checkpointing import tensor_state_sha256
from fastwam_ood_eval.thought5.ray_pose_encoder import RayPoseEncoder
from fastwam_ood_eval.thought5.schemas import (
    Thought5ArtifactError,
    file_sha256,
    object_sha256,
    seal_full_object,
)
from fastwam_ood_eval.thought5.statistics import grouped_bootstrap_mean


class ReadonlyFailureAnalysisError(Thought5ArtifactError):
    """Raised when a frozen input or a diagnostic invariant is violated."""


VARIANTS = ("B1", "G3", "G4")
UTILITY_CONDITIONS = ("clean", "camera")
ROLLOUT_CONDITIONS = ("clean", "camera", "lighting", "robot_init")
CONTRASTS = ("correct_null", "correct_shuffle", "null_shuffle")
ACTION_SEGMENTS: Mapping[str, tuple[int, ...]] = {
    # The rollout executes the first ten predictions from each 32-step chunk.
    "executed_prefix_0_9": tuple(range(0, 10)),
    "unexecuted_middle_10_20": tuple(range(10, 21)),
    "unexecuted_tail_21_31": tuple(range(21, 32)),
}
NOISE_BINS: tuple[tuple[str, float, float], ...] = (
    ("[0.00,0.25)", 0.00, 0.25),
    ("[0.25,0.50)", 0.25, 0.50),
    ("[0.50,0.75)", 0.50, 0.75),
    ("[0.75,1.00]", 0.75, 1.000001),
)

# Exact bytes used by the analysis.  This closes the read-only boundary and
# prevents an accidental rerun or edited result from being analyzed as pilot v4.
EXPECTED_SOURCE_SHA256: Mapping[str, str] = {
    "run_status.json": "165f2657317b49b71e74c720096292d82236d174cb5a2a0226e4fd5afe46d3b8",
    "pilot_direction.json": "1bb2944d196253b7002daaa87f340bbad61c86e4e7e2a3e05f0c0fbe46a98c3d",
    "training_results.json": "2c46a539bb4c2fc27c138f593694d29fa26d755bd069dece16ad3f0ed6313b36",
    "representation_results.json": "5604d8ff8f6de52ae1744feb856f0678dcddb19e3f5e953fdfef6be52b8a5efa",
    "future_geometry_results.json": "582d836ff2b28eb2febef5627007975c460be7a83c73bbc2f5ebe7b1bbdaa4a2",
    "future_utility_results.json": "612c8f8cecdf9d3ade8e033aa6d996de4dd083d676c8a03bcd382a543d998bba",
    "rollout_results.json": "fac48a4b92a59820c77efcd6ca1e64bdc24ca0640fa26202b0782a2da943aa3e",
    "tracks/b1/track_result.json": "d21f99a424c1511e6cf9da402e94e936a71c2f102858dfad7206938b567b985c",
    "tracks/b1/representation_bundle.pkl": "2a72561130d9ec52cc209eafbe64e8d86003ad77d33782ed53e383f04509738c",
    "tracks/b1/future_bundle.pkl": "1c377c7e0731b6419a17f5ce150c53daf3997ea0990b001eb52c975561fccf50",
    "tracks/b1/utility/future_utility_bundle.pkl": "719e1d5b27ff1e69e1232508a3158858c5e50509b478f3cf283d5b640ffd1ebf",
    "tracks/b1/checkpoints/step_00000100/manifest.json": "f8d99e8eddba65a0beb38c8bd214f9ae985cec265aadea5752633a73a24887f2",
    "tracks/b1/checkpoints/step_00000100/geoeq_state.pt": "76b697aea5f57817cc87eb4740ba13660c4e4c70e48f8d96c9f57bbfd2b12512",
    "tracks/g3/track_result.json": "cd78d870c75466f60e27a531bddae04debf501a7bfd5b5e64eee840a2a4e8399",
    "tracks/g3/representation_bundle.pkl": "998049f17ee098912d7e06ac70d7f262ee20445d39cd9bbafc187536c7cfb4f4",
    "tracks/g3/future_bundle.pkl": "4a7798e037aae06ce8a9beb2f7db16bdbda83a68fbeb48cfb084c3e7ec3f9d48",
    "tracks/g3/utility/future_utility_bundle.pkl": "95ca6e2769928233868e925dc1a7994827bc3647ada92fc87706fd9d64f03846",
    "tracks/g3/checkpoints/step_00000100/manifest.json": "dae87852cbdf1594ab309dd8d888f174e1fd1308ae1c4a4d3c8f9a8de90e4994",
    "tracks/g3/checkpoints/step_00000100/geoeq_state.pt": "8d7bd6cc1fd46a33ff223a9f177d9d8f36410eb0fec7e3eebf19c90d26659e98",
    "tracks/g4/track_result.json": "f86004b65c81485e000313833f2c2fe5b77938ec0daeb5571a5b510a9d7a66c7",
    "tracks/g4/representation_bundle.pkl": "ea9d476360aa1a8f0fe46eba7938cf78b3b3cf0169f0579bde474f1b56ccd29c",
    "tracks/g4/future_bundle.pkl": "7748bef14e504f0c36945995a663260e9387d981e53bfb53c1801a54db813e5a",
    "tracks/g4/utility/future_utility_bundle.pkl": "c301525bc0ee990e27807cd85a42d58d82212898fc2ed9008d9e61fa39fc9622",
    "tracks/g4/checkpoints/step_00000100/manifest.json": "ea292684767ff2442c0b0ab220506965a3f8aea1524db44a484079048cbf9569",
    "tracks/g4/checkpoints/step_00000100/geoeq_state.pt": "b11e9c6cc1c41107e9fc3b3cbf1b03c026f28f2140b3aad1ec25e3de01c29e1a",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReadonlyFailureAnalysisError(f"JSON root is not an object: {path}")
    return value


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ReadonlyFailureAnalysisError(f"pickle root is not a mapping: {path}")
    return value


def validate_frozen_sources(source_root: Path) -> dict[str, str]:
    """Validate exact source bytes and the already-sealed negative decision."""

    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file():
            raise ReadonlyFailureAnalysisError(f"frozen source is absent: {path}")
        digest = file_sha256(path)
        if digest != expected:
            raise ReadonlyFailureAnalysisError(
                f"frozen source checksum differs: {relative}: {digest} != {expected}"
            )
        observed[relative] = digest

    run_status = _load_json(source_root / "run_status.json")
    direction = _load_json(source_root / "pilot_direction.json")
    if run_status.get("status") != "complete" or run_status.get("stage") != "pilot":
        raise ReadonlyFailureAnalysisError("pilot v4 is not a completed pilot")
    if run_status.get("formal_unlocked") is not False:
        raise ReadonlyFailureAnalysisError("pilot v4 formal lock was not preserved")
    if direction.get("formal_unlocked") is not False:
        raise ReadonlyFailureAnalysisError("direction artifact unexpectedly unlocks formal")
    if direction.get("g3_direction_observed") is not False:
        raise ReadonlyFailureAnalysisError("negative pilot direction unexpectedly changed")
    if (source_root / "formal_protocol_frozen.json").exists():
        raise ReadonlyFailureAnalysisError("formal protocol exists despite the negative gate")
    return observed


def _interval_dict(values_by_group: Mapping[str, Sequence[float]], seed: int) -> dict[str, Any]:
    return asdict(
        grouped_bootstrap_mean(values_by_group, replicates=2000, seed=seed)
    )


def summarize_condition_utility(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return post-hoc condition summaries without changing the frozen gate."""

    expected_conditions = set(UTILITY_CONDITIONS)
    observed_conditions = {str(row["condition"]) for row in rows}
    if observed_conditions != expected_conditions:
        raise ReadonlyFailureAnalysisError(
            f"utility conditions differ: {observed_conditions}"
        )
    result: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    by_variant_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant_condition[(str(row["backbone"]), str(row["condition"]))].append(row)

    seed = 5801
    for variant in VARIANTS:
        result[variant] = {"conditions": {}}
        for condition in UTILITY_CONDITIONS:
            values = by_variant_condition[(variant, condition)]
            if len(values) != 128:
                raise ReadonlyFailureAnalysisError(
                    f"{variant}/{condition} expected 128 utility rows, got {len(values)}"
                )
            utilities = [float(row["utility"]) for row in values]
            specificities = [float(row["specificity"]) for row in values]
            losses = {
                name: fmean(float(row["losses"][name]) for row in values)
                for name in ("A0", "A1", "AS")
            }
            by_episode: dict[str, list[float]] = defaultdict(list)
            for row in values:
                by_episode[f'{row["task_id"]}/{row["episode_id"]}'].append(
                    float(row["utility"])
                )
            summary = {
                "row_count": len(values),
                "episode_count": len(by_episode),
                "mean_loss": losses,
                "mean_utility_a0_minus_a1": fmean(utilities),
                "mean_specificity_as_minus_a1": fmean(specificities),
                "negative_utility_count": sum(value < 0 for value in utilities),
                "negative_utility_fraction": fmean(value < 0 for value in utilities),
                "posthoc_episode_grouped_bootstrap": _interval_dict(by_episode, seed),
            }
            seed += 1
            result[variant]["conditions"][condition] = summary
            csv_rows.append(
                {
                    "variant": variant,
                    "condition": condition,
                    **{f"mean_loss_{key.lower()}": value for key, value in losses.items()},
                    "mean_utility_a0_minus_a1": summary["mean_utility_a0_minus_a1"],
                    "mean_specificity_as_minus_a1": summary["mean_specificity_as_minus_a1"],
                    "negative_utility_count": summary["negative_utility_count"],
                    "row_count": summary["row_count"],
                    "bootstrap_lower": summary["posthoc_episode_grouped_bootstrap"]["lower"],
                    "bootstrap_upper": summary["posthoc_episode_grouped_bootstrap"]["upper"],
                }
            )

        indexed: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
        for condition in UTILITY_CONDITIONS:
            for row in by_variant_condition[(variant, condition)]:
                key = (str(row["task_id"]), str(row["episode_id"]), int(row["flow_slot"]))
                indexed[key][condition] = float(row["utility"])
        if any(set(pair) != expected_conditions for pair in indexed.values()):
            raise ReadonlyFailureAnalysisError(f"{variant} clean/camera rows are not paired")
        differences: dict[str, list[float]] = defaultdict(list)
        for (task, episode, _slot), pair in indexed.items():
            differences[f"{task}/{episode}"].append(pair["camera"] - pair["clean"])
        result[variant]["camera_minus_clean"] = _interval_dict(differences, seed)
        seed += 1

    result["availability"] = {
        "future_utility": {
            "clean": "available",
            "camera": "available",
            "lighting": "unavailable_not_collected",
            "robot_init": "unavailable_not_collected",
        },
        "rollout_success": {condition: "available_4_episodes" for condition in ROLLOUT_CONDITIONS},
        "warning": (
            "Lighting and robot-init success cannot be substituted for a missing "
            "future-utility decomposition."
        ),
    }
    return result, csv_rows


def effective_flow_sigma(action_timestep_seed: int) -> float:
    """Reconstruct the exact BF16 action-objective sigma without a model load."""

    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(action_timestep_seed))
    uniform = torch.rand((1,), generator=generator, dtype=torch.float32)
    sigma_fp32 = 5.0 * uniform / (1.0 + 4.0 * uniform)
    timestep_bf16 = (sigma_fp32 * 1000.0).to(dtype=torch.bfloat16)
    sigma_bf16 = timestep_bf16 / 1000.0
    return float(sigma_bf16.float().item())


def _seed_map_from_utility_bundle(bundle: Mapping[str, Any]) -> dict[tuple[str, str, str, int], int]:
    mapping: dict[tuple[str, str, str, int], set[int]] = defaultdict(set)
    for row in bundle["records"]:
        key = (row.task_id, row.episode_id, row.condition, int(row.flow_slot))
        mapping[key].add(int(row.action_timestep_seed))
    if any(len(values) != 1 for values in mapping.values()):
        raise ReadonlyFailureAnalysisError("paired A0/A1/AS timestep seeds differ")
    return {key: next(iter(values)) for key, values in mapping.items()}


def summarize_flow_objectives(
    rows: Sequence[Mapping[str, Any]], source_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Localize utility by unordered flow seed slot and sampled noise level."""

    all_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        bundle = _load_pickle(
            source_root / f"tracks/{variant.lower()}/utility/future_utility_bundle.pkl"
        )
        if bundle.get("status") != "complete" or bundle.get("backbone") != variant:
            raise ReadonlyFailureAnalysisError(f"invalid {variant} utility bundle")
        if list(bundle.get("formal_flow_steps", [])) != list(range(171, 203)):
            raise ReadonlyFailureAnalysisError(f"{variant} flow slots are not frozen 171..202")
        seed_map = _seed_map_from_utility_bundle(bundle)
        variant_rows = [row for row in rows if row["backbone"] == variant]
        for row in variant_rows:
            key = (
                str(row["task_id"]),
                str(row["episode_id"]),
                str(row["condition"]),
                int(row["flow_slot"]),
            )
            if key not in seed_map:
                raise ReadonlyFailureAnalysisError(f"missing timestep seed for {variant}/{key}")
            all_rows.append(
                {
                    "variant": variant,
                    "task_id": key[0],
                    "episode_id": key[1],
                    "condition": key[2],
                    "flow_slot": key[3],
                    "action_timestep_seed": seed_map[key],
                    "effective_sigma": effective_flow_sigma(seed_map[key]),
                    "utility_a0_minus_a1": float(row["utility"]),
                    "specificity_as_minus_a1": float(row["specificity"]),
                }
            )

    result: dict[str, Any] = {
        "flow_slot_semantics": (
            "flow_slot 171..202 is an unordered deterministic identity/seed slot; "
            "it is neither an action-horizon index nor an inference denoising iteration."
        ),
        "effective_sigma_semantics": (
            "Reconstructed BF16 sigma for the weighted training-style flow objective: "
            "near 0 is low noise/near target and near 1 is high noise."
        ),
        "variants": {},
    }
    for variant in VARIANTS:
        values = [row for row in all_rows if row["variant"] == variant]
        slots: dict[int, list[float]] = defaultdict(list)
        for row in values:
            slots[int(row["flow_slot"])].append(float(row["utility_a0_minus_a1"]))
        slot_rows = [
            {
                "flow_slot": slot,
                "mean_utility": fmean(slot_values),
                "negative_count": sum(value < 0 for value in slot_values),
                "row_count": len(slot_values),
            }
            for slot, slot_values in sorted(slots.items())
        ]
        binned: dict[str, Any] = {}
        for label, lower, upper in NOISE_BINS:
            selected = [
                row for row in values if lower <= float(row["effective_sigma"]) < upper
            ]
            binned[label] = {
                "row_count": len(selected),
                "mean_utility": fmean(
                    float(row["utility_a0_minus_a1"]) for row in selected
                ),
                "negative_utility_fraction": fmean(
                    float(row["utility_a0_minus_a1"]) < 0 for row in selected
                ),
                "by_condition": {
                    condition: {
                        "row_count": len(
                            conditioned := [
                                row for row in selected if row["condition"] == condition
                            ]
                        ),
                        "mean_utility": fmean(
                            float(row["utility_a0_minus_a1"])
                            for row in conditioned
                        ),
                    }
                    for condition in UTILITY_CONDITIONS
                },
            }
        sigmas = np.asarray([row["effective_sigma"] for row in values], dtype=np.float64)
        utilities = np.asarray(
            [row["utility_a0_minus_a1"] for row in values], dtype=np.float64
        )
        result["variants"][variant] = {
            "slot_count": len(slot_rows),
            "slot_mean_negative_count": sum(row["mean_utility"] < 0 for row in slot_rows),
            "individual_negative_count": int(np.sum(utilities < 0)),
            "individual_row_count": len(values),
            "worst_five_slots": sorted(slot_rows, key=lambda row: row["mean_utility"])[:5],
            "best_five_slots": sorted(slot_rows, key=lambda row: row["mean_utility"], reverse=True)[:5],
            "noise_bins": binned,
            "pearson_sigma_utility": float(np.corrcoef(sigmas, utilities)[0, 1]),
            "slot_rows": slot_rows,
        }
    return result, all_rows


def summarize_action_horizon(
    technical: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize final action-change magnitude, not per-slot utility loss."""

    result: dict[str, Any] = {
        "metric_semantics": (
            "Mean L2 change of each predicted action vector after a correct-null, "
            "correct-shuffle, or null-shuffle intervention. This is sensitivity, "
            "not target loss or rollout harm."
        ),
        "segments": {name: list(indices) for name, indices in ACTION_SEGMENTS.items()},
        "variants": {},
        "inference_denoising_localization": {
            "status": "unavailable_from_existing_artifacts",
            "recorded_inference_steps": 20,
            "reason": (
                "Only final 32-step action chunks and a schedule hash were retained; "
                "no intermediate denoising-state actions, latents, or losses exist."
            ),
        },
        "per_action_slot_utility": {
            "status": "unavailable_from_existing_artifacts",
            "reason": "The saved utility objective is reduced over the complete action chunk.",
        },
    }
    csv_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        panel = technical[variant]
        rows = panel["rows"]
        if panel.get("status") != "complete" or len(rows) != 8:
            raise ReadonlyFailureAnalysisError(f"invalid {variant} action sensitivity panel")
        if int(panel.get("action_denoise_steps", -1)) != 20:
            raise ReadonlyFailureAnalysisError(f"{variant} denoise step count differs")
        result["variants"][variant] = {}
        for contrast in CONTRASTS:
            condition_rows: dict[str, list[Mapping[str, Any]]] = {
                "all": list(rows),
                "clean": [row for row in rows if str(row["sample_id"]).endswith(":clean")],
                "camera": [row for row in rows if str(row["sample_id"]).endswith(":camera")],
            }
            contrast_result: dict[str, Any] = {}
            for condition, selected in condition_rows.items():
                if not selected:
                    raise ReadonlyFailureAnalysisError(
                        f"{variant}/{contrast}/{condition} has no sensitivity rows"
                    )
                per_timestep = [
                    fmean(float(row[contrast]["per_timestep_l2"][index]) for row in selected)
                    for index in range(32)
                ]
                segment_means = {
                    name: fmean(per_timestep[index] for index in indices)
                    for name, indices in ACTION_SEGMENTS.items()
                }
                contrast_result[condition] = {
                    "sample_count": len(selected),
                    "overall_mean_l2": fmean(per_timestep),
                    "segment_mean_l2": segment_means,
                    "peak_action_index": int(np.argmax(per_timestep)),
                    "peak_mean_l2": max(per_timestep),
                    "tail_over_executed_prefix_ratio": (
                        segment_means["unexecuted_tail_21_31"]
                        / segment_means["executed_prefix_0_9"]
                    ),
                }
                for index, value in enumerate(per_timestep):
                    csv_rows.append(
                        {
                            "variant": variant,
                            "contrast": contrast,
                            "condition": condition,
                            "action_index": index,
                            "mean_l2_change": value,
                        }
                    )
            result["variants"][variant][contrast] = contrast_result
    return result, csv_rows


def _torch_load_state(path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ReadonlyFailureAnalysisError(f"invalid checkpoint state: {path}")
    return state


def _validate_checkpoint(source_root: Path, variant: str) -> dict[str, Any]:
    directory = source_root / f"tracks/{variant.lower()}/checkpoints/step_00000100"
    manifest = _load_json(directory / "manifest.json")
    stored_manifest_sha = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if stored_manifest_sha != object_sha256(unhashed):
        raise ReadonlyFailureAnalysisError(f"{variant} checkpoint manifest is invalid")
    for filename, digest in manifest["files_sha256"].items():
        if file_sha256(directory / filename) != digest:
            raise ReadonlyFailureAnalysisError(f"{variant} checkpoint file is invalid")
    state = _torch_load_state(directory / "geoeq_state.pt")
    if tensor_state_sha256(state) != manifest["state_sha256"]:
        raise ReadonlyFailureAnalysisError(f"{variant} semantic checkpoint hash differs")
    return state


def low_rank_delta_statistics(
    left_a: Any,
    left_b: Any,
    right_a: Any | None = None,
    right_b: Any | None = None,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    """Frobenius/cosine statistics without materializing a 3072x3072 delta."""

    import torch

    la = left_a.detach().float()
    lb = left_b.detach().float()
    ra = la if right_a is None else right_a.detach().float()
    rb = lb if right_b is None else right_b.detach().float()
    left_inner = torch.sum((lb.T @ lb) * (la @ la.T)) * (scale**2)
    right_inner = torch.sum((rb.T @ rb) * (ra @ ra.T)) * (scale**2)
    cross = torch.sum((lb.T @ rb) * (la @ ra.T)) * (scale**2)
    left_norm = float(torch.sqrt(torch.clamp(left_inner, min=0)).item())
    right_norm = float(torch.sqrt(torch.clamp(right_inner, min=0)).item())
    cross_value = float(cross.item())
    difference_squared = max(
        0.0, left_norm * left_norm + right_norm * right_norm - 2.0 * cross_value
    )
    difference_norm = math.sqrt(difference_squared)
    cosine = cross_value / max(left_norm * right_norm, 1e-30)
    return {
        "left_frobenius": left_norm,
        "right_frobenius": right_norm,
        "cosine": cosine,
        "difference_frobenius": difference_norm,
        "difference_over_left": difference_norm / max(left_norm, 1e-30),
    }


def _tensor_group_similarity(
    left: Mapping[str, Any], right: Mapping[str, Any], keys: Iterable[str]
) -> dict[str, float]:
    import torch

    chosen = tuple(sorted(keys))
    if not chosen:
        raise ReadonlyFailureAnalysisError("empty tensor comparison group")
    left_norm_sq = 0.0
    right_norm_sq = 0.0
    difference_norm_sq = 0.0
    inner = 0.0
    for key in chosen:
        lhs = left[key].detach().float().reshape(-1)
        rhs = right[key].detach().float().reshape(-1)
        if lhs.shape != rhs.shape:
            raise ReadonlyFailureAnalysisError(f"tensor shapes differ for {key}")
        left_norm_sq += float(torch.dot(lhs, lhs).item())
        right_norm_sq += float(torch.dot(rhs, rhs).item())
        inner += float(torch.dot(lhs, rhs).item())
        delta = lhs - rhs
        difference_norm_sq += float(torch.dot(delta, delta).item())
    left_norm = math.sqrt(left_norm_sq)
    right_norm = math.sqrt(right_norm_sq)
    difference_norm = math.sqrt(difference_norm_sq)
    return {
        "tensor_count": len(chosen),
        "left_l2": left_norm,
        "right_l2": right_norm,
        "cosine": inner / max(left_norm * right_norm, 1e-30),
        "difference_l2": difference_norm,
        "difference_over_left": difference_norm / max(left_norm, 1e-30),
    }


def _ray_injection_summary(
    state: Mapping[str, Any], future_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    encoder = RayPoseEncoder(model_dim=3072, hidden_dim=128, pose_dim=12)
    encoder.load_state_dict(
        {
            name.removeprefix("ray_pose_encoder."): tensor
            for name, tensor in state.items()
            if name.startswith("ray_pose_encoder.")
        },
        strict=True,
    )
    encoder.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for entry in future_bundle["adapter_entries"]:
            pose = entry.pose_12.float()
            rays = entry.rays.float()
            expanded_pose = pose[:, None, :].expand(-1, rays.shape[1], -1)
            raw = encoder.net(torch.cat([rays, expanded_pose], dim=-1))
            injected = torch.tanh(encoder.gate) * raw
            rows.append(
                {
                    "condition": entry.condition,
                    "split": entry.split,
                    "raw_encoder_rms": float(torch.sqrt(torch.mean(raw.square())).item()),
                    "injected_rms": float(
                        torch.sqrt(torch.mean(injected.square())).item()
                    ),
                    "injected_per_token_l2": float(
                        torch.linalg.vector_norm(injected, dim=-1).mean().item()
                    ),
                    "injected_abs_max": float(injected.abs().max().item()),
                }
            )

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "sample_count": len(selected),
            **{
                name: fmean(float(row[name]) for row in selected)
                for name in (
                    "raw_encoder_rms",
                    "injected_rms",
                    "injected_per_token_l2",
                    "injected_abs_max",
                )
            },
        }

    return {
        "all": aggregate(rows),
        "by_condition": {
            condition: aggregate([row for row in rows if row["condition"] == condition])
            for condition in sorted({str(row["condition"]) for row in rows})
        },
        "robot_init": {
            "status": "unavailable_not_in_adapter_bundle",
            "sample_count": 0,
        },
        "relative_to_backbone_hidden": {
            "status": "unavailable_from_existing_artifacts",
            "reason": "The pre-injection backbone hidden tensor was not retained.",
        },
    }


def _trajectory_similarity(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], key: str
) -> dict[str, float]:
    if [int(row["step"]) for row in left] != [int(row["step"]) for row in right]:
        raise ReadonlyFailureAnalysisError("G3/G4 trajectory steps differ")
    lhs = np.asarray([float(row[key]) for row in left], dtype=np.float64)
    rhs = np.asarray([float(row[key]) for row in right], dtype=np.float64)
    return {
        "pearson": float(np.corrcoef(lhs, rhs)[0, 1]),
        "mean_absolute_difference": float(np.mean(np.abs(lhs - rhs))),
        "final_g3": float(lhs[-1]),
        "final_g4": float(rhs[-1]),
    }


def summarize_parameter_use(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states = {variant: _validate_checkpoint(source_root, variant) for variant in VARIANTS}
    tracks = {
        variant: _load_json(source_root / f"tracks/{variant.lower()}/track_result.json")
        for variant in VARIANTS
    }
    initial_hashes = {track["initial_state_sha256"] for track in tracks.values()}
    if len(initial_hashes) != 1:
        raise ReadonlyFailureAnalysisError("matched variants do not share initialization")

    result: dict[str, Any] = {
        "matched_initial_state_sha256": next(iter(initial_hashes)),
        "gradient_observation_scope": {
            "recorded_steps": [1, 2],
            "later_steps": "not_recorded",
        },
        "checkpoint_trajectory_scope": {
            "available_steps": [100],
            "gate_and_parameter_norm_trajectory": "unavailable_from_existing_artifacts",
        },
        "variants": {},
    }
    trajectory_csv: list[dict[str, Any]] = []
    for variant in VARIANTS:
        state = states[variant]
        gate = float(state["ray_pose_encoder.gate"].float().item())
        gradients: dict[str, Any] = {}
        for row in tracks[variant]["training_rows"]:
            if row["gradients"] is not None:
                gradients[str(row["step"])] = row["gradients"]
            trajectory_csv.append(
                {
                    "variant": variant,
                    "split": "training",
                    "step": int(row["step"]),
                    **{key: float(value) for key, value in row["components"].items()},
                }
            )
        for row in tracks[variant]["development_rows"]:
            trajectory_csv.append(
                {
                    "variant": variant,
                    "split": "development",
                    "step": int(row["step"]),
                    **{key: float(value) for key, value in row.items() if key != "step"},
                }
            )
        future_bundle = _load_pickle(source_root / f"tracks/{variant.lower()}/future_bundle.pkl")
        injection = _ray_injection_summary(state, future_bundle)
        canonical = {
            key for key in state if key.startswith("backbone.video_expert.blocks.15")
        }
        if len(canonical) != 4:
            raise ReadonlyFailureAnalysisError(f"{variant} canonical LoRA key count differs")
        lora_delta: dict[str, Any] = {}
        for projection in ("k", "v"):
            prefix = f"backbone.video_expert.blocks.15.self_attn.{projection}"
            lora_delta[projection] = low_rank_delta_statistics(
                state[f"{prefix}.lora_A"], state[f"{prefix}.lora_B"]
            )["left_frobenius"]
        result["variants"][variant] = {
            "ray_pose_gate": gate,
            "ray_pose_tanh_gate": math.tanh(gate),
            "gradients": gradients,
            "effective_lora_delta_frobenius": lora_delta,
            "ray_pose_injection": injection,
            "final_development": tracks[variant]["development_rows"][-1],
        }

    comparisons: dict[str, Any] = {}
    for left_name, right_name in (("B1", "G3"), ("G3", "G4")):
        left = states[left_name]
        right = states[right_name]
        lora: dict[str, Any] = {}
        for projection in ("k", "v"):
            prefix = f"backbone.video_expert.blocks.15.self_attn.{projection}"
            lora[projection] = low_rank_delta_statistics(
                left[f"{prefix}.lora_A"],
                left[f"{prefix}.lora_B"],
                right[f"{prefix}.lora_A"],
                right[f"{prefix}.lora_B"],
            )
        comparisons[f"{left_name}_vs_{right_name}"] = {
            "effective_lora_delta": lora,
            "parameter_groups": {
                "canonical_lora_factors": _tensor_group_similarity(
                    left,
                    right,
                    [key for key in left if key.startswith("backbone.video_expert.blocks.15")],
                ),
                "geo_projector": _tensor_group_similarity(
                    left, right, [key for key in left if key.startswith("geo_projector.")]
                ),
                "ray_pose_without_gate": _tensor_group_similarity(
                    left,
                    right,
                    [
                        key
                        for key in left
                        if key.startswith("ray_pose_encoder.")
                        and key != "ray_pose_encoder.gate"
                    ],
                ),
            },
        }
    result["comparisons"] = comparisons

    g3_train = [row["components"] | {"step": row["step"]} for row in tracks["G3"]["training_rows"]]
    g4_train = [row["components"] | {"step": row["step"]} for row in tracks["G4"]["training_rows"]]
    result["g3_g4_training_trajectory"] = {
        key: _trajectory_similarity(g3_train, g4_train, key)
        for key in ("original_fastwam", "pose_aux", "equivariance", "geo_repa", "total")
    }
    result["ray_pose_use_interpretation"] = {
        "classification": "executed_nonzero_but_causal_contribution_not_isolated",
        "positive_evidence": [
            "G3/G4 final tanh(gate) is nonzero.",
            "RayPoseEncoder gradients are nonzero at recorded steps 1 and 2.",
            "Reconstructed final injected residual is nonzero.",
        ],
        "missing_identification": [
            "No stored G3 checkpoint with gate forced to zero.",
            "LoRA and RayPose changed together, so output changes cannot be assigned to RayPose.",
            "No pre-injection backbone hidden tensor, so relative residual magnitude is unavailable.",
        ],
    }
    return result, trajectory_csv


def _feature_delta_analysis(source_root: Path) -> dict[str, Any]:
    import torch

    bundles = {
        variant: _load_pickle(
            source_root / f"tracks/{variant.lower()}/representation_bundle.pkl"
        )
        for variant in VARIANTS
    }
    examples = {variant: bundles[variant]["examples"] for variant in VARIANTS}
    lengths = {len(value) for value in examples.values()}
    if lengths != {192}:
        raise ReadonlyFailureAnalysisError(f"representation example counts differ: {lengths}")
    rows: list[dict[str, Any]] = []
    identity_fields = (
        "sample_id",
        "episode_id",
        "split",
        "condition",
        "source",
        "module_path",
        "layer_index",
        "denoise_step_index",
        "pooling",
    )
    for index, triplet in enumerate(zip(*(examples[variant] for variant in VARIANTS))):
        b1, g3, g4 = triplet
        for field in identity_fields:
            if len({getattr(value, field) for value in triplet}) != 1:
                raise ReadonlyFailureAnalysisError(
                    f"representation identity differs at {index}/{field}"
                )
        delta_g3 = g3.feature.detach().float() - b1.feature.detach().float()
        delta_g4 = g4.feature.detach().float() - b1.feature.detach().float()
        norm_g3 = float(torch.linalg.vector_norm(delta_g3).item())
        norm_g4 = float(torch.linalg.vector_norm(delta_g4).item())
        cosine = float(
            torch.dot(delta_g3, delta_g4).item() / max(norm_g3 * norm_g4, 1e-30)
        )
        rows.append(
            {
                "source": b1.source,
                "condition": b1.condition,
                "g3_minus_b1_l2": norm_g3,
                "g4_minus_b1_l2": norm_g4,
                "g3_minus_g4_l2": float(
                    torch.linalg.vector_norm(g3.feature.float() - g4.feature.float()).item()
                ),
                "g3_g4_delta_cosine": cosine,
            }
        )

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        cosines = [float(row["g3_g4_delta_cosine"]) for row in selected]
        return {
            "row_count": len(selected),
            "mean_g3_minus_b1_l2": fmean(
                float(row["g3_minus_b1_l2"]) for row in selected
            ),
            "mean_g4_minus_b1_l2": fmean(
                float(row["g4_minus_b1_l2"]) for row in selected
            ),
            "mean_g3_minus_g4_l2": fmean(
                float(row["g3_minus_g4_l2"]) for row in selected
            ),
            "mean_g3_g4_delta_cosine": fmean(cosines),
            "median_g3_g4_delta_cosine": float(np.median(cosines)),
        }

    action_hash_changes: dict[str, int] = {}
    inference = {variant: bundles[variant]["inference_rows"] for variant in VARIANTS}
    for left, right in (("B1", "G3"), ("B1", "G4"), ("G3", "G4")):
        left_map = {
            (row["sample_id"], row["condition"], int(row["action_seed"])): row["action_sha256"]
            for row in inference[left]
        }
        right_map = {
            (row["sample_id"], row["condition"], int(row["action_seed"])): row["action_sha256"]
            for row in inference[right]
        }
        if set(left_map) != set(right_map):
            raise ReadonlyFailureAnalysisError(f"{left}/{right} inference rows are unmatched")
        action_hash_changes[f"{left}_vs_{right}"] = sum(
            left_map[key] != right_map[key] for key in left_map
        )
    return {
        "all": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in ("A", "B")
        },
        "by_condition": {
            condition: aggregate([row for row in rows if row["condition"] == condition])
            for condition in ROLLOUT_CONDITIONS
        },
        "action_hash_changes_out_of_64": action_hash_changes,
        "action_hash_warning": (
            "Hash changes establish different outputs, not magnitude, quality, or a "
            "causal RayPose-only effect."
        ),
    }


def summarize_g4_question(source_root: Path, parameter_use: Mapping[str, Any]) -> dict[str, Any]:
    representation = _load_json(source_root / "representation_results.json")
    future_geometry = _load_json(source_root / "future_geometry_results.json")
    b1_gap = float(representation["b1_camera_gap"])
    g3_gap = float(representation["g3_camera_gap"])
    g4_gap = float(representation["g4_camera_gap"])
    feature_delta = _feature_delta_analysis(source_root)
    return {
        "representation_gap": {
            "B1": b1_gap,
            "G3": g3_gap,
            "G4": g4_gap,
            "g3_reduction_vs_b1": b1_gap - g3_gap,
            "g4_reduction_vs_b1": b1_gap - g4_gap,
            "g4_minus_g3": g4_gap - g3_gap,
            "g3_grouped_ci": representation["g3_minus_b1_camera_grouped_bootstrap"],
        },
        "future_geometry_primary_camera_rmse": future_geometry["main_camera_error"],
        "feature_delta": feature_delta,
        "parameter_and_trajectory_similarity": {
            "g3_vs_g4": parameter_use["comparisons"]["G3_vs_G4"],
            "training": parameter_use["g3_g4_training_trajectory"],
        },
        "control_semantics": {
            "G3": "correct Geo-REPA target + correct equivariance/pose losses + RayPose",
            "G4": "shuffled Geo-REPA target + the same correct equivariance/pose losses + RayPose",
            "important_caveat": (
                "G4 shuffles only the per-sample Geo-REPA correspondence. It is not a "
                "control for every geometry signal and is not a pure regularization-only track."
            ),
        },
        "interpretation": {
            "classification": "correct_georepa_correspondence_not_identified_as_cause",
            "supported": (
                "G4's smaller gap, near-identical G3/G4 trajectory, and highly aligned "
                "video-feature deltas favor a shared loss/conditioning/regularization route "
                "over learning the correct per-sample Geo-REPA correspondence."
            ),
            "not_identifiable": (
                "The pilot omitted G1 (Geo-REPA without RayPose) and G2 (RayPose without "
                "Geo-REPA), so it cannot separate equivariance, pose conditioning, shared "
                "LoRA regularization, or their interaction."
            ),
        },
    }


def _rollout_condition_summary(source_root: Path) -> dict[str, Any]:
    rollout = _load_json(source_root / "rollout_results.json")
    return {
        variant: {
            condition: rollout["summaries"][f"{variant}:{condition}"]["success_rate"]
            for condition in ROLLOUT_CONDITIONS
        }
        for variant in VARIANTS
    }


def build_analysis(source_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    source_hashes_before = validate_frozen_sources(source_root)
    utility_result = _load_json(source_root / "future_utility_results.json")
    if utility_result.get("status") != "complete" or len(utility_result.get("rows", [])) != 768:
        raise ReadonlyFailureAnalysisError("future utility result is incomplete")
    condition, condition_csv = summarize_condition_utility(utility_result["rows"])
    condition["rollout_success_rate"] = _rollout_condition_summary(source_root)
    flow, flow_csv = summarize_flow_objectives(utility_result["rows"], source_root)
    horizon, horizon_csv = summarize_action_horizon(
        utility_result["technical_action_sensitivity"]
    )
    parameter_use, trajectory_csv = summarize_parameter_use(source_root)
    g4 = summarize_g4_question(source_root, parameter_use)
    source_hashes_after = validate_frozen_sources(source_root)
    if source_hashes_after != source_hashes_before:
        raise ReadonlyFailureAnalysisError("frozen input changed during analysis")

    result = seal_full_object(
        {
            "schema_version": "thought5.phase5.pilot_v4_readonly_failure_analysis.v1",
            "status": "complete",
            "analysis_role": "posthoc_readonly_exploratory_not_a_pilot_gate",
            "analyzer_provenance": {
                "module": "fastwam_ood_eval.thought5.readonly_failure_analysis",
                "module_sha256": file_sha256(Path(__file__).resolve()),
            },
            "source_root": str(source_root),
            "source_sha256": source_hashes_before,
            "immutability": {
                "source_hashes_equal_before_after": True,
                "gpu_used": False,
                "model_loaded": False,
                "training_run": False,
                "simulator_run": False,
                "rollout_run": False,
                "future_rgb_read": False,
                "success_outcome_used_for_tuning": False,
            },
            "pilot_decision": {
                "preserved": True,
                "g3_direction_observed": False,
                "formal_unlocked": False,
                "current_recipe_status": "stopped",
                "scientific_result": False,
            },
            "questions": {
                "condition_failure": condition,
                "flow_and_action_localization": {
                    "flow_objective": flow,
                    "action_horizon": horizon,
                },
                "raypose_lora_training_use": parameter_use,
                "why_g4_gap_is_smaller": g4,
            },
            "next_hypothesis": {
                "status": "diagnostic_hypothesis_only_not_unlocked_for_execution",
                "statement": (
                    "The next method should first target condition-aware/low-noise future "
                    "fusion and independently identify RayPose versus shared regularization; "
                    "the stopped G3 recipe should not advance unchanged."
                ),
                "evidence": [
                    "G3 mean utility is positive on Camera but negative on Clean.",
                    "G3 damage is concentrated at low effective sigma.",
                    "G3/G4 share nearly the same training and parameter trajectory.",
                    "Existing artifacts do not causally isolate RayPose or inference denoise steps.",
                ],
                "formal_phase": "locked",
            },
        }
    )
    tables = {
        "condition_utility.csv": condition_csv,
        "flow_objective_rows.csv": flow_csv,
        "action_horizon_sensitivity.csv": horizon_csv,
        "training_trajectory.csv": trajectory_csv,
    }
    return result, tables


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ReadonlyFailureAnalysisError(f"refusing to write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_number(value: float) -> str:
    return f"{value:+.6f}"


def render_report(result: Mapping[str, Any]) -> str:
    questions = result["questions"]
    condition = questions["condition_failure"]
    flow = questions["flow_and_action_localization"]["flow_objective"]
    horizon = questions["flow_and_action_localization"]["action_horizon"]
    parameters = questions["raypose_lora_training_use"]
    g4 = questions["why_g4_gap_is_smaller"]
    g3_clean = condition["G3"]["conditions"]["clean"]
    g3_camera = condition["G3"]["conditions"]["camera"]
    g3_flow = flow["variants"]["G3"]
    gate = parameters["variants"]["G3"]
    lora_similarity = parameters["comparisons"]["G3_vs_G4"]["parameter_groups"]
    feature = g4["feature_delta"]
    return f"""# Thought5 Pilot v4 只读失败分解

性质：**post-hoc、只读、探索性诊断**。它不是新的 Pilot Gate，不修改原判定，
不解锁 formal，也没有运行 GPU、模型、模拟器、训练或 rollout。

## 结论摘要

1. G3 的 aggregate future-utility 伤害主要来自 **Clean**：Clean 的
   `A0-A1={_format_number(g3_clean['mean_utility_a0_minus_a1'])}`，Camera 为
   `{_format_number(g3_camera['mean_utility_a0_minus_a1'])}`。因此不是所有条件都受害；
   Camera 的均值已转正，但单任务 4 episode 的 post-hoc 区间仍跨 0。Lighting 与
   Robot-init 没有被 future-utility collector 收集，不能用 rollout success 代替。
2. G3 有 {g3_flow['slot_mean_negative_count']}/32 个 unordered flow-slot 的均值为负。
   伤害集中在低噪声 objective：`sigma<0.25` 的均值为
   `{_format_number(g3_flow['noise_bins']['[0.00,0.25)']['mean_utility'])}`；
   `0.50<=sigma<0.75` 已为
   `{_format_number(g3_flow['noise_bins']['[0.50,0.75)']['mean_utility'])}`。
   32 步动作块的 final-action sensitivity 在 tail 略大，但没有逐 action-slot loss；
   20 次 inference denoising 也没有中间工件，所以不能定位某个去噪迭代。
3. RayPoseEncoder **确实执行并得到非零更新/注入**：G3 gate=`{gate['ray_pose_gate']:.9f}`，
   `tanh(gate)={gate['ray_pose_tanh_gate']:.9f}`，最终 injection RMS 约
   `{gate['ray_pose_injection']['all']['injected_rms']:.9f}`；记录的 step 1/2 梯度非零。
   但 gate 很小，且没有 gate-zero checkpoint ablation，LoRA 与 RayPose 同时变化，
   因而现有工件不能证明动作变化由 RayPose 单独造成。
4. G4 representation gap (`{g4['representation_gap']['G4']:.9f}`) 小于 G3
   (`{g4['representation_gap']['G3']:.9f}`)，但 G4 只 shuffle Geo-REPA target，
   仍保留正确的 equivariance/pose loss 与 RayPose。G3/G4 RayPose 非 gate 参数余弦为
   `{lora_similarity['ray_pose_without_gate']['cosine']:.9f}`，video-source feature-delta
   方向余弦为 `{feature['by_source']['A']['mean_g3_g4_delta_cosine']:.6f}`。
   这支持“正确逐样本 Geo-REPA 对应不是已识别原因”，更像共享条件/辅助损失/正则化
   路径；没有 G1/G2，仍不能进一步拆分。

## Condition 分解

| Variant | Clean utility | Camera utility | Clean negative | Camera negative |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {variant} | {_format_number(condition[variant]['conditions']['clean']['mean_utility_a0_minus_a1'])} "
        f"| {_format_number(condition[variant]['conditions']['camera']['mean_utility_a0_minus_a1'])} "
        f"| {condition[variant]['conditions']['clean']['negative_utility_count']}/128 "
        f"| {condition[variant]['conditions']['camera']['negative_utility_count']}/128 |"
        for variant in VARIANTS
    ) + f"""

Rollout 的 4-episode success 仅作已有结果描述：B1/G3 在 Clean、Camera、Lighting、
Robot-init 分别完全相同（0.25、0.25、1.00、1.00）；G4 Camera 为 0，其余为
0.25、1.00、1.00。它不能回答 Lighting/Robot-init 的 future utility。

## Flow / action / denoising 边界

- `flow_slot=171..202` 是 seed identity，不表示早/晚动作，也不表示第几次去噪。
- G3 sigma 与 utility 的 Pearson 相关为 `{g3_flow['pearson_sigma_utility']:.6f}`；
  这是 post-hoc 的 training-style objective noise-level 关联，不是因果结论。
- G3 correct-null final action change 的 executed prefix、middle、tail 均值分别为
  `{horizon['variants']['G3']['correct_null']['all']['segment_mean_l2']['executed_prefix_0_9']:.6f}`、
  `{horizon['variants']['G3']['correct_null']['all']['segment_mean_l2']['unexecuted_middle_10_20']:.6f}`、
  `{horizon['variants']['G3']['correct_null']['all']['segment_mean_l2']['unexecuted_tail_21_31']:.6f}`。
- 因为 utility loss 已对 32 步 action chunk 聚合，不能说 tail 是“伤害来源”；它只在
  technical sensitivity 上变化略大。

## 判定与下一条假设

原 Pilot 仍是 `g3_direction_observed=false`、`formal_unlocked=false`，当前 recipe
停止。下一条只作为待预注册假设：先处理 **condition-aware / low-noise future
fusion**，并用独立的 RayPose gate-zero 或 G1/G2 对照区分几何条件与共享正则化；
不能把本次 post-hoc 结果用于放宽旧阈值或直接启动 formal。
"""


def write_analysis(
    source_root: str | Path,
    output_root: str | Path,
) -> Path:
    source = Path(source_root).resolve()
    target = Path(output_root).resolve()
    if target.exists():
        raise ReadonlyFailureAnalysisError(f"refusing to overwrite analysis: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        result, tables = build_analysis(source)
        (temporary / "analysis_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (temporary / "report.md").write_text(render_report(result), encoding="utf-8")
        for filename, rows in tables.items():
            _write_csv(temporary / filename, rows)
        files = {
            path.name: file_sha256(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = seal_full_object(
            {
                "schema_version": "thought5.phase5.pilot_v4_readonly_failure_manifest.v1",
                "status": "complete",
                "analysis_result_full_object_sha256": result["full_object_sha256"],
                "files_sha256": files,
                "source_sha256": dict(EXPECTED_SOURCE_SHA256),
                "source_unchanged": True,
                "pilot_decision_preserved": True,
                "formal_unlocked": False,
            }
        )
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target
