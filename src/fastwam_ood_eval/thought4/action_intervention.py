"""Matched action counterfactuals for a probe-defined geometry subspace."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from fastwam_ood_eval.thought4.geometry_subspace import (
    GeometrySubspace,
    geometry_coordinates,
    replace_geometry_coordinates,
)
from fastwam_ood_eval.thought4.schemas import sha256_canonical


class ActionInterventionError(RuntimeError):
    """Raised when a matched donor/action comparison cannot be constructed."""


@dataclass(frozen=True)
class DonorCandidate:
    sample_id: str
    task_id: str
    episode_id: str
    progress_bin: int


@dataclass(frozen=True)
class DonorPair:
    target_sample_id: str
    donor_sample_id: str
    task_id: str
    target_episode_id: str
    donor_episode_id: str
    target_progress_bin: int
    donor_progress_bin: int


@dataclass(frozen=True)
class ActionSeedIdentity:
    seed: int
    initial_noise_sha256: str
    denoise_schedule_sha256: str
    checkpoint_sha256: str
    preprocessing_sha256: str


def build_deterministic_derangement(
    candidates: Iterable[DonorCandidate],
    *,
    seed: int,
) -> tuple[DonorPair, ...]:
    """Build a one-to-one same-task, cross-episode donor permutation."""

    if seed < 0:
        raise ActionInterventionError("donor seed must be non-negative")
    values = list(candidates)
    if len({value.sample_id for value in values}) != len(values):
        raise ActionInterventionError("donor candidates contain duplicate sample IDs")
    groups: dict[str, list[DonorCandidate]] = {}
    for value in values:
        groups.setdefault(value.task_id, []).append(value)
    result: list[DonorPair] = []
    for task_id, group in sorted(groups.items()):
        if len({value.episode_id for value in group}) < 2:
            raise ActionInterventionError(
                f"task {task_id} needs candidates from at least two episodes"
            )
        recipients = sorted(
            group,
            key=lambda value: hashlib.sha256(
                f"thought4-recipient-v1\0{seed}\0{value.sample_id}".encode()
            ).hexdigest(),
        )
        preferences: dict[str, list[DonorCandidate]] = {}
        for recipient in recipients:
            legal = [
                donor
                for donor in group
                if donor.sample_id != recipient.sample_id
                and donor.episode_id != recipient.episode_id
            ]
            preferences[recipient.sample_id] = sorted(
                legal,
                key=lambda donor: (
                    abs(donor.progress_bin - recipient.progress_bin),
                    hashlib.sha256(
                        f"thought4-donor-v1\0{seed}\0{recipient.sample_id}\0"
                        f"{donor.sample_id}".encode()
                    ).hexdigest(),
                ),
            )
        donor_owner: dict[str, str] = {}
        assignment: dict[str, DonorCandidate] = {}
        recipient_by_id = {value.sample_id: value for value in recipients}

        def augment(recipient: DonorCandidate, visited: set[str]) -> bool:
            for donor in preferences[recipient.sample_id]:
                if donor.sample_id in visited:
                    continue
                visited.add(donor.sample_id)
                prior_id = donor_owner.get(donor.sample_id)
                if prior_id is None or augment(recipient_by_id[prior_id], visited):
                    donor_owner[donor.sample_id] = recipient.sample_id
                    assignment[recipient.sample_id] = donor
                    return True
            return False

        for recipient in recipients:
            if not augment(recipient, set()):
                raise ActionInterventionError(
                    f"cannot build one-to-one cross-episode derangement for {task_id}"
                )
        for target in sorted(group, key=lambda value: value.sample_id):
            donor = assignment[target.sample_id]
            result.append(
                DonorPair(
                    target_sample_id=target.sample_id,
                    donor_sample_id=donor.sample_id,
                    task_id=task_id,
                    target_episode_id=target.episode_id,
                    donor_episode_id=donor.episode_id,
                    target_progress_bin=target.progress_bin,
                    donor_progress_bin=donor.progress_bin,
                )
            )
    validate_derangement(result)
    return tuple(result)


def validate_derangement(pairs: Sequence[DonorPair]) -> None:
    if not pairs:
        raise ActionInterventionError("donor mapping must not be empty")
    groups: dict[str, list[DonorPair]] = {}
    for pair in pairs:
        if pair.target_sample_id == pair.donor_sample_id:
            raise ActionInterventionError("donor derangement has a fixed point")
        if pair.target_episode_id == pair.donor_episode_id:
            raise ActionInterventionError("donor must be from a different episode")
        groups.setdefault(pair.task_id, []).append(pair)
    for task_id, group in groups.items():
        recipients = {pair.target_sample_id for pair in group}
        donors = {pair.donor_sample_id for pair in group}
        if len(recipients) != len(group) or len(donors) != len(group):
            raise ActionInterventionError(
                f"donor mapping is not one-to-one for task {task_id}"
            )
        if recipients != donors:
            raise ActionInterventionError(
                f"donor mapping is not a permutation for task {task_id}"
            )


def donor_manifest(pairs: Sequence[DonorPair], *, seed: int) -> dict[str, Any]:
    rows = [asdict(value) for value in pairs]
    payload = {
        "schema_version": "thought4.phase4.donor_derangement.v1",
        "seed": int(seed),
        "pairs": rows,
    }
    payload["mapping_sha256"] = sha256_canonical(payload)
    return payload


def validate_seed_identity(
    correct: ActionSeedIdentity, shuffled: ActionSeedIdentity
) -> None:
    if correct != shuffled:
        raise ActionInterventionError(
            "correct/shuffle action seed, noise, schedule or preprocessing differs"
        )


def compare_action_chunks(reference: Any, intervention: Any) -> dict[str, Any]:
    """Thought3-compatible metrics plus axis/timestep-resolved differences."""

    import torch

    from fastwam_ood_eval.thought3.counterfactuals import (
        compare_action_chunks as thought3_compare,
    )

    base = thought3_compare(reference, intervention)
    first = reference.detach().float()
    second = intervention.detach().float()
    if first.shape != second.shape or first.ndim not in {2, 3}:
        raise ActionInterventionError("action chunks must share [S,A] or [B,S,A]")
    delta = second - first
    step_dim = -2
    translation_dims = min(3, delta.shape[-1])
    rotation_end = min(6, delta.shape[-1])
    base.update(
        {
            "translation_difference": float(
                delta[..., :translation_dims].square().mean().sqrt().cpu()
            ),
            "rotation_difference": float(
                delta[..., 3:rotation_end].square().mean().sqrt().cpu()
            )
            if rotation_end > 3
            else 0.0,
            "gripper_difference": float(delta[..., -1].abs().mean().cpu()),
            "timestep_l2": [
                float(value)
                for value in delta.square()
                .mean(dim=-1)
                .sqrt()
                .mean(dim=0 if delta.ndim == 3 else ())
                .cpu()
                .reshape(-1)
            ],
        }
    )
    return base


def replay_floor(
    action_function: Callable[[int], Any],
    *,
    seeds: Sequence[int],
    repeats: int,
) -> dict[str, Any]:
    if not seeds or repeats < 2:
        raise ActionInterventionError("replay floor needs seeds and >=2 repeats")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        reference = action_function(int(seed))
        for repeat in range(1, repeats):
            replay = action_function(int(seed))
            metrics = compare_action_chunks(reference, replay)
            rows.append({"seed": int(seed), "repeat": repeat, **metrics})
    return {
        "rows": rows,
        "max_action_l2": max(float(row["action_l2"]) for row in rows),
        "max_action_l1": max(float(row["action_l1"]) for row in rows),
    }


def geometry_shuffle_hidden(
    target_hidden: Any,
    donor_hidden: Any,
    subspace: GeometrySubspace,
) -> tuple[Any, dict[str, float]]:
    import torch

    if target_hidden.shape != donor_hidden.shape:
        raise ActionInterventionError("target/donor hidden shapes differ")
    donor_coordinates = geometry_coordinates(donor_hidden, subspace)
    target_coordinates = geometry_coordinates(target_hidden, subspace)
    basis = subspace.basis.to(
        device=target_hidden.device, dtype=torch.float32
    )
    projected = target_coordinates @ basis.T
    projected_energy = float(projected.float().square().sum().detach().cpu())
    hidden_energy = float(target_hidden.float().square().sum().detach().cpu())
    result = replace_geometry_coordinates(
        target_hidden,
        donor_coordinates,
        subspace,
        norm_match=True,
    )
    return result.output, {
        "geometry_subspace_rank": subspace.rank,
        "explained_weight_energy": subspace.explained_weight_energy,
        "explained_feature_energy": projected_energy / max(hidden_energy, 1e-20),
        "intervention_norm": result.intervention_norm,
        "hidden_norm": result.hidden_norm,
        "intervention_hidden_ratio": result.intervention_hidden_ratio,
        "residual_reconstruction_error": result.residual_reconstruction_error,
        "coordinate_norm_ratio": result.norm_ratio,
    }
