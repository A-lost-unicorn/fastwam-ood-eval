"""Deterministic wrong-future assignments and action intervention metrics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor

from fastwam_ood_eval.thought3.schemas import (
    FutureLatentRecord,
    sha256_canonical,
)


class CounterfactualError(RuntimeError):
    """Raised when a valid intervention cannot be constructed."""


@dataclass(frozen=True)
class ShuffleCandidate:
    base_sample_id: str
    episode_id: str
    task_id: str
    split: str
    k: int
    cache_fingerprint: str

    @classmethod
    def from_cache_metadata(
        cls,
        payload: Mapping[str, Any],
    ) -> "ShuffleCandidate":
        record = FutureLatentRecord.from_dict(payload["record"])
        return cls(
            base_sample_id=record.base_sample_id,
            episode_id=str(payload["episode_id"]),
            task_id=str(payload["task_id"]),
            split=record.split,
            k=record.k,
            cache_fingerprint=record.cache_fingerprint,
        )


@dataclass(frozen=True)
class ShufflePair:
    recipient_base_sample_id: str
    donor_base_sample_id: str
    recipient_episode_id: str
    donor_episode_id: str
    recipient_task_id: str
    donor_task_id: str
    split: str
    k: int
    cache_fingerprint: str


def _legal_donor(
    recipient: ShuffleCandidate,
    donor: ShuffleCandidate,
) -> bool:
    return (
        donor.base_sample_id != recipient.base_sample_id
        and donor.episode_id != recipient.episode_id
        and donor.task_id != recipient.task_id
        and donor.k == recipient.k
        and donor.split == recipient.split
        and donor.cache_fingerprint == recipient.cache_fingerprint
    )


def build_shuffle_pairs(
    candidates: Iterable[ShuffleCandidate],
    *,
    seed: int = 3407,
) -> tuple[ShufflePair, ...]:
    """Find a deterministic one-to-one cross-task derangement per cache group."""

    if seed < 0:
        raise CounterfactualError("shuffle seed must be non-negative")
    values = list(candidates)
    identities = {
        (
            candidate.base_sample_id,
            candidate.k,
            candidate.split,
            candidate.cache_fingerprint,
        )
        for candidate in values
    }
    if len(identities) != len(values):
        raise CounterfactualError(
            "shuffle candidates contain duplicate base/K identities"
        )
    groups: dict[tuple[str, int, str], list[ShuffleCandidate]] = {}
    for candidate in values:
        key = (
            candidate.split,
            candidate.k,
            candidate.cache_fingerprint,
        )
        groups.setdefault(key, []).append(candidate)

    pairs: list[ShufflePair] = []
    for key, group in sorted(groups.items()):
        ordered_recipients = sorted(
            group,
            key=lambda value: hashlib.sha256(
                f"thought3-shuffle-recipient-v1\0{seed}\0"
                f"{value.base_sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        candidates_by_recipient = {
            recipient.base_sample_id: sorted(
                [
                    donor
                    for donor in group
                    if _legal_donor(recipient, donor)
                ],
                key=lambda donor: hashlib.sha256(
                    f"thought3-shuffle-donor-v1\0{seed}\0"
                    f"{recipient.base_sample_id}\0"
                    f"{donor.base_sample_id}".encode("utf-8")
                ).hexdigest(),
            )
            for recipient in ordered_recipients
        }
        impossible = [
            base_id
            for base_id, donors in candidates_by_recipient.items()
            if not donors
        ]
        if impossible:
            raise CounterfactualError(
                "no legal cross-task donor for recipients: "
                f"{impossible[:5]} in group={key}"
            )

        donor_to_recipient: dict[str, str] = {}
        recipient_to_donor: dict[str, ShuffleCandidate] = {}

        def augment(
            recipient: ShuffleCandidate,
            visited_donors: set[str],
        ) -> bool:
            for donor in candidates_by_recipient[recipient.base_sample_id]:
                if donor.base_sample_id in visited_donors:
                    continue
                visited_donors.add(donor.base_sample_id)
                prior_recipient_id = donor_to_recipient.get(donor.base_sample_id)
                if prior_recipient_id is None:
                    donor_to_recipient[donor.base_sample_id] = (
                        recipient.base_sample_id
                    )
                    recipient_to_donor[recipient.base_sample_id] = donor
                    return True
                prior_recipient = next(
                    value
                    for value in ordered_recipients
                    if value.base_sample_id == prior_recipient_id
                )
                if augment(prior_recipient, visited_donors):
                    donor_to_recipient[donor.base_sample_id] = (
                        recipient.base_sample_id
                    )
                    recipient_to_donor[recipient.base_sample_id] = donor
                    return True
            return False

        for recipient in ordered_recipients:
            if not augment(recipient, set()):
                raise CounterfactualError(
                    "cannot construct one-to-one cross-task derangement "
                    f"for group={key}"
                )
        for recipient in group:
            donor = recipient_to_donor[recipient.base_sample_id]
            pairs.append(
                ShufflePair(
                    recipient_base_sample_id=recipient.base_sample_id,
                    donor_base_sample_id=donor.base_sample_id,
                    recipient_episode_id=recipient.episode_id,
                    donor_episode_id=donor.episode_id,
                    recipient_task_id=recipient.task_id,
                    donor_task_id=donor.task_id,
                    split=recipient.split,
                    k=recipient.k,
                    cache_fingerprint=recipient.cache_fingerprint,
                )
            )
    validate_shuffle_pairs(pairs)
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                pair.k,
                pair.split,
                pair.recipient_base_sample_id,
            ),
        )
    )


def validate_shuffle_pairs(pairs: Iterable[ShufflePair]) -> None:
    by_group: dict[tuple[str, int, str], list[ShufflePair]] = {}
    for pair in pairs:
        if pair.recipient_base_sample_id == pair.donor_base_sample_id:
            raise CounterfactualError("shuffle donor equals recipient")
        if pair.recipient_episode_id == pair.donor_episode_id:
            raise CounterfactualError("shuffle donor is from recipient episode")
        if pair.recipient_task_id == pair.donor_task_id:
            raise CounterfactualError("shuffle donor is from recipient task")
        key = (pair.split, pair.k, pair.cache_fingerprint)
        by_group.setdefault(key, []).append(pair)
    for key, group in by_group.items():
        recipients = [pair.recipient_base_sample_id for pair in group]
        donors = [pair.donor_base_sample_id for pair in group]
        if len(recipients) != len(set(recipients)):
            raise CounterfactualError(f"duplicate shuffle recipient in group={key}")
        if len(donors) != len(set(donors)):
            raise CounterfactualError(f"shuffle donors are not one-to-one in group={key}")
        if set(recipients) != set(donors):
            raise CounterfactualError(
                f"shuffle mapping is not a permutation in group={key}"
            )


def shuffle_manifest(
    pairs: Iterable[ShufflePair],
    *,
    seed: int,
) -> dict[str, Any]:
    rows = [asdict(pair) for pair in pairs]
    return {
        "schema_version": "thought3.shuffle_manifest.v1",
        "seed": seed,
        "pairs": rows,
        "fingerprint": sha256_canonical(
            {
                "schema_version": "thought3.shuffle_manifest.v1",
                "seed": seed,
                "pairs": rows,
            }
        ),
    }


def action_sha256(action: Tensor) -> str:
    value = action.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def compare_action_chunks(
    reference: Tensor,
    intervention: Tensor,
) -> dict[str, float | str]:
    if reference.shape != intervention.shape or reference.ndim not in {2, 3}:
        raise CounterfactualError(
            "action chunks must share [S,A] or [B,S,A] shape"
        )
    first = reference.detach().float()
    second = intervention.detach().float()
    delta = second - first
    flattened_first = (
        first.reshape(first.shape[0], -1)
        if first.ndim == 3
        else first.reshape(1, -1)
    )
    flattened_second = (
        second.reshape(second.shape[0], -1)
        if second.ndim == 3
        else second.reshape(1, -1)
    )
    cosine = torch.nn.functional.cosine_similarity(
        flattened_first,
        flattened_second,
        dim=-1,
        eps=1e-8,
    ).mean()
    xyz_dim = min(3, first.shape[-1])
    trajectory_first = first[..., :xyz_dim].cumsum(dim=-2)
    trajectory_second = second[..., :xyz_dim].cumsum(dim=-2)
    trajectory_change = (
        trajectory_second - trajectory_first
    ).square().sum(dim=-1).sqrt().mean()
    return {
        "action_direction_cosine": float(cosine.cpu()),
        "action_hash": action_sha256(intervention),
        "action_l1": float(delta.abs().mean().cpu()),
        "action_l2": float(delta.square().mean().sqrt().cpu()),
        "end_effector_trajectory_l2": float(trajectory_change.cpu()),
        "gripper_action_change": float(delta[..., -1].abs().mean().cpu()),
        "reference_action_hash": action_sha256(reference),
    }


ActionFunction = Callable[[Tensor, int], Tensor]


def run_action_counterfactuals(
    action_function: ActionFunction,
    *,
    correct_future: Tensor,
    shuffled_future: Tensor,
    action_seed: int,
    different_k_futures: Mapping[int, Tensor] | None = None,
) -> dict[str, Any]:
    """Hold action noise seed fixed while replacing only the future tensor."""

    if correct_future.shape != shuffled_future.shape:
        raise CounterfactualError("correct and shuffled future shapes differ")
    if action_seed < 0:
        raise CounterfactualError("action seed must be non-negative")
    reference = action_function(correct_future, action_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(action_seed ^ 0x5A17)
    random_future = torch.randn(
        correct_future.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=correct_future.device, dtype=correct_future.dtype)
    interventions: dict[str, Tensor] = {
        "null": torch.zeros_like(correct_future),
        "shuffle": shuffled_future,
        "random": random_future,
    }
    for k, future in sorted((different_k_futures or {}).items()):
        if future.shape != correct_future.shape:
            raise CounterfactualError(f"K={k} future shape differs from reference")
        interventions[f"k{k}"] = future
    metrics = {
        name: compare_action_chunks(
            reference,
            action_function(future, action_seed),
        )
        for name, future in interventions.items()
    }
    return {
        "action_seed": action_seed,
        "correct_action_hash": action_sha256(reference),
        "interventions": metrics,
    }
