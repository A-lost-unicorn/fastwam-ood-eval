"""Frozen definitions of the six Phase 6 future-fusion modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fastwam_ood_eval.thought6 import FUSION_MODES
from fastwam_ood_eval.thought6.schemas import Thought6Error
from fastwam_ood_eval.thought6.sigma_gate import sigma_gate


class FusionMode(str, Enum):
    B0 = "B0"
    F0 = "F0"
    FSIGMA = "Fsigma"
    LABEL_ORACLE = "Label-Oracle"
    LABEL_ORACLE_FSIGMA = "Label-Oracle+Fsigma"
    SHUFFLE_FSIGMA = "Shuffle+Fsigma"


class FuturePayload(str, Enum):
    NULL = "null"
    CORRECT = "correct"
    SHUFFLE = "shuffle"


@dataclass(frozen=True)
class FusionDecision:
    mode: str
    condition: str
    effective_sigma: float
    external_gate: int
    adapter_called: bool
    payload: str
    label_oracle_used: bool
    reason: str


def decide_future_fusion(
    mode: FusionMode | str,
    *,
    condition: str,
    effective_sigma: float,
    payload_override: FuturePayload | str | None = None,
) -> FusionDecision:
    try:
        normalized_mode = FusionMode(mode)
    except ValueError as exc:
        raise Thought6Error(f"unknown fusion mode: {mode}") from exc
    if normalized_mode.value not in FUSION_MODES:
        raise Thought6Error("fusion mode is outside the frozen protocol")
    normalized_condition = condition.strip().lower().replace("-", "_")
    if normalized_condition not in {"clean", "camera"}:
        raise Thought6Error("Phase 6 condition must be Clean or Camera")
    high_sigma = bool(sigma_gate(effective_sigma))
    oracle = normalized_mode in {
        FusionMode.LABEL_ORACLE,
        FusionMode.LABEL_ORACLE_FSIGMA,
    }
    if normalized_mode is FusionMode.B0:
        enabled = False
        default_payload = FuturePayload.NULL
        reason = "no_future_baseline"
    elif normalized_mode is FusionMode.F0:
        enabled = True
        default_payload = FuturePayload.CORRECT
        reason = "full_stage"
    elif normalized_mode is FusionMode.FSIGMA:
        enabled = high_sigma
        default_payload = FuturePayload.CORRECT
        reason = "sigma_threshold"
    elif normalized_mode is FusionMode.LABEL_ORACLE:
        enabled = normalized_condition == "camera"
        default_payload = FuturePayload.CORRECT
        reason = "diagnostic_label_oracle"
    elif normalized_mode is FusionMode.LABEL_ORACLE_FSIGMA:
        enabled = normalized_condition == "camera" and high_sigma
        default_payload = FuturePayload.CORRECT
        reason = "diagnostic_label_oracle_and_sigma_threshold"
    else:
        enabled = high_sigma
        default_payload = FuturePayload.SHUFFLE
        reason = "sigma_threshold_shuffled_content"
    payload = default_payload if payload_override is None else FuturePayload(payload_override)
    if payload is FuturePayload.NULL:
        enabled = False
        reason = f"{reason}_formal_null"
    return FusionDecision(
        mode=normalized_mode.value,
        condition=normalized_condition,
        effective_sigma=float(effective_sigma),
        external_gate=int(enabled),
        adapter_called=bool(enabled),
        payload=(payload.value if enabled else FuturePayload.NULL.value),
        label_oracle_used=oracle,
        reason=reason,
    )
