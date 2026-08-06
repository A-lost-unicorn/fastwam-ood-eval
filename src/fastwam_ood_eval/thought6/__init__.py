"""Phase 6: Sigma-Aware Selective Future Fusion.

This namespace is deliberately independent from the stopped Thought5 GeoEq
recipe.  It reuses only the original frozen Fast-WAM backbone and the fixed
Thought3 K=1 Future-to-Action Adapter checkpoint.
"""

from __future__ import annotations

SIGMA_THRESHOLD = 0.5
ACTION_DENOISE_STEPS = 20
FUTURE_K = 1
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 6607

FUSION_MODES = (
    "B0",
    "F0",
    "Fsigma",
    "Label-Oracle",
    "Label-Oracle+Fsigma",
    "Shuffle+Fsigma",
)

MECHANISM_CLASSIFICATIONS = (
    "full_support",
    "utility_only_support",
    "performance_without_utility_mediation",
    "oracle_only_support",
    "not_supported",
)

__all__ = [
    "ACTION_DENOISE_STEPS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "FUSION_MODES",
    "FUTURE_K",
    "MECHANISM_CLASSIFICATIONS",
    "SIGMA_THRESHOLD",
]
