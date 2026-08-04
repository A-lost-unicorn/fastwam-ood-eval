"""Phase 5: camera-equivariant geometry alignment for Fast-WAM.

This namespace is intentionally independent from the frozen Thought1--4
artifacts.  Importing it never loads Fast-WAM, MuJoCo, or CUDA.
"""

from __future__ import annotations

SCHEMA_PREFIX = "thought5.phase5"
METHOD_NAME = "Fast-WAM-GeoEq"
VARIANTS = ("B0", "B1", "G1", "G2", "G3", "G4")
FORMAL_VARIANTS = ("B0", "B1", "G1", "G2", "G3")
MECHANISM_CLASSIFICATIONS = (
    "full_mechanism_support",
    "representation_only_support",
    "utility_without_closed_loop_support",
    "closed_loop_without_future_mediation",
    "mechanism_not_supported",
)

__all__ = [
    "FORMAL_VARIANTS",
    "MECHANISM_CLASSIFICATIONS",
    "METHOD_NAME",
    "SCHEMA_PREFIX",
    "VARIANTS",
]
