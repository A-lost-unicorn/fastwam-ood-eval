"""Linear-probe geometry subspace and coordinate-only replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GeometrySubspaceError(ValueError):
    """Raised when a geometry basis/intervention is ill-defined."""


@dataclass(frozen=True)
class GeometrySubspace:
    basis: Any  # [hidden_dim, rank], columns are orthonormal
    rank: int
    explained_weight_energy: float
    singular_values: Any


@dataclass(frozen=True)
class SubspaceIntervention:
    output: Any
    residual: Any
    original_coordinates: Any
    replacement_coordinates: Any
    intervention_norm: float
    hidden_norm: float
    intervention_hidden_ratio: float
    residual_reconstruction_error: float
    norm_ratio: float


def subspace_from_linear_weight(
    weight: Any,
    *,
    energy_threshold: float = 0.95,
    max_rank: int = 32,
) -> GeometrySubspace:
    import torch

    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise GeometrySubspaceError("linear weight must be a [target,hidden] Tensor")
    value = weight.detach().float()
    if not bool(value.isfinite().all().item()):
        raise GeometrySubspaceError("linear weight contains NaN/Inf")
    if not 0 < energy_threshold <= 1 or max_rank <= 0:
        raise GeometrySubspaceError("invalid energy threshold/max_rank")
    _left, singular, right_h = torch.linalg.svd(value, full_matrices=False)
    energy = singular.square()
    total = float(energy.sum().item())
    if total <= 1e-20:
        raise GeometrySubspaceError("linear probe weight has zero geometry energy")
    cumulative = energy.cumsum(dim=0) / energy.sum()
    threshold_rank = int(
        torch.nonzero(cumulative >= energy_threshold, as_tuple=False)[0].item()
    ) + 1
    rank = min(threshold_rank, max_rank, right_h.shape[0])
    basis = right_h[:rank].T.contiguous()
    gram = basis.T @ basis
    if not torch.allclose(
        gram, torch.eye(rank, device=gram.device), atol=1e-5, rtol=1e-5
    ):
        raise GeometrySubspaceError("derived geometry basis is not orthonormal")
    explained = float(energy[:rank].sum().item() / total)
    return GeometrySubspace(
        basis=basis.detach(),
        rank=rank,
        explained_weight_energy=explained,
        singular_values=singular.detach(),
    )


def geometry_coordinates(hidden: Any, subspace: GeometrySubspace) -> Any:
    import torch

    if not isinstance(hidden, torch.Tensor) or hidden.shape[-1] != subspace.basis.shape[0]:
        raise GeometrySubspaceError("hidden dimension does not match geometry basis")
    if not bool(hidden.isfinite().all().item()):
        raise GeometrySubspaceError("hidden contains NaN/Inf")
    basis = subspace.basis.to(device=hidden.device, dtype=hidden.dtype)
    return hidden @ basis


def _match_coordinate_norm(source: Any, reference: Any, *, epsilon: float = 1e-8) -> Any:
    source_norm = source.norm(dim=-1, keepdim=True)
    reference_norm = reference.norm(dim=-1, keepdim=True)
    scale = reference_norm / source_norm.clamp_min(epsilon)
    matched = source * scale
    # If a donor coordinate is exactly zero, retain it rather than inventing direction.
    return matched.where(source_norm > epsilon, source)


def replace_geometry_coordinates(
    hidden: Any,
    replacement_coordinates: Any,
    subspace: GeometrySubspace,
    *,
    norm_match: bool,
) -> SubspaceIntervention:
    import torch

    original = geometry_coordinates(hidden, subspace)
    replacement = replacement_coordinates.to(
        device=hidden.device, dtype=hidden.dtype
    )
    if replacement.shape != original.shape:
        raise GeometrySubspaceError(
            f"coordinate shape mismatch: {replacement.shape} vs {original.shape}"
        )
    if not bool(replacement.isfinite().all().item()):
        raise GeometrySubspaceError("replacement coordinates contain NaN/Inf")
    if norm_match:
        replacement = _match_coordinate_norm(replacement, original)
    basis = subspace.basis.to(device=hidden.device, dtype=hidden.dtype)
    projected = original @ basis.T
    residual = hidden - projected
    output = residual + replacement @ basis.T
    reconstruction = residual + original @ basis.T
    error = float(
        (reconstruction.float() - hidden.float()).abs().max().detach().cpu()
    )
    delta_norm = float((output.float() - hidden.float()).norm().detach().cpu())
    hidden_norm = float(hidden.float().norm().detach().cpu())
    original_norm = float(original.float().norm().detach().cpu())
    replacement_norm = float(replacement.float().norm().detach().cpu())
    return SubspaceIntervention(
        output=output,
        residual=residual,
        original_coordinates=original,
        replacement_coordinates=replacement,
        intervention_norm=delta_norm,
        hidden_norm=hidden_norm,
        intervention_hidden_ratio=delta_norm / max(hidden_norm, 1e-12),
        residual_reconstruction_error=error,
        norm_ratio=replacement_norm / max(original_norm, 1e-12),
    )


def correct_reconstruction(
    hidden: Any, subspace: GeometrySubspace
) -> SubspaceIntervention:
    return replace_geometry_coordinates(
        hidden,
        geometry_coordinates(hidden, subspace),
        subspace,
        norm_match=False,
    )

