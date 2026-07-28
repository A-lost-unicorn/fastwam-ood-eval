"""Small, zero-gated future-to-action cross-attention Adapter.

This module deliberately has no dependency on Fast-WAM internals.  The only
contract is the audited native future tensor ``[B,C,T,H,W]`` and an action
hidden-state tensor ``[B,S,D]``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FutureAdapterError(ValueError):
    """Raised when future/action tensors violate the frozen interface."""


@dataclass(frozen=True)
class FutureAdapterSpec:
    input_channels: int = 48
    action_hidden_dim: int = 1024
    future_dim: int = 256
    attention_dim: int = 512
    num_heads: int = 8
    max_projected_grid: tuple[int, int, int] = (2, 7, 14)
    zero_init_gate: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.input_channels,
            self.action_hidden_dim,
            self.future_dim,
            self.attention_dim,
            self.num_heads,
            *self.max_projected_grid,
        )
        if min(dimensions) <= 0:
            raise FutureAdapterError("all Adapter dimensions must be positive")
        if self.attention_dim % self.num_heads:
            raise FutureAdapterError(
                "attention_dim must be divisible by num_heads"
            )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FutureAdapterDiagnostics:
    gate_raw: float
    gate_scale: float
    action_hidden_norm: float
    future_token_norm: float
    attention_residual_norm: float
    gated_delta_norm: float
    gated_delta_nonzero_fraction: float
    valid_token_fraction: float
    projected_grid: tuple[int, int, int]


class FutureToActionAdapter(nn.Module):
    """Project native future latents and condition action hidden states once.

    The projection is a ``[1,2,2]`` Conv3d, matching the audited Video DiT
    spatial patching while preserving latent time.  Factorized learned
    positional embeddings permit any projected token grid up to the configured
    maximum.  A scalar zero-initialized gate makes construction an exact
    identity at initialization.
    """

    def __init__(self, spec: FutureAdapterSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or FutureAdapterSpec()
        spec = self.spec
        self.future_projector = nn.Conv3d(
            spec.input_channels,
            spec.future_dim,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=True,
        )
        self.future_norm = nn.LayerNorm(spec.future_dim)
        max_t, max_h, max_w = spec.max_projected_grid
        self.time_position = nn.Parameter(torch.zeros(max_t, spec.future_dim))
        self.height_position = nn.Parameter(torch.zeros(max_h, spec.future_dim))
        self.width_position = nn.Parameter(torch.zeros(max_w, spec.future_dim))
        self.query_norm = nn.LayerNorm(spec.action_hidden_dim)
        self.query_projection = nn.Linear(
            spec.action_hidden_dim, spec.attention_dim
        )
        self.key_projection = nn.Linear(spec.future_dim, spec.attention_dim)
        self.value_projection = nn.Linear(spec.future_dim, spec.attention_dim)
        self.output_projection = nn.Linear(
            spec.attention_dim, spec.action_hidden_dim
        )
        gate_value = 0.0 if spec.zero_init_gate else 1.0
        self.gate = nn.Parameter(torch.tensor(gate_value, dtype=torch.float32))
        self._last_diagnostics: FutureAdapterDiagnostics | None = None
        self.capture_diagnostics = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.future_projector.weight, a=math.sqrt(5))
        if self.future_projector.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.future_projector.weight
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.future_projector.bias, -bound, bound)
        nn.init.normal_(self.time_position, std=0.02)
        nn.init.normal_(self.height_position, std=0.02)
        nn.init.normal_(self.width_position, std=0.02)
        self.future_norm.reset_parameters()
        self.query_norm.reset_parameters()
        for layer in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            layer.reset_parameters()
        with torch.no_grad():
            self.gate.fill_(0.0 if self.spec.zero_init_gate else 1.0)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def last_diagnostics(self) -> FutureAdapterDiagnostics | None:
        return self._last_diagnostics

    def _validate_inputs(
        self,
        action_hidden: Tensor,
        future_latent: Tensor,
    ) -> None:
        if action_hidden.ndim != 3:
            raise FutureAdapterError(
                "action_hidden must have shape [B,S,D], "
                f"got {tuple(action_hidden.shape)}"
            )
        if action_hidden.shape[-1] != self.spec.action_hidden_dim:
            raise FutureAdapterError(
                "action hidden dimension mismatch: expected "
                f"{self.spec.action_hidden_dim}, got {action_hidden.shape[-1]}"
            )
        if future_latent.ndim != 5:
            raise FutureAdapterError(
                "future_latent must have shape [B,C,T,H,W], "
                f"got {tuple(future_latent.shape)}"
            )
        if future_latent.shape[0] != action_hidden.shape[0]:
            raise FutureAdapterError("action and future batch sizes differ")
        if future_latent.shape[1] != self.spec.input_channels:
            raise FutureAdapterError(
                "future channel mismatch: expected "
                f"{self.spec.input_channels}, got {future_latent.shape[1]}"
            )
        if future_latent.shape[2] <= 0:
            raise FutureAdapterError("future latent must contain time tokens")
        if future_latent.shape[3] < 2 or future_latent.shape[4] < 2:
            raise FutureAdapterError(
                "future latent height/width must be at least 2 for [1,2,2] projection"
            )
        if not action_hidden.is_floating_point() or not future_latent.is_floating_point():
            raise FutureAdapterError("action_hidden and future_latent must be floating tensors")
        if action_hidden.device != future_latent.device:
            raise FutureAdapterError("action_hidden and future_latent must share a device")

    def _project_mask(
        self,
        future_mask: Tensor | None,
        *,
        batch_size: int,
        latent_grid: tuple[int, int, int],
        projected_grid: tuple[int, int, int],
        device: torch.device,
    ) -> Tensor:
        token_count = math.prod(projected_grid)
        if future_mask is None:
            return torch.ones(
                batch_size,
                token_count,
                dtype=torch.bool,
                device=device,
            )
        if future_mask.device != device:
            raise FutureAdapterError("future_mask must share the input device")
        if future_mask.ndim == 2:
            if tuple(future_mask.shape) != (batch_size, token_count):
                raise FutureAdapterError(
                    "projected future_mask must have shape "
                    f"[{batch_size},{token_count}], got {tuple(future_mask.shape)}"
                )
            projected = future_mask.to(dtype=torch.bool)
        elif future_mask.ndim in {4, 5}:
            latent_mask = future_mask
            if latent_mask.ndim == 5:
                if latent_mask.shape[1] != 1:
                    raise FutureAdapterError(
                        "5D latent future_mask must have singleton channel"
                    )
                latent_mask = latent_mask[:, 0]
            if tuple(latent_mask.shape) != (batch_size, *latent_grid):
                raise FutureAdapterError(
                    "latent future_mask must have shape "
                    f"[{batch_size},{','.join(map(str, latent_grid))}], "
                    f"got {tuple(latent_mask.shape)}"
                )
            pooled = F.max_pool3d(
                latent_mask.to(dtype=torch.float32).unsqueeze(1),
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
            )
            projected = pooled[:, 0].reshape(batch_size, -1) > 0
        else:
            raise FutureAdapterError(
                "future_mask must be [B,N], [B,T,H,W], or [B,1,T,H,W]"
            )
        if (~projected).all(dim=1).any():
            raise FutureAdapterError(
                "every sample must retain at least one valid future token"
            )
        return projected

    def _positions(
        self,
        grid: tuple[int, int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        time, height, width = grid
        max_time, max_height, max_width = self.spec.max_projected_grid
        if time > max_time or height > max_height or width > max_width:
            raise FutureAdapterError(
                f"projected grid {grid} exceeds configured maximum "
                f"{self.spec.max_projected_grid}"
            )
        positions = (
            self.time_position[:time, None, None, :]
            + self.height_position[None, :height, None, :]
            + self.width_position[None, None, :width, :]
        )
        return positions.reshape(1, time * height * width, -1).to(
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        action_hidden: Tensor,
        future_latent: Tensor,
        future_mask: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, FutureAdapterDiagnostics]:
        self._validate_inputs(action_hidden, future_latent)
        batch_size, sequence_length, _ = action_hidden.shape
        latent_grid = tuple(int(value) for value in future_latent.shape[2:])
        projected = self.future_projector(
            future_latent.to(dtype=self.future_projector.weight.dtype)
        )
        projected_grid = tuple(int(value) for value in projected.shape[2:])
        tokens = projected.flatten(2).transpose(1, 2)
        tokens = self.future_norm(tokens)
        tokens = tokens + self._positions(
            projected_grid,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        valid = self._project_mask(
            future_mask,
            batch_size=batch_size,
            latent_grid=latent_grid,
            projected_grid=projected_grid,
            device=tokens.device,
        )

        # The Adapter remains fp32 while Fast-WAM emits bf16 hidden states.
        # CUDA layer_norm requires its activation and affine parameters to use
        # compatible dtypes, so cast before normalization rather than after it.
        query_input = action_hidden.to(dtype=self.query_norm.weight.dtype)
        query = self.query_projection(self.query_norm(query_input))
        key = self.key_projection(tokens)
        value = self.value_projection(tokens)
        heads = self.spec.num_heads
        head_dim = self.spec.attention_dim // heads

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(batch_size, -1, heads, head_dim).transpose(1, 2)

        query_heads = split_heads(query)
        key_heads = split_heads(key)
        value_heads = split_heads(value)
        scores = torch.matmul(
            query_heads.float(), key_heads.float().transpose(-2, -1)
        ) / math.sqrt(head_dim)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1).to(value_heads.dtype)
        attended = torch.matmul(weights, value_heads)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.spec.attention_dim
        )
        residual = self.output_projection(attended).to(action_hidden.dtype)
        gate_scale = torch.tanh(self.gate).to(action_hidden.dtype)
        output = action_hidden + gate_scale * residual
        gated_delta = output - action_hidden

        diagnostics: FutureAdapterDiagnostics | None = None
        if return_diagnostics or self.capture_diagnostics:
            diagnostics = FutureAdapterDiagnostics(
                gate_raw=float(self.gate.detach().float().cpu()),
                gate_scale=float(torch.tanh(self.gate.detach()).float().cpu()),
                action_hidden_norm=float(
                    action_hidden.detach()
                    .float()
                    .norm(dim=-1)
                    .mean()
                    .cpu()
                ),
                future_token_norm=float(
                    tokens.detach().float().norm(dim=-1).mean().cpu()
                ),
                attention_residual_norm=float(
                    residual.detach().float().norm(dim=-1).mean().cpu()
                ),
                gated_delta_norm=float(
                    gated_delta.detach()
                    .float()
                    .norm(dim=-1)
                    .mean()
                    .cpu()
                ),
                gated_delta_nonzero_fraction=float(
                    (gated_delta.detach() != 0).float().mean().cpu()
                ),
                valid_token_fraction=float(valid.detach().float().mean().cpu()),
                projected_grid=projected_grid,
            )
            self._last_diagnostics = diagnostics
        if return_diagnostics:
            assert diagnostics is not None
            return output, diagnostics
        return output
