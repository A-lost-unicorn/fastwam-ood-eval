"""Project-local LoRA injection for a frozen Fast-WAM Video DiT window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_LORA_TARGETS = (
    "video_expert.blocks.15.self_attn.k",
    "video_expert.blocks.15.self_attn.v",
)


class LoRATargetError(RuntimeError):
    pass


def _resolve_parent(root: Any, dotted: str) -> tuple[Any, str, Any]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            if not hasattr(parent, part):
                raise LoRATargetError(f"module path does not exist: {dotted}")
            parent = getattr(parent, part)
    name = parts[-1]
    original = parent[int(name)] if name.isdigit() else getattr(parent, name, None)
    if original is None:
        raise LoRATargetError(f"module path does not exist: {dotted}")
    return parent, name, original


def _assign(parent: Any, name: str, value: Any) -> None:
    if name.isdigit():
        parent[int(name)] = value
    else:
        setattr(parent, name, value)


def make_lora_linear(
    base: Any, *, rank: int, alpha: float, dropout: float, module_path: str
) -> Any:
    import torch

    if not isinstance(base, torch.nn.Linear):
        raise LoRATargetError(
            f"LoRA target must be torch.nn.Linear: {module_path} ({type(base)!r})"
        )
    if rank <= 0 or rank > min(base.in_features, base.out_features):
        raise LoRATargetError(f"invalid LoRA rank {rank} for {module_path}")

    class LoRALinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.base.requires_grad_(False)
            self.lora_A = torch.nn.Parameter(
                torch.empty(
                    rank,
                    base.in_features,
                    dtype=torch.float32,
                    device=base.weight.device,
                )
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(
                    base.out_features,
                    rank,
                    dtype=torch.float32,
                    device=base.weight.device,
                )
            )
            torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            self.scale = float(alpha) / rank
            self.dropout = torch.nn.Dropout(float(dropout))
            self.module_path = module_path

        @property
        def in_features(self) -> int:
            return self.base.in_features

        @property
        def out_features(self) -> int:
            return self.base.out_features

        def forward(self, value: Any) -> Any:
            base_output = self.base(value)
            update = torch.nn.functional.linear(
                torch.nn.functional.linear(self.dropout(value.float()), self.lora_A),
                self.lora_B,
            )
            return base_output + (update * self.scale).to(base_output.dtype)

    return LoRALinear()


@dataclass(frozen=True)
class InstalledLoRA:
    module_path: str
    rank: int
    alpha: float
    trainable_parameters: int


def install_lora_targets(
    model: Any,
    targets: Iterable[str] = DEFAULT_LORA_TARGETS,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> tuple[InstalledLoRA, ...]:
    installed: list[InstalledLoRA] = []
    seen: set[str] = set()
    for path in targets:
        if path in seen:
            raise LoRATargetError(f"duplicate LoRA target: {path}")
        seen.add(path)
        parent, name, original = _resolve_parent(model, path)
        wrapper = make_lora_linear(
            original,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            module_path=path,
        )
        _assign(parent, name, wrapper)
        installed.append(
            InstalledLoRA(
                module_path=path,
                rank=rank,
                alpha=float(alpha),
                trainable_parameters=rank
                * (original.in_features + original.out_features),
            )
        )
    return tuple(installed)


def adapter_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_A" in name or ".lora_B" in name
    }
