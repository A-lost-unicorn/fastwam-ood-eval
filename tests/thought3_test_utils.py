from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_thought3_config(
    root: Path,
    *,
    variant: str = "A1",
    sample_count: int = 6,
    shard_size: int = 2,
    max_steps: int = 12,
    overrides: dict[str, Any] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    thought3_root = root / "thought3"
    data: dict[str, Any] = {
        "variant": variant,
        "experiment": {
            "name": f"mock_{variant.lower()}",
            "output_dir": str(thought3_root / "experiment"),
            "seed": 3407,
        },
        "runtime": {
            "backend": "mock",
            "device": "cpu",
        },
        "sampler": {
            "active_k": {
                "B0": 0,
                "A0": 0,
                "A1": 1,
                "A2": 2,
                "A4": 4,
                "A-shuffle": 4,
            }[variant],
        },
        "adapter": {
            "enabled": variant != "B0",
            "input_channels": 48,
            "action_hidden_dim": 32,
            "future_dim": 8,
            "attention_dim": 16,
            "num_heads": 4,
            "max_projected_grid": [2, 7, 14],
        },
        "data": {
            "dataset_revision": "mock-test-v1",
            "mock_sample_count": sample_count,
        },
        "cache": {
            "root": str(thought3_root / "cache"),
            "shard_size": shard_size,
            "required_free_space_fraction": 0.0,
        },
        "training": {
            "max_steps": max_steps,
            "microbatch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
    }

    def merge(target: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    if overrides:
        merge(data, overrides)
    path = root / "thought3_config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path
