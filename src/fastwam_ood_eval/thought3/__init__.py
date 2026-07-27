"""Isolated Partial-Future Adapter experiment subsystem.

The package intentionally exports only lightweight constants.  Torch, Hydra,
Fast-WAM and simulator modules are imported inside the commands that need
them, so Thought 1/2 commands and Thought 3 dry-runs do not load a model.
"""

from __future__ import annotations

THOUGHT3_CONFIG_SCHEMA = "thought3.config.v1"
THOUGHT3_CACHE_SCHEMA = "thought3.future_cache.v1"
THOUGHT3_CACHE_PLAN_SCHEMA = "thought3.cache_plan.v1"
THOUGHT3_CACHE_SHARD_SCHEMA = "thought3.cache_shard.v1"
THOUGHT3_CHECKPOINT_SCHEMA = "thought3.adapter_checkpoint.v1"
THOUGHT3_SPLIT_SCHEMA = "thought3.episode_split.v1"

__all__ = [
    "THOUGHT3_CACHE_PLAN_SCHEMA",
    "THOUGHT3_CACHE_SCHEMA",
    "THOUGHT3_CACHE_SHARD_SCHEMA",
    "THOUGHT3_CHECKPOINT_SCHEMA",
    "THOUGHT3_CONFIG_SCHEMA",
    "THOUGHT3_SPLIT_SCHEMA",
]
