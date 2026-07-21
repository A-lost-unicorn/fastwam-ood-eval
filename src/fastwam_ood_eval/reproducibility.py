"""Deterministic seed assignment without importing heavyweight dependencies."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any


def stable_int(*parts: Any, bits: int = 31) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**bits)


def episode_seed(base_seed: int, suite: str, task_name: str, episode_index: int) -> int:
    """Return a condition-independent seed so clean and OOD results can be paired."""
    return stable_int(base_seed, suite, task_name, episode_index)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

