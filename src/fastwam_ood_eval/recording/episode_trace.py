"""Optional compact JSONL episode trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class EpisodeTrace:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "EpisodeTrace":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")
        return self

    def record(self, step: int, *, action: Any = None, robot_state: Any = None) -> None:
        if self._handle is None:
            return
        self._handle.write(
            json.dumps(
                {"step": step, "action": _serializable(action), "robot_state": _serializable(robot_state)},
                ensure_ascii=False,
            )
            + "\n"
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()

