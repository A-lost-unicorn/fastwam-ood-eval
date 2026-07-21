"""Optional in-memory video recorder with failure-only retention."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VideoRecorder:
    def __init__(self, enabled: bool, fps: int, output_path: Path) -> None:
        self.enabled = enabled
        self.fps = fps
        self.output_path = output_path
        self.frames: list[Any] = []

    def add_observation(self, observation: dict[str, Any]) -> None:
        if not self.enabled:
            return
        frame = observation.get("agentview_image")
        if frame is not None:
            try:
                self.frames.append(frame.copy())
            except AttributeError:
                self.frames.append(frame)

    def save(self) -> str | None:
        if not self.enabled or not self.frames:
            return None
        try:
            import imageio.v2 as imageio
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Video recording requires imageio and numpy from the Fast-WAM environment") from exc
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(self.output_path), fps=self.fps) as writer:
            for frame in self.frames:
                array = np.asarray(frame)
                # Match the official evaluator's 180-degree image orientation.
                writer.append_data(np.ascontiguousarray(array[::-1, ::-1]))
        return str(self.output_path)

    def discard(self) -> None:
        self.frames.clear()

