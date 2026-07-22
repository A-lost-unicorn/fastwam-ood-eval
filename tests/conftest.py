from __future__ import annotations

from pathlib import Path

import yaml


def write_config(
    root: Path,
    *,
    perturbation: bool = False,
    episodes: int = 4,
    output_name: str = "mock_eval",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    suite_path = root / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "suite": "libero_spatial",
                "tasks": [
                    {"id": 0, "name": "mock_task_zero"},
                    {"id": 1, "name": "mock_task_one"},
                ],
            }
        ),
        encoding="utf-8",
    )
    data = {
        "experiment": {
            "name": output_name,
            "output_dir": str(root / output_name),
            "seed": 17,
            "overwrite": False,
            "resume": True,
            "save_video": False,
            "save_failure_video_only": True,
            "log_level": "WARNING",
        },
        "hardware": {
            "devices": [0, 1, 2],
            "workers_per_gpu": 1,
            "precision": "fp32",
            "max_gpu_memory_gb": 23,
            "enable_tf32": False,
        },
        "checkpoint": {"path": None, "model_name": "mock", "config_path": None},
        "benchmark": {
            "backend": "mock",
            "suite": "libero_spatial",
            "suite_config": str(suite_path),
            "tasks": "all",
            "episodes_per_task": episodes,
            "max_steps": 6,
            "num_steps_wait": 0,
            "control_horizon": 2,
            "image_size": [32, 32],
        },
        "perturbation": {
            "enabled": perturbation,
            "category": ["camera_viewpoints", "light_conditions"] if perturbation else [],
            "level": ["easy", "hard"] if perturbation else [],
            "parameters": {},
        },
        "recording": {
            "fps": 10,
            "save_observations": False,
            "save_actions": True,
            "save_robot_state": True,
            "video_format": "mp4",
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
