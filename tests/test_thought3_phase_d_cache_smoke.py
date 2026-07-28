from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F

from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.phase_d_cache_smoke import (
    PHASE_D_TASK_NAME,
    _assert_phase_d_scope,
    _inventory_rows,
)
from fastwam_ood_eval.thought3.real_cache_builder import (
    CurrentOnlyLiberoSource,
    RealCacheBuildError,
    build_real_cache,
    preprocess_current_camera_frames,
    preprocess_current_proprio,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_dataset(root: Path) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": (
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": PHASE_D_TASK_NAME},
            {"task_index": 1, "task": "another task"},
        ],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": index,
                "length": 2,
                "tasks": [PHASE_D_TASK_NAME],
            }
            for index in range(3)
        ]
        + [
            {
                "episode_index": 3,
                "length": 2,
                "tasks": ["another task"],
            }
        ],
    )
    for episode_index in range(4):
        task_index = 0 if episode_index < 3 else 1
        table = pa.table(
            {
                "timestamp": [0.0, 0.05],
                "frame_index": [0, 1],
                "episode_index": [episode_index, episode_index],
                "task_index": [task_index, task_index],
            }
        )
        path = (
            root
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def test_phase_d_inventory_selects_one_current_frame_per_task_episode(
    tmp_path,
) -> None:
    root = tmp_path / "libero_goal"
    _fake_dataset(root)
    rows, manifest = _inventory_rows(
        root,
        camera_keys=("image", "wrist_image"),
    )
    assert len(rows) == 3
    assert [row.episode_index for row in rows] == [0, 1, 2]
    assert all(row.frame_index == 0 for row in rows)
    assert all(row.task_id == "task_0" for row in rows)
    assert manifest["selection"] == "first_frame_per_episode"
    assert manifest["future_rgb_requested"] is False
    assert len(manifest["parquet_sha256"]) == 3


class _ResizeToFloat:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value.float() / 255.0,
            size=(224, 224),
            mode="nearest",
        )


class _Identity:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_current_only_camera_preprocessing_concatenates_exactly_two_frames() -> None:
    processor = SimpleNamespace(
        shape_meta={
            "images": [
                {"key": "image", "shape": [3, 224, 224]},
                {"key": "wrist_image", "shape": [3, 224, 224]},
            ]
        },
        val_transforms=[_ResizeToFloat()],
    )
    dataset = SimpleNamespace(
        resize_transform=_Identity(),
        crop_transform=_Identity(),
        normalize_transform=lambda value: value * 2.0 - 1.0,
    )
    frames = {
        "observation.images.image": torch.zeros((3, 512, 512)),
        "observation.images.wrist_image": torch.ones((3, 512, 512)),
    }
    image = preprocess_current_camera_frames(
        frames,
        processor=processor,
        robot_video_dataset=dataset,
    )
    assert image.shape == (1, 3, 224, 448)
    assert torch.equal(image[..., :224], torch.full_like(image[..., :224], -1))
    assert torch.equal(image[..., 224:], torch.ones_like(image[..., 224:]))


def test_current_proprio_uses_official_transform_normalizer_and_merger() -> None:
    class _Normalizer:
        def forward(self, batch):
            batch["state"]["default"] += 2
            return batch

    class _Merger:
        def forward(self, batch):
            batch["state"] = batch["state"]["default"]
            return batch

    processor = SimpleNamespace(
        shape_meta={"state": [{"key": "default"}]},
        action_state_transform=lambda batch: batch,
        normalizer=_Normalizer(),
        action_state_merger=_Merger(),
    )
    output = preprocess_current_proprio(
        torch.arange(8, dtype=torch.float32),
        processor=processor,
    )
    assert output.shape == (1, 8)
    assert torch.equal(output, torch.arange(8).unsqueeze(0) + 2)


def test_current_only_source_api_exposes_no_future_or_action_target() -> None:
    parameters = set(inspect.signature(CurrentOnlyLiberoSource.load).parameters)
    assert parameters == {"self", "entry"}
    forbidden = {
        "action",
        "target_action",
        "future_frames",
        "next_observation",
        "success",
    }
    assert not parameters & forbidden


def test_phase_d_config_scope_is_frozen() -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_d_cache_smoke.yaml"
    )
    _assert_phase_d_scope(cfg)
    assert cfg.cache.pilot_limit == 32
    assert cfg.cache.shard_size == 8


def test_real_cache_refuses_without_explicit_confirmation(monkeypatch) -> None:
    cfg = load_thought3_config(
        "configs/thought3/phase_d_cache_smoke.yaml"
    )
    monkeypatch.delenv("CONFIRM_THOUGHT3_PHASE_D", raising=False)
    with pytest.raises(
        RealCacheBuildError,
        match="CONFIRM_THOUGHT3_PHASE_D",
    ):
        build_real_cache(cfg, resume=False)
