"""Static and runtime audits for the frozen Thought4 chain."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from fastwam_ood_eval.thought4.config import Thought4Config
from fastwam_ood_eval.thought4.feature_hooks import (
    action_hook_specs,
    resolve_module,
    validate_layer_indices,
    video_kv_cache_specs,
    video_hook_specs,
)
from fastwam_ood_eval.thought4.schemas import sha256_canonical, sha256_file


class Thought4AuditError(RuntimeError):
    """Raised when a frozen dependency or hook boundary is unavailable."""


def _git_head(path: Path) -> str:
    if not (path / ".git").exists():
        raise Thought4AuditError(f"upstream is not a git checkout: {path}")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_worktree_status(path: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    ).strip()


def require_project_clean(path: str | Path = ".") -> str:
    """Require a committed project snapshot before a real scientific run."""

    root = Path(path)
    commit = _git_head(root)
    status = _git_worktree_status(root)
    if status:
        preview = status.splitlines()[:8]
        raise Thought4AuditError(
            "real Thought4 execution requires a clean committed project "
            f"worktree; dirty entries={preview}"
        )
    return commit


def static_audit(cfg: Thought4Config) -> dict[str, Any]:
    """Read-only audit; deliberately does not import Torch or load a model."""

    required = {
        "checkpoint": cfg.backbone.checkpoint_path,
        "dataset_stats": cfg.backbone.dataset_stats_path,
        "dataset_info": cfg.cohort.dataset_root / "meta" / "info.json",
        "dataset_episodes": cfg.cohort.dataset_root / "meta" / "episodes.jsonl",
        "fastwam_config": Path("third_party/FastWAM/configs/model/fastwam.yaml"),
        "libero_plus_classification": Path(
            "third_party/LIBERO-plus/libero/libero/benchmark/"
            "task_classification.json"
        ),
        "audit_document": Path("docs/thought4/code_data_audit.md"),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise Thought4AuditError(f"required files are missing: {missing}")
    fastwam_commit = _git_head(Path("third_party/FastWAM"))
    if fastwam_commit != cfg.backbone.fastwam_commit:
        raise Thought4AuditError(
            f"Fast-WAM commit mismatch: {fastwam_commit}"
        )
    project_status = _git_worktree_status(Path("."))
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.static_audit.v1",
        "config_fingerprint": cfg.fingerprint,
        "would_load_torch": False,
        "would_load_model": False,
        "would_write": False,
        "fastwam_commit": fastwam_commit,
        "project_commit": _git_head(Path(".")),
        "project_worktree_clean": not bool(project_status),
        "project_dirty_entry_count": len(project_status.splitlines()),
        "libero_commit": _git_head(Path("third_party/LIBERO")),
        "libero_plus_commit": _git_head(Path("third_party/LIBERO-plus")),
        "required_files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                # Avoid re-hashing the 12 GB checkpoint during every dry-run.
                "sha256": (
                    cfg.backbone.checkpoint_sha256
                    if name == "checkpoint"
                    else sha256_file(path)
                ),
                "sha_source": (
                    "frozen_thought3_verified"
                    if name == "checkpoint"
                    else "computed"
                ),
            }
            for name, path in required.items()
        },
        "frozen_hooks": {
            "video": [
                {
                    "name": spec.name,
                    "module_path": spec.module_path,
                    "location": spec.location,
                }
                for spec in video_hook_specs(
                    cfg.backbone.video_layers, include_kv=False
                )
            ],
            "video_kv_cache_consumer": [
                {
                    "name": spec.name,
                    "module_path": spec.module_path,
                    "location": "forward_action_with_video_cache argument",
                }
                for spec in video_kv_cache_specs(cfg.backbone.video_layers)
            ],
            "action": [
                {
                    "name": spec.name,
                    "module_path": spec.module_path,
                    "location": spec.location,
                }
                for spec in action_hook_specs(cfg.backbone.action_hooks)
            ],
        },
        "scope": {
            "train_backbone": False,
            "read_future_rgb": False,
            "run_success_rollout": False,
            "train_probe_only": True,
            "one_geometry_subspace_intervention": True,
        },
        "status": "ready",
    }
    payload["audit_sha256"] = sha256_canonical(payload)
    return payload


def runtime_model_audit(model: Any, cfg: Thought4Config) -> dict[str, Any]:
    validate_layer_indices(
        model.video_expert, cfg.backbone.video_layers, "Video DiT"
    )
    validate_layer_indices(model.action_expert, (15, 29), "Action DiT")
    specs = (
        *video_hook_specs(cfg.backbone.video_layers, include_kv=False),
        *action_hook_specs(cfg.backbone.action_hooks),
    )
    resolved = []
    for spec in specs:
        module = resolve_module(model, spec.module_path)
        resolved.append(
            {
                "name": spec.name,
                "module_path": spec.module_path,
                "class": type(module).__name__,
                "location": spec.location,
            }
        )
    if len(model.video_expert.blocks) != 30 or len(model.action_expert.blocks) != 30:
        raise Thought4AuditError("official Video/Action DiT block count is not 30")
    if (
        int(model.video_expert.hidden_dim) != 3072
        or int(model.action_expert.hidden_dim) != 1024
    ):
        raise Thought4AuditError(
            "official Video/Action hidden dimensions are not 3072/1024"
        )
    if not hasattr(model, "mot") or not callable(
        getattr(model.mot, "forward_action_with_video_cache", None)
    ):
        raise Thought4AuditError(
            "actual Action-consumed Video K/V cache boundary is unavailable"
        )
    return {
        "schema_version": "thought4.phase4.runtime_model_audit.v1",
        "video_block_count": len(model.video_expert.blocks),
        "action_block_count": len(model.action_expert.blocks),
        "video_hidden_dim": int(model.video_expert.hidden_dim),
        "action_hidden_dim": int(model.action_expert.hidden_dim),
        "resolved_hooks": resolved,
        "video_kv_cache_consumer_specs": [
            {
                "name": spec.name,
                "module_path": spec.module_path,
                "expected_shape": [1, 98, 3072],
            }
            for spec in video_kv_cache_specs(cfg.backbone.video_layers)
        ],
        "status": "ready",
    }
