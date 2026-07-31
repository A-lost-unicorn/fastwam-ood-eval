"""Layer summaries, integrity manifests and the Phase 4 diagnostic report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.thought4.schemas import sha256_canonical, sha256_file


def build_layer_summary(
    video_result: Mapping[str, Any],
    action_result: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in (video_result, action_result):
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in payload.get("rows", []):
            key = (
                row["source"],
                row["module_path"],
                row["layer_index"],
                row["denoise_step_index"],
                row["pooling"],
                row["target"],
                row["probe_kind"],
            )
            grouped[key].append(row)
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
            development_losses = [float(value["development_loss"]) for value in values]
            camera_gaps = [
                float(value["gaps_vs_clean_rmse"]["camera"]) for value in values
            ]
            lighting_gaps = [
                float(value["gaps_vs_clean_rmse"]["lighting"]) for value in values
            ]
            robot_gaps = [
                float(value["gaps_vs_clean_rmse"]["robot_init"]) for value in values
            ]
            rows.append(
                {
                    "source": key[0],
                    "module_path": key[1],
                    "layer_index": key[2],
                    "denoise_step_index": key[3],
                    "pooling": key[4],
                    "target": key[5],
                    "probe_kind": key[6],
                    "seed_count": len(values),
                    "development_loss_mean": sum(development_losses)
                    / len(development_losses),
                    "camera_minus_clean_rmse_mean": sum(camera_gaps)
                    / len(camera_gaps),
                    "lighting_minus_clean_rmse_mean": sum(lighting_gaps)
                    / len(lighting_gaps),
                    "robot_init_minus_clean_rmse_mean": sum(robot_gaps)
                    / len(robot_gaps),
                    "test_used_for_layer_selection": False,
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.layer_summary.v1",
        "rows": rows,
    }
    payload["summary_sha256"] = sha256_canonical(payload)
    return payload


def execution_integrity(
    *,
    config_fingerprint: str,
    backbone_sha_before: str,
    backbone_sha_after: str,
    checkpoint_sha256: str,
    cohort_sha256: str,
    future_rgb_read: bool,
    success_outcome_read: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.execution_integrity.v1",
        "config_fingerprint": config_fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "backbone_parameter_sha256_before": backbone_sha_before,
        "backbone_parameter_sha256_after": backbone_sha_after,
        "backbone_unchanged": backbone_sha_before == backbone_sha_after,
        "backbone_trainable_parameter_count": 0,
        "future_rgb_read": bool(future_rgb_read),
        "success_outcome_read": bool(success_outcome_read),
        "fastwam_training_performed": False,
        "probe_training_only": True,
    }
    if not payload["backbone_unchanged"]:
        raise RuntimeError("frozen Fast-WAM parameter SHA changed")
    if future_rgb_read or success_outcome_read:
        raise RuntimeError("Thought4 diagnostic scope was violated")
    payload["integrity_sha256"] = sha256_canonical(payload)
    return payload


def build_artifact_manifest(
    run_dir: str | Path,
    paths: Sequence[str | Path],
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).resolve()
        if root not in path.parents:
            raise RuntimeError(f"artifact escapes run directory: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "thought4.phase4.artifact_manifest.v1",
        "artifacts": sorted(rows, key=lambda row: row["path"]),
    }
    payload["manifest_sha256"] = sha256_canonical(payload)
    return payload


def diagnostic_report_markdown(
    *,
    method_selection: Mapping[str, Any],
    evidence: Mapping[str, Any],
    layer_summary: Mapping[str, Any],
    intervention: Mapping[str, Any],
) -> str:
    recommendation = str(method_selection["recommendation"])
    classification = str(method_selection["classification"])
    development = {
        "Geo-REPA": "约 1–2 周",
        "SE(3)-Align": "约 1 周",
        "Geo-REPA + relative pose / camera-ray equivariance": "约 2–3 周",
        "geometry hypothesis not supported": "先用约 3–5 天审计预处理/坐标系",
    }[recommendation]
    rows = layer_summary.get("rows", [])
    best_video = min(
        (
            row
            for row in rows
            if row["source"] == "A"
            and row["target"] == "eef_object_translation_camera"
            and row["probe_kind"] == "linear"
        ),
        key=lambda row: float(row["development_loss_mean"]),
        default=None,
    )
    best_action = min(
        (
            row
            for row in rows
            if row["source"] == "B"
            and row["target"] == "action_se3_trajectory"
            and row["probe_kind"] == "linear"
        ),
        key=lambda row: float(row["development_loss_mean"]),
        default=None,
    )
    return (
        "# Thought4 Phase 4：Geometry–Action Gap Diagnosis\n\n"
        "## 证据链\n\n"
        "Camera OOD failure → OOD future consistency degradation → "
        "future-content action sensitivity → no held-out future utility → "
        "Phase 4 gap localization → targeted repair → future Camera OOD validation。\n\n"
        "这不是因为其他工作使用 depth 而直接加入 depth；本阶段先在冻结模型中"
        "定位几何是否存在、是否进入 Action DiT、跨视角时是否失效。\n\n"
        "## 唯一方法选择\n\n"
        f"- 分类：`{classification}`\n"
        f"- 唯一建议：**{recommendation}**\n"
        f"- 预计开发时间：{development}\n"
        f"- 判定理由：{method_selection['rationale']}\n\n"
        "## 十二个问题\n\n"
        "1. Video 主目标（camera-frame EEF–object translation）最可读位置："
        f"`{best_video}`。\n"
        "2. Action 主目标（完整 SE(3) trajectory）最可读位置："
        f"`{best_action}`。\n"
        f"3. Clean 几何可读：`{evidence['video_clean_readable']}`。\n"
        "4. Camera 相对 Lighting 的 exact-state gap 由冻结 paired bootstrap "
        f"判定：camera={evidence['camera_paired_gap']}，"
        f"lighting={evidence['lighting_paired_gap']}；所选 geometry subspace 的 "
        "Camera−Lighting coordinate-shift="
        f"{evidence['geometry_coordinate_camera_minus_lighting']}。\n"
        "5. Robot-init 单独报告，未当作 exact-state pair；"
        f"与 Camera gap 差为 {evidence['robot_init_minus_camera_gap']}。\n"
        "6. Video→Action 几何传递：Action hidden 的当前 EEF–object geometry "
        f"可读性={evidence['action_geometry_clean_readable']}；未来 SE(3) "
        f"可读性={evidence['action_clean_readable']}。\n"
        "7. Geometry shuffle 的稳定动作影响："
        f"{intervention.get('correct_shuffle_above_floor_count', 0)}/"
        f"{intervention.get('comparison_count', 0)} 超过 replay floor。\n"
        f"8. 缺口定位：`{classification}`。\n"
        f"9. 只建议实现：{recommendation}。\n"
        "10. 最小后续评测：冻结 checkpoint/seed/action schedule，先做 held-out "
        "representation/SE(3) 指标，再做预注册 Clean/Camera/Lighting paired "
        "rollout；本阶段尚未做 rollout。\n"
        "11. 能证明：冻结表征的可读性、配对条件 gap、probe-defined 子空间的"
        "动作敏感性。不能证明：成功率提升、几何是唯一原因、建议方法必然有效。\n"
        f"12. 唯一允许启动的后续分支：`{recommendation}`。\n\n"
        "## 结论边界\n\n"
        "本报告是诊断结果，不是 OOD improvement 或 policy success 结果；"
        "没有读取 future RGB，没有训练 Fast-WAM 主干，也没有运行环境成功率"
        "rollout。\n"
    )
