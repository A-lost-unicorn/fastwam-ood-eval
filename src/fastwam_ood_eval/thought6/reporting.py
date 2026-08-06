"""Phase 6 preregistration and final-report rendering."""

from __future__ import annotations

from typing import Any, Mapping


FINAL_QUESTIONS = (
    "B0 是否与 formal-null 逐位一致？",
    "F0 是否在全部 flow stage 使用 future？",
    "Fsigma 是否严格只在 sigma >= 0.5 使用 future？",
    "低 sigma 是否恢复 B0？",
    "Clean utility 是否非劣？",
    "Camera utility 是否转正？",
    "Correct 是否优于 null？",
    "Correct 是否优于 shuffle？",
    "Fsigma 是否优于 F0？",
    "离线 improvement 是否集中于低 sigma 伤害消除？",
    "Camera success 是否提高？",
    "Clean success 是否保持？",
    "Adapter 平均激活比例是多少？",
    "推理延迟增加多少？",
    "每 task/suite 是否一致？",
    "成功率改善是否通过 future utility 中介？",
    "Label-Oracle 是否只作为诊断上界？",
    "最终属于哪种机制分类？",
    "当前证据能写什么？",
    "当前证据不能写什么？",
)


def not_run_result(schema: str, *, reason: str, prerequisites: list[str]) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "status": "NOT RUN",
        "scientific_result": False,
        "reason": reason,
        "prerequisites": prerequisites,
        "result_fields": None,
    }


def render_not_run_report(*, audit_status: str, blockers: list[str]) -> str:
    lines = [
        "# Phase 6: Sigma-Aware Selective Future Fusion",
        "",
        "**NOT RUN** — this file is a preregistered report shell, not an experiment result.",
        "",
        f"Audit status: `{audit_status}`",
        "",
        "## Current blockers",
        "",
    ]
    lines.extend(f"- {value}" for value in blockers)
    lines.extend(["", "## Frozen questions", ""])
    lines.extend(f"{index}. {question} **NOT RUN**" for index, question in enumerate(FINAL_QUESTIONS, 1))
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "No Phase 6 utility or rollout improvement is claimed before the corresponding frozen gates complete.",
            "Label-Oracle is diagnostic only and may never be described as deployable.",
            "",
        ]
    )
    return "\n".join(lines)


def render_step_table(trace: Mapping[str, Any]) -> str:
    lines = [
        "| step | sigma | gate | adapter_called | adapter_rms | action_hash |",
        "|---:|---:|---:|:---:|---:|---|",
    ]
    for row in trace.get("steps", []):
        lines.append(
            "| {denoising_step_index} | {effective_scheduler_sigma:.6f} | {gate} | "
            "{adapter_called} | {adapter_output_rms:.8f} | `{action_state_sha256}` |".format(**row)
        )
    return "\n".join(lines)

