#!/usr/bin/env python3
"""Build the paper figures from frozen Thought1–Thought3 result artifacts.

The script is read-only with respect to ``outputs/``.  It writes deterministic
SVG figures, two compact CSV tables, and a manifest under ``docs/paper/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "paper" / "figures"
TABLE_DIR = ROOT / "docs" / "paper" / "tables"

SOURCES = {
    "thought1": ROOT
    / "outputs/thought1/fastwam/combined/summary/metrics.json",
    "thought2": ROOT
    / "outputs/thought2/five_category_formal_v1/formal_analysis_v1/formal_analysis.json",
    "thought3_phase1": ROOT
    / "outputs/thought3/phase1_k1_online_counterfactual_v1/aggregate.json",
    "thought3_phase2": ROOT
    / "outputs/thought3/phase2_full_28_4_a0_a1_v1/phase2_training_result.json",
    "thought3_phase2_a0_samples": ROOT
    / "outputs/thought3/phase2_full_28_4_a0_a1_v1/tracks/a0/development_final_objectives.jsonl",
    "thought3_phase2_a1_samples": ROOT
    / "outputs/thought3/phase2_full_28_4_a0_a1_v1/tracks/a1/development_final_objectives.jsonl",
}

COLORS = {
    "blue": "#3568B0",
    "orange": "#E8873A",
    "green": "#3D9271",
    "red": "#C34A4A",
    "gray": "#64748B",
    "light_blue": "#DCE8F7",
    "light_orange": "#FBE7D6",
    "light_green": "#DCEFE7",
    "light_red": "#F5DADA",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "svg.hashsalt": "fastwam-ood-paper-v1",
        }
    )


def save_svg(fig: plt.Figure, name: str) -> None:
    fig.savefig(
        FIGURE_DIR / name,
        format="svg",
        metadata={"Creator": "build_paper_figures.py", "Date": None},
    )
    plt.close(fig)


def figure_ood_success(thought1: dict[str, Any]) -> dict[str, Any]:
    clean = thought1["clean"]
    ood = thought1["ood"]
    category_order = [
        "camera_viewpoints",
        "robot_initial_states",
        "background_textures",
        "objects_layout",
        "light_conditions",
    ]
    display = {
        "camera_viewpoints": "Camera",
        "robot_initial_states": "Robot init.",
        "background_textures": "Background",
        "objects_layout": "Object layout",
        "light_conditions": "Lighting",
    }
    by_category = {
        row["perturbation_category"]: row
        for row in thought1["by_perturbation"]
        if row["condition"] == "ood"
    }
    rows = [
        ("Clean", clean),
        ("OOD overall", ood),
        *[(display[key], by_category[key]) for key in category_order],
    ]

    labels = [label for label, _ in rows]
    values = [100.0 * row["success_rate"] for _, row in rows]
    lower = [
        100.0 * (row["success_rate"] - row["success_ci95_low"])
        for _, row in rows
    ]
    upper = [
        100.0 * (row["success_ci95_high"] - row["success_rate"])
        for _, row in rows
    ]
    colors = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["red"],
        COLORS["red"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["green"],
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar(
        labels,
        values,
        yerr=[lower, upper],
        capsize=3,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 106)
    ax.set_title("Fast-WAM is strong in-distribution but fragile under LIBERO-Plus shifts")
    ax.tick_params(axis="x", rotation=22)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.annotate(
        "−49.55 pp",
        xy=(1, values[1]),
        xytext=(0.48, 77),
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"]},
        ha="center",
        color=COLORS["gray"],
        fontsize=9,
    )
    fig.tight_layout()
    save_svg(fig, "figure1_ood_success.svg")

    return {
        "labels": labels,
        "success_rate_percent": values,
        "ci95_low_percent": [
            100.0 * row["success_ci95_low"] for _, row in rows
        ],
        "ci95_high_percent": [
            100.0 * row["success_ci95_high"] for _, row in rows
        ],
        "attempted": [row["attempted"] for _, row in rows],
    }


def select_metric(
    rows: list[dict[str, Any]], metric: str, probe_mode: str = "all_available"
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["metric"] == metric and row.get("probe_mode") == probe_mode
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {metric}/{probe_mode} row, found {len(matches)}"
        )
    return matches[0]


def paired_panel(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    title: str,
    delta_text: str,
    colors: list[str],
    ylim: tuple[float, float],
) -> None:
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025 * (ylim[1] - ylim[0]),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.text(
        0.5,
        0.96,
        delta_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["gray"],
    )


def figure_future_consistency(thought2: dict[str, Any]) -> dict[str, Any]:
    distance = select_metric(
        thought2["primary_contrasts"], "future_latent_cosine_distance"
    )
    direction = select_metric(
        thought2["primary_contrasts"], "motion_direction_cosine"
    )
    failure_distance = select_metric(
        thought2["outcome_associations"], "future_latent_cosine_distance"
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    paired_panel(
        axes[0],
        ["Clean", "OOD"],
        [distance["clean_task_equal_mean"], distance["ood_task_equal_mean"]],
        "Future–realized distance ↓",
        (
            f"Δ={distance['ood_minus_clean']:+.3f}\n"
            f"95% CI [{distance['ci95_low']:.3f}, {distance['ci95_high']:.3f}]"
        ),
        [COLORS["blue"], COLORS["orange"]],
        (0.0, 0.185),
    )
    paired_panel(
        axes[1],
        ["Clean", "OOD"],
        [direction["clean_task_equal_mean"], direction["ood_task_equal_mean"]],
        "Motion-direction cosine ↑",
        (
            f"Δ={direction['ood_minus_clean']:+.3f}\n"
            f"95% CI [{direction['ci95_low']:.3f}, {direction['ci95_high']:.3f}]"
        ),
        [COLORS["blue"], COLORS["orange"]],
        (0.0, 0.9),
    )
    paired_panel(
        axes[2],
        ["Success", "Failure"],
        [
            failure_distance["success_task_equal_mean"],
            failure_distance["failure_task_equal_mean"],
        ],
        "OOD outcome association",
        (
            f"Failure−success={failure_distance['failure_minus_success']:+.3f}\n"
            f"95% CI [{failure_distance['ci95_low']:.3f}, "
            f"{failure_distance['ci95_high']:.3f}]"
        ),
        [COLORS["green"], COLORS["red"]],
        (0.0, 0.19),
    )
    fig.suptitle(
        "Shadow futures become less consistent under OOD and are associated with failure",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    save_svg(fig, "figure2_future_consistency.svg")

    return {
        "clean_vs_ood": {
            "future_latent_cosine_distance": distance,
            "motion_direction_cosine": direction,
        },
        "ood_failure_association": failure_distance,
    }


def figure_sensitivity_utility(
    phase1: dict[str, Any], phase2: dict[str, Any]
) -> dict[str, Any]:
    pair_order = ["b0_null", "correct_null", "correct_shuffle"]
    pair_labels = ["B0–null", "Correct–null", "Correct–shuffle"]
    means = [phase1["pair_metrics"][key]["l2"]["mean"] for key in pair_order]
    p95 = [phase1["pair_metrics"][key]["l2"]["p95"] for key in pair_order]

    initial = phase2["development_initial_mean_loss"]
    a0 = phase2["tracks"]["A0"]["development_final_mean_loss"]
    a1 = phase2["tracks"]["A1"]["development_final_mean_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    bars = axes[0].bar(
        pair_labels,
        means,
        color=[COLORS["gray"], COLORS["blue"], COLORS["orange"]],
    )
    axes[0].scatter(range(3), p95, marker="_", s=240, color="#111827", zorder=3)
    axes[0].set_ylabel("Action RMS difference")
    axes[0].set_ylim(0, max(p95) * 1.25)
    axes[0].set_title("K=1 future content changes actions")
    axes[0].tick_params(axis="x", rotation=17)
    axes[0].text(
        0.03,
        0.96,
        "bars: mean; black ticks: p95; n=8",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.2,
        color=COLORS["gray"],
    )
    for bar, value in zip(bars, means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(p95) * 0.035,
            f"{value:.4f}",
            ha="center",
            fontsize=8.3,
        )

    loss_labels = ["Initial", "A0 (K=0)", "A1 (K=1)"]
    loss_values = [initial, a0, a1]
    loss_bars = axes[1].bar(
        loss_labels,
        loss_values,
        color=[COLORS["gray"], COLORS["blue"], COLORS["orange"]],
    )
    axes[1].set_ylabel("Held-out action objective ↓")
    axes[1].set_ylim(0.0038, 0.0045)
    axes[1].set_title("Sensitivity does not become offline utility")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for bar, value in zip(loss_bars, loss_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.000018,
            f"{value:.6f}",
            ha="center",
            fontsize=8.3,
        )
    axes[1].text(
        0.5,
        0.07,
        "A1 +3.624% vs A0; worse on 4/4 samples",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8.2,
        color=COLORS["red"],
    )
    fig.suptitle("Future sensitivity is not evidence of future utility", y=1.02, fontsize=12)
    fig.tight_layout()
    save_svg(fig, "figure3_sensitivity_vs_utility.svg")

    return {
        "phase1_action_l2": {
            label: {"mean": mean, "p95": high}
            for label, mean, high in zip(pair_labels, means, p95)
        },
        "phase2_development_loss": {
            "initial": initial,
            "A0_K0": a0,
            "A1_K1": a1,
            "A1_minus_A0": phase2["a1_minus_a0_final_mean_loss"],
        },
    }


def aggregate_development_samples(
    a0_rows: list[dict[str, Any]], a1_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"A0": [], "A1": []}
    )
    for row in a0_rows:
        grouped[row["base_sample_id"]]["A0"].append(float(row["action_loss"]))
    for row in a1_rows:
        grouped[row["base_sample_id"]]["A1"].append(float(row["action_loss"]))

    result = []
    for sample_id, tracks in grouped.items():
        if len(tracks["A0"]) != 32 or len(tracks["A1"]) != 32:
            raise RuntimeError(
                f"Expected 32 matched flows for {sample_id}, got "
                f"{len(tracks['A0'])}/{len(tracks['A1'])}"
            )
        a0 = sum(tracks["A0"]) / len(tracks["A0"])
        a1 = sum(tracks["A1"]) / len(tracks["A1"])
        result.append(
            {
                "sample_id": sample_id,
                "A0_final_mean_loss": a0,
                "A1_final_mean_loss": a1,
                "A1_minus_A0": a1 - a0,
                "A1_vs_A0_percent": 100.0 * (a1 - a0) / a0,
            }
        )
    return result


def figure_phase2_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row["sample_id"][:12] for row in rows]
    values = [row["A1_vs_A0_percent"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bars = ax.barh(labels, values, color=COLORS["red"], alpha=0.86)
    ax.axvline(0.0, color="#111827", linewidth=0.9)
    ax.set_xlabel("A1 loss relative to A0 (%)  → worse")
    ax.set_title("K=1 is worse than matched K=0 on all four development samples")
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.22,
            bar.get_y() + bar.get_height() / 2,
            f"+{value:.2f}%",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    save_svg(fig, "figure4_phase2_per_sample.svg")
    return {"samples": rows}


def figure_evidence_chain() -> dict[str, Any]:
    stages = [
        {
            "title": "Thought 1",
            "question": "Is Fast-WAM robust\nto environment shift?",
            "evidence": "7,571 attempted rollouts",
            "finding": "97.25% → 47.70% success",
            "grade": "Behavioral evaluation",
            "color": COLORS["light_blue"],
        },
        {
            "title": "Thought 2",
            "question": "Does the shadow future\ntrack realized change?",
            "evidence": "732 episodes / 1,010 probes",
            "finding": "OOD distance +0.0316\nDirection −0.1898",
            "grade": "Observational association",
            "color": COLORS["light_orange"],
        },
        {
            "title": "Thought 3 · Phase 1",
            "question": "Does future content\naffect the action?",
            "evidence": "correct / null / shuffle; n=8",
            "finding": "Action hash changed in 8/8",
            "grade": "Technical causal intervention",
            "color": COLORS["light_green"],
        },
        {
            "title": "Thought 3 · Phase 2",
            "question": "Does K=1 improve\nheld-out utility?",
            "evidence": "Matched 28 train /\n4 development",
            "finding": "A1 3.624% worse than A0",
            "grade": "Offline negative result · STOP",
            "color": COLORS["light_red"],
        },
    ]

    fig, ax = plt.subplots(figsize=(12.0, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [0.13, 0.38, 0.63, 0.88]
    width = 0.205
    height = 0.69
    for index, (stage, center) in enumerate(zip(stages, centers)):
        left = center - width / 2
        patch = FancyBboxPatch(
            (left, 0.18),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=stage["color"],
            edgecolor=COLORS["gray"],
            linewidth=1.0,
        )
        ax.add_patch(patch)
        ax.text(center, 0.79, stage["title"], ha="center", weight="bold", fontsize=10.5)
        ax.text(
            center,
            0.65,
            stage["question"],
            ha="center",
            va="center",
            fontsize=8.6,
            wrap=True,
        )
        ax.text(center, 0.48, stage["evidence"], ha="center", fontsize=8.4)
        ax.text(
            center,
            0.36,
            stage["finding"],
            ha="center",
            fontsize=8.6,
            weight="bold",
        )
        ax.text(
            center,
            0.23,
            stage["grade"],
            ha="center",
            fontsize=7.9,
            color=COLORS["gray"],
        )
        if index < len(stages) - 1:
            arrow = FancyArrowPatch(
                (center + width / 2 + 0.006, 0.525),
                (centers[index + 1] - width / 2 - 0.006, 0.525),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color=COLORS["gray"],
            )
            ax.add_patch(arrow)
    ax.text(
        0.5,
        0.06,
        "Evidence ladder: robustness gap → association → action sensitivity → held-out utility",
        ha="center",
        fontsize=10,
        color="#1F2937",
    )
    fig.tight_layout()
    save_svg(fig, "figure5_evidence_chain.svg")
    return {"stages": stages}


def write_tables(
    thought1_data: dict[str, Any],
    thought2_data: dict[str, Any],
    phase1_data: dict[str, Any],
    phase2_data: dict[str, Any],
    sample_rows: list[dict[str, Any]],
) -> None:
    core_path = TABLE_DIR / "core_results.csv"
    with core_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["stage", "comparison", "metric", "estimate", "ci95_low", "ci95_high", "unit"]
        )
        writer.writerow(
            [
                "Thought1",
                "Clean",
                "success_rate",
                thought1_data["clean"]["success_rate"],
                thought1_data["clean"]["success_ci95_low"],
                thought1_data["clean"]["success_ci95_high"],
                "proportion",
            ]
        )
        writer.writerow(
            [
                "Thought1",
                "OOD",
                "success_rate",
                thought1_data["ood"]["success_rate"],
                thought1_data["ood"]["success_ci95_low"],
                thought1_data["ood"]["success_ci95_high"],
                "proportion",
            ]
        )
        for metric in (
            "future_latent_cosine_distance",
            "future_latent_l1",
            "motion_direction_cosine",
        ):
            row = select_metric(thought2_data["primary_contrasts"], metric)
            writer.writerow(
                [
                    "Thought2",
                    "OOD_minus_Clean",
                    metric,
                    row["ood_minus_clean"],
                    row["ci95_low"],
                    row["ci95_high"],
                    "task_equal_difference",
                ]
            )
        for pair in ("b0_null", "correct_null", "correct_shuffle"):
            row = phase1_data["pair_metrics"][pair]["l2"]
            writer.writerow(
                [
                    "Thought3_Phase1",
                    pair,
                    "action_l2_rms",
                    row["mean"],
                    "",
                    row["p95"],
                    "normalized_action",
                ]
            )
        for track in ("A0", "A1"):
            writer.writerow(
                [
                    "Thought3_Phase2",
                    track,
                    "development_final_mean_loss",
                    phase2_data["tracks"][track]["development_final_mean_loss"],
                    "",
                    "",
                    "action_objective",
                ]
            )

    sample_path = TABLE_DIR / "phase2_per_sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    thought1 = load_json(SOURCES["thought1"])
    thought2 = load_json(SOURCES["thought2"])
    phase1 = load_json(SOURCES["thought3_phase1"])
    phase2 = load_json(SOURCES["thought3_phase2"])
    sample_rows = aggregate_development_samples(
        load_jsonl(SOURCES["thought3_phase2_a0_samples"]),
        load_jsonl(SOURCES["thought3_phase2_a1_samples"]),
    )

    figure_data = {
        "figure1_ood_success.svg": figure_ood_success(thought1),
        "figure2_future_consistency.svg": figure_future_consistency(thought2),
        "figure3_sensitivity_vs_utility.svg": figure_sensitivity_utility(
            phase1, phase2
        ),
        "figure4_phase2_per_sample.svg": figure_phase2_samples(sample_rows),
        "figure5_evidence_chain.svg": figure_evidence_chain(),
    }
    write_tables(thought1, thought2, phase1, phase2, sample_rows)

    manifest = {
        "schema_version": "fastwam_ood.paper_figures.v1",
        "generator": "scripts/build_paper_figures.py",
        "source_artifacts": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for key, path in SOURCES.items()
        },
        "figures": figure_data,
    }
    (FIGURE_DIR / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
