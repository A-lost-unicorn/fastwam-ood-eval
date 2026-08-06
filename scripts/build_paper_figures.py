#!/usr/bin/env python3
"""Build the six core paper figures from frozen Thought1–Thought5 artifacts.

The script is read-only with respect to ``outputs/``.  It writes deterministic
SVG figures, evidence tables, and a manifest under ``docs/paper/``.
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
    "thought4_evidence": ROOT
    / "outputs/thought4/phase4_geometry_action_diagnosis_v6/diagnostic_evidence.json",
    "thought4_intervention": ROOT
    / "outputs/thought4/phase4_geometry_action_diagnosis_v6/intervention_results.json",
    "thought5_direction": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/pilot_direction.json",
    "thought5_training": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/training_results.json",
    "thought5_representation": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/representation_results.json",
    "thought5_future_geometry": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/future_geometry_results.json",
    "thought5_future_utility": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/future_utility_results.json",
    "thought5_rollout": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4/rollout_results.json",
    "thought5_readonly": ROOT
    / "outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4_readonly_failure_v1/analysis_result.json",
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
    save_svg(fig, "figure2_ood_success.svg")

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
    save_svg(fig, "figure3_future_consistency.svg")

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
    save_svg(fig, "figure4_sensitivity_vs_utility.svg")

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


def aggregate_variant_utility(
    condition_data: dict[str, Any], variant: str
) -> float:
    """Average Clean/Camera utility; both conditions have 128 matched rows."""
    return sum(
        condition_data[variant]["conditions"][condition][
            "mean_utility_a0_minus_a1"
        ]
        for condition in ("clean", "camera")
    ) / 2.0


def figure_evidence_chain(
    thought1: dict[str, Any],
    thought2: dict[str, Any],
    phase1: dict[str, Any],
    phase2: dict[str, Any],
    thought4: dict[str, Any],
    representation: dict[str, Any],
    rollout: dict[str, Any],
    readonly: dict[str, Any],
) -> dict[str, Any]:
    distance = select_metric(
        thought2["primary_contrasts"], "future_latent_cosine_distance"
    )
    category_rows = {
        row["perturbation_category"]: row
        for row in thought1["by_perturbation"]
        if row["condition"] == "ood"
    }
    a0_loss = phase2["tracks"]["A0"]["development_final_mean_loss"]
    a1_loss = phase2["tracks"]["A1"]["development_final_mean_loss"]
    condition_data = readonly["questions"]["condition_failure"]
    g3_utility = aggregate_variant_utility(condition_data, "G3")
    g3_sigma = readonly["questions"]["flow_and_action_localization"][
        "flow_objective"
    ]["variants"]["G3"]
    stages = [
        {
            "title": "1 · Failure",
            "question": "Where does the\npolicy fail?",
            "evidence": f"Thought 1 · {thought1['clean']['attempted'] + thought1['ood']['attempted']:,} rollouts",
            "finding": (
                f"OOD drop {(thought1['clean']['success_rate'] - thought1['ood']['success_rate']) * 100:.2f} pp\n"
                f"Camera SR {category_rows['camera_viewpoints']['success_rate'] * 100:.2f}%"
            ),
            "grade": "Behavioral failure",
            "color": COLORS["light_blue"],
        },
        {
            "title": "2 · Representation",
            "question": "Does the shadow future\ntrack realized change?",
            "evidence": "Thought 2 · 732 episodes",
            "finding": f"OOD distance {distance['ood_minus_clean']:+.4f}\nAssociation, not cause",
            "grade": "Representation audit",
            "color": COLORS["light_orange"],
        },
        {
            "title": "3 · Sensitivity",
            "question": "Does future content\naffect action and utility?",
            "evidence": f"Thought 3 · n={phase1['sample_count']} + matched 28/4",
            "finding": (
                f"Action changed {phase1['sample_count']}/{phase1['sample_count']}\n"
                f"K=1 loss {(a1_loss / a0_loss - 1) * 100:+.3f}% vs K=0"
            ),
            "grade": "Sensitivity ≠ utility",
            "color": COLORS["light_green"],
        },
        {
            "title": "4 · Geometry",
            "question": "Is Camera failure a\ngeometry-equivariance gap?",
            "evidence": "64 states / 36 interventions",
            "finding": (
                f"Gap {thought4['camera_paired_gap']['estimate_mean']:.4f} m vs "
                f"{thought4['lighting_paired_gap']['estimate_mean']:.4f} m\n"
                "36/36 action effect"
            ),
            "grade": "Thought 4 · localization",
            "color": COLORS["light_orange"],
        },
        {
            "title": "5 · Intervention",
            "question": "Can Geo-REPA +\nPose/Ray repair utility?",
            "evidence": "3-GPU matched pilot; 8/4/4",
            "finding": (
                f"Gap −{representation['gap_reduction_fraction'] * 100:.2f}% (<25%)\n"
                f"Utility {g3_utility:+.6f}; Camera "
                f"{round(4 * rollout['summaries']['B1:camera']['success_rate'])}/4="
                f"{round(4 * rollout['summaries']['G3:camera']['success_rate'])}/4"
            ),
            "grade": "Thought 5 · falsification",
            "color": COLORS["light_red"],
        },
        {
            "title": "6 · Failure analysis",
            "question": "Where does the\nintervention still fail?",
            "evidence": "Read-only post-hoc audit",
            "finding": (
                f"Clean utility {condition_data['G3']['conditions']['clean']['mean_utility_a0_minus_a1']:+.4f}\n"
                f"Low-σ utility {g3_sigma['noise_bins']['[0.00,0.25)']['mean_utility']:+.4f}"
            ),
            "grade": "Condition / noise dependence",
            "color": COLORS["light_red"],
        },
    ]

    fig, ax = plt.subplots(figsize=(16.5, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [0.085, 0.251, 0.417, 0.583, 0.749, 0.915]
    width = 0.142
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
        ax.text(center, 0.79, stage["title"], ha="center", weight="bold", fontsize=9.6)
        ax.text(
            center,
            0.65,
            stage["question"],
            ha="center",
            va="center",
            fontsize=7.7,
            wrap=True,
        )
        ax.text(center, 0.48, stage["evidence"], ha="center", fontsize=7.5)
        ax.text(
            center,
            0.36,
            stage["finding"],
            ha="center",
            fontsize=7.7,
            weight="bold",
        )
        ax.text(
            center,
            0.23,
            stage["grade"],
            ha="center",
            fontsize=7.0,
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
        "Hypothesis → intervention → falsification: a Camera-equivariance repair did not restore aggregate future utility",
        ha="center",
        fontsize=10,
        color="#1F2937",
    )
    fig.tight_layout()
    save_svg(fig, "figure1_research_chain.svg")
    return {"stages": stages}


def figure_camera_equivariance_gap(
    evidence: dict[str, Any], intervention: dict[str, Any]
) -> dict[str, Any]:
    gap_rows = [evidence["camera_paired_gap"], evidence["lighting_paired_gap"]]
    gap_labels = ["Camera", "Lighting"]
    gap_values = [row["estimate_mean"] for row in gap_rows]
    gap_lower = [row["estimate_mean"] - row["lower_min"] for row in gap_rows]
    gap_upper = [row["upper_max"] - row["estimate_mean"] for row in gap_rows]

    shift_data = intervention["geometry_coordinate_condition_shift"][
        "condition_summaries"
    ]
    shift_keys = ["camera", "lighting", "robot_init"]
    shift_labels = ["Camera", "Lighting", "Robot init.*"]
    shift_rows = [
        shift_data[key]["coordinate_l2_grouped_bootstrap"] for key in shift_keys
    ]
    shift_values = [row["estimate"] for row in shift_rows]
    shift_lower = [row["estimate"] - row["lower"] for row in shift_rows]
    shift_upper = [row["upper"] - row["estimate"] for row in shift_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    bars = axes[0].bar(
        gap_labels,
        gap_values,
        yerr=[gap_lower, gap_upper],
        capsize=4,
        color=[COLORS["red"], COLORS["green"]],
    )
    axes[0].set_ylabel("Paired geometry RMSE gap (m) ↓")
    axes[0].set_title("Exact-state probe gap")
    axes[0].set_ylim(0, max(row["upper_max"] for row in gap_rows) * 1.18)
    for bar, value in zip(bars, gap_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.6f}",
            ha="center",
            fontsize=8.5,
        )
    axes[0].text(
        0.5,
        0.94,
        "Camera is 73.87% larger than Lighting",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["gray"],
    )

    shift_bars = axes[1].bar(
        shift_labels,
        shift_values,
        yerr=[shift_lower, shift_upper],
        capsize=4,
        color=[COLORS["red"], COLORS["green"], COLORS["gray"]],
    )
    axes[1].set_ylabel("Rank-3 geometry-subspace shift ↓")
    axes[1].set_title("Probe-defined geometry coordinates")
    axes[1].set_ylim(0, max(row["upper"] for row in shift_rows) * 1.18)
    for bar, value in zip(shift_bars, shift_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            fontsize=8.5,
        )
    axes[1].text(
        0.5,
        0.94,
        "Camera−Lighting=+0.146 [0.089, 0.200]\n36/36 shuffles changed action",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color=COLORS["gray"],
    )
    axes[1].text(
        0.5,
        -0.19,
        "* Robot-init is not an exact-state pair and is exploratory only.",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=7.5,
        color=COLORS["gray"],
    )
    fig.suptitle(
        "Camera shift creates the largest frozen geometry-equivariance gap",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    save_svg(fig, "figure5_camera_equivariance_gap.svg")
    return {
        "probe_gap": dict(zip(gap_labels, gap_values)),
        "geometry_subspace_shift": dict(zip(shift_labels, shift_values)),
        "camera_minus_lighting": evidence[
            "geometry_coordinate_camera_minus_lighting"
        ],
        "action_shuffle_above_floor_fraction": evidence[
            "intervention_fraction_above_floor"
        ],
    }


def figure_phase5_failure_decomposition(
    representation: dict[str, Any],
    readonly: dict[str, Any],
) -> dict[str, Any]:
    variants = ["B1", "G3", "G4"]
    gaps = [
        representation["b1_camera_gap"],
        representation["g3_camera_gap"],
        representation["g4_camera_gap"],
    ]
    threshold = 0.75 * gaps[0]

    condition_data = readonly["questions"]["condition_failure"]
    clean_utility = [
        condition_data[variant]["conditions"]["clean"]["mean_utility_a0_minus_a1"]
        for variant in variants
    ]
    camera_utility = [
        condition_data[variant]["conditions"]["camera"]["mean_utility_a0_minus_a1"]
        for variant in variants
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    gap_bars = axes[0].bar(
        variants,
        gaps,
        color=[COLORS["gray"], COLORS["blue"], COLORS["orange"]],
    )
    axes[0].axhline(
        threshold,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.0,
        label="25% reduction threshold",
    )
    axes[0].set_ylabel("Camera representation gap ↓")
    axes[0].set_title("Weak, non-specific representation change")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[0].set_ylim(0, max(gaps) * 1.28)
    for bar, value in zip(gap_bars, gaps):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(gaps) * 0.035,
            f"{value:.6f}",
            ha="center",
            fontsize=7.7,
        )
    axes[0].text(
        0.5,
        0.03,
        "G3: −20.94%; G4: −25.82%",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=7.8,
        color=COLORS["gray"],
    )

    aggregate_utility = [
        aggregate_variant_utility(condition_data, variant) for variant in variants
    ]
    utility_bars = axes[1].bar(
        variants,
        aggregate_utility,
        color=[COLORS["gray"], COLORS["blue"], COLORS["orange"]],
    )
    axes[1].axhline(0.0, color="#111827", linewidth=0.9)
    axes[1].set_ylim(-0.019, 0.003)
    axes[1].set_ylabel("Future utility: loss(null) − loss(correct)")
    axes[1].set_title("Aggregate future utility remains negative")
    for bar, value in zip(utility_bars, aggregate_utility):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.0007,
            f"{value:+.6f}",
            ha="center",
            va="top",
            fontsize=8.0,
        )
    axes[1].text(
        0.5,
        0.06,
        "G3 reduces harm but does not cross zero",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=7.8,
        color=COLORS["gray"],
    )

    sigma_rows = readonly["questions"]["flow_and_action_localization"][
        "flow_objective"
    ]["variants"]["G3"]
    sigma_labels = ["0–.25", ".25–.50", ".50–.75", ".75–1"]
    sigma_bins = list(sigma_rows["noise_bins"].values())
    sigma_values = [row["mean_utility"] for row in sigma_bins]
    sigma_clean = [row["by_condition"]["clean"]["mean_utility"] for row in sigma_bins]
    sigma_camera = [row["by_condition"]["camera"]["mean_utility"] for row in sigma_bins]
    x_positions = list(range(len(sigma_labels)))
    sigma_bars = axes[2].bar(
        x_positions,
        sigma_values,
        color=[COLORS["red"] if value < 0 else COLORS["green"] for value in sigma_values],
        alpha=0.82,
        label="Overall",
    )
    axes[2].plot(x_positions, sigma_clean, "o-", color=COLORS["blue"], label="Clean")
    axes[2].plot(x_positions, sigma_camera, "s-", color=COLORS["orange"], label="Camera")
    axes[2].axhline(0.0, color="#111827", linewidth=0.9)
    axes[2].set_xticks(x_positions, sigma_labels)
    axes[2].set_ylim(-0.07, 0.025)
    axes[2].set_xlabel("Effective sigma bucket")
    axes[2].set_ylabel("G3 future utility")
    axes[2].set_title("Post-hoc harm concentrates at low σ")
    axes[2].legend(frameon=False, fontsize=7.6, ncol=3)
    axes[2].text(
        0.5,
        0.05,
        f"Pearson r={sigma_rows['pearson_sigma_utility']:+.3f}; read-only exploratory",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=7.4,
        color=COLORS["gray"],
    )
    for bar, value in zip(sigma_bars, sigma_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.002 if value >= 0 else -0.004),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.0,
        )

    fig.suptitle(
        "Phase 5: weak representation movement does not restore aggregate future utility",
        y=1.02,
        fontsize=11.5,
    )
    fig.tight_layout()
    save_svg(fig, "figure6_phase5_failure_decomposition.svg")

    return {
        "camera_representation_gap": dict(zip(variants, gaps)),
        "preregistered_g3_threshold": threshold,
        "aggregate_future_utility": dict(zip(variants, aggregate_utility)),
        "future_utility_by_condition": {
            variant: {"Clean": clean, "Camera": camera}
            for variant, clean, camera in zip(variants, clean_utility, camera_utility)
        },
        "g3_sigma_bucket_utility": dict(zip(sigma_labels, sigma_values)),
        "g3_sigma_utility_pearson": sigma_rows["pearson_sigma_utility"],
    }


def write_tables(
    thought1_data: dict[str, Any],
    thought2_data: dict[str, Any],
    phase1_data: dict[str, Any],
    phase2_data: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    thought4_evidence: dict[str, Any],
    thought4_intervention: dict[str, Any],
    thought5_training: dict[str, Any],
    thought5_representation: dict[str, Any],
    thought5_future_geometry: dict[str, Any],
    thought5_future_utility: dict[str, Any],
    thought5_rollout: dict[str, Any],
    thought5_readonly: dict[str, Any],
) -> None:
    core_path = TABLE_DIR / "core_results.csv"
    with core_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
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
        for condition, row in (
            ("Camera", thought4_evidence["camera_paired_gap"]),
            ("Lighting", thought4_evidence["lighting_paired_gap"]),
        ):
            writer.writerow(
                [
                    "Thought4",
                    f"{condition}_minus_Clean",
                    "video_translation_rmse_gap",
                    row["estimate_mean"],
                    row["lower_min"],
                    row["upper_max"],
                    "m_conservative_three_seed_envelope",
                ]
            )
        writer.writerow(
            [
                "Thought4",
                "Camera_minus_Lighting",
                "rank3_geometry_subspace_shift",
                thought4_evidence["geometry_coordinate_camera_minus_lighting"][
                    "estimate"
                ],
                thought4_evidence["geometry_coordinate_camera_minus_lighting"][
                    "lower"
                ],
                thought4_evidence["geometry_coordinate_camera_minus_lighting"][
                    "upper"
                ],
                "coordinate_l2",
            ]
        )
        for variant, value in zip(
            ("B1", "G3", "G4"),
            (
                thought5_representation["b1_camera_gap"],
                thought5_representation["g3_camera_gap"],
                thought5_representation["g4_camera_gap"],
            ),
        ):
            writer.writerow(
                [
                    "Thought5",
                    variant,
                    "camera_representation_gap",
                    value,
                    "",
                    "",
                    "probe_error",
                ]
            )

    sample_path = TABLE_DIR / "phase2_per_sample.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    thought5_path = TABLE_DIR / "thought5_pilot_diagnostics.csv"
    header = [
        "section",
        "variant",
        "condition",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
        "unit",
        "evidence_role",
    ]
    rows: list[list[Any]] = []
    for variant, value in zip(
        ("B1", "G3", "G4"),
        (
            thought5_representation["b1_camera_gap"],
            thought5_representation["g3_camera_gap"],
            thought5_representation["g4_camera_gap"],
        ),
    ):
        rows.append(
            [
                "representation",
                variant,
                "camera_minus_clean",
                "camera_representation_gap",
                value,
                "",
                "",
                "probe_error",
                "directional_pilot",
            ]
        )

    for variant, value in thought5_future_geometry["main_camera_error"].items():
        rows.append(
            [
                "future_geometry",
                variant,
                "camera",
                "future_camera_geometry_rmse",
                value,
                "",
                "",
                "rmse",
                "directional_pilot",
            ]
        )

    condition_data = thought5_readonly["questions"]["condition_failure"]
    for variant in ("B1", "G3", "G4"):
        for condition in ("clean", "camera"):
            row = condition_data[variant]["conditions"][condition]
            interval = row["posthoc_episode_grouped_bootstrap"]
            rows.append(
                [
                    "future_utility",
                    variant,
                    condition,
                    "loss_null_minus_correct",
                    row["mean_utility_a0_minus_a1"],
                    interval["lower"],
                    interval["upper"],
                    "action_objective",
                    "posthoc_readonly_exploratory",
                ]
            )

    aggregate_interval = thought5_future_utility[
        "g3_correct_minus_null_utility_grouped_bootstrap"
    ]
    rows.append(
        [
            "future_utility",
            "G3",
            "clean_and_camera",
            "loss_null_minus_correct",
            aggregate_interval["estimate"],
            aggregate_interval["lower"],
            aggregate_interval["upper"],
            "action_objective",
            "directional_pilot_primary",
        ]
    )

    for key, summary in thought5_rollout["summaries"].items():
        variant, condition = key.split(":", maxsplit=1)
        rows.append(
            [
                "rollout",
                variant,
                condition,
                "success_rate",
                summary["success_rate"],
                "",
                "",
                "proportion_n4",
                "directional_pilot",
            ]
        )

    raypose = thought5_readonly["questions"]["raypose_lora_training_use"]
    for variant in ("G3", "G4"):
        rows.append(
            [
                "mechanism",
                variant,
                "all",
                "ray_pose_tanh_gate",
                raypose["variants"][variant]["ray_pose_tanh_gate"],
                "",
                "",
                "scalar",
                "posthoc_readonly_exploratory",
            ]
        )

    with thought5_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    cohort_rows = [
        ["Thought1", "Clean online rollout", "4 suites / 40 tasks", thought1_data["clean"]["attempted"], "FORMAL"],
        ["Thought1", "OOD online rollout", "5 perturbation categories", thought1_data["ood"]["attempted"], "FORMAL"],
        ["Thought2", "Shadow diagnostic episode", "200 Clean + 532 OOD", 732, "FORMAL-COLLECTION / POST-RUN"],
        ["Thought2", "Future probe", "2,020 aligned frames", 1010, "POST-RUN ASSOCIATION"],
        ["Thought3-Phase1", "Fixed action counterfactual sample", "correct/null/shuffle", phase1_data["sample_count"], "SMOKE"],
        ["Thought3-Phase2", "Train / development sample", "single LIBERO-Goal task", "28 / 4", "VALID OFFLINE DEVELOPMENT"],
        ["Thought4", "Base state / paired render", "40/12/12 split; 4 conditions", "64 / 256", "FORMAL DIAGNOSTIC"],
        ["Thought4", "Action intervention", "12 held-out states × 3 seeds", 36, "FORMAL DIAGNOSTIC"],
        ["Thought5", "Train / development / pilot-test", "single task; disjoint episodes", "8 / 4 / 4", "DIRECTIONAL PILOT"],
        ["Thought5", "Matched condition rollout", "B1/G3/G4 × 4 conditions × 4 episodes", 48, "DIRECTIONAL PILOT"],
        ["Thought5", "Future-utility objective", "3 variants × 2 conditions × 128", 768, "DIRECTIONAL PILOT"],
    ]
    with (TABLE_DIR / "cohort_scale.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["stage", "unit", "scope", "count", "evidence_grade"])
        writer.writerows(cohort_rows)

    condition_data = thought5_readonly["questions"]["condition_failure"]
    sigma_data = thought5_readonly["questions"]["flow_and_action_localization"][
        "flow_objective"
    ]["variants"]["G3"]
    stage_rows = [
        ["Thought1", "Failure", "Environment shift", "Clean 97.25%; OOD 47.70%; Camera 15.13%", "OOD robustness gap established"],
        ["Thought2", "Representation", "Control-isolated shadow future", "Distance +0.0316; direction cosine -0.1898", "Association established; causality unresolved"],
        ["Thought3", "Sensitivity / utility", "correct/null/shuffle + matched K0/K1", "8/8 action sensitivity; K1 3.624% worse than K0", "Sensitivity does not imply utility"],
        ["Thought4", "Geometry", "Exact-state probes + rank-3 shuffle", "Camera gap 0.020273 m; Lighting 0.011660 m; 36/36 action effect", "camera_equivariance_gap"],
        ["Thought5", "Intervention", "B1/G3/G4 Geo-REPA + Pose/Ray pilot", f"G3 gap -{thought5_representation['gap_reduction_fraction'] * 100:.2f}%; utility {aggregate_variant_utility(condition_data, 'G3'):+.6f}; Camera 1/4=1/4", "Full mechanism chain not supported; STOP"],
        ["Thought5-readonly", "Failure analysis", "Immutable post-hoc decomposition", f"Clean utility {condition_data['G3']['conditions']['clean']['mean_utility_a0_minus_a1']:+.6f}; sigma<0.25 utility {sigma_data['noise_bins']['[0.00,0.25)']['mean_utility']:+.6f}", "Condition/noise-stage dependence hypothesis"],
    ]
    with (TABLE_DIR / "stage_findings.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["stage", "role", "intervention", "finding", "decision"])
        writer.writerows(stage_rows)

    def selected_development_row(variant: str) -> dict[str, Any]:
        track = thought5_training["tracks"][variant]
        matches = [
            row
            for row in track["development_rows"]
            if row["step"] == track["selected_step"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one selected development row for {variant}")
        return matches[0]

    phase5_rows = []
    for variant, gap in zip(
        ("B1", "G3", "G4"),
        (
            thought5_representation["b1_camera_gap"],
            thought5_representation["g3_camera_gap"],
            thought5_representation["g4_camera_gap"],
        ),
    ):
        selected = selected_development_row(variant)
        aggregate_utility = aggregate_variant_utility(condition_data, variant)
        phase5_rows.append(
            [
                variant,
                selected["selection_objective"],
                selected["original_fastwam"],
                gap,
                (thought5_representation["b1_camera_gap"] - gap)
                / thought5_representation["b1_camera_gap"],
                thought5_future_geometry["main_camera_error"][variant],
                aggregate_utility,
                condition_data[variant]["conditions"]["clean"]["mean_utility_a0_minus_a1"],
                condition_data[variant]["conditions"]["camera"]["mean_utility_a0_minus_a1"],
                thought5_rollout["summaries"][f"{variant}:clean"]["success_rate"],
                thought5_rollout["summaries"][f"{variant}:camera"]["success_rate"],
                thought5_rollout["summaries"][f"{variant}:lighting"]["success_rate"],
                thought5_rollout["summaries"][f"{variant}:robot_init"]["success_rate"],
            ]
        )
    with (TABLE_DIR / "phase5_b1_g3_g4.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "variant",
                "development_selection_objective",
                "development_original_fastwam_loss",
                "camera_representation_gap",
                "gap_reduction_vs_b1",
                "camera_future_geometry_rmse",
                "aggregate_future_utility",
                "clean_future_utility_posthoc",
                "camera_future_utility_posthoc",
                "clean_success_rate_n4",
                "camera_success_rate_n4",
                "lighting_success_rate_n4",
                "robot_init_success_rate_n4",
            ]
        )
        writer.writerows(phase5_rows)

    boundary_rows = [
        ["Fast-WAM is fragile under the evaluated LIBERO-Plus shifts", "supported", "Fixed release checkpoint; simulation only"],
        ["OOD shadow-future consistency degrades", "supported association", "Decoded-frame proxy; not a causal failure explanation"],
        ["K=1 future content changes actions", "supported technical sensitivity", "One checkpoint/task; n=8; not success"],
        ["K=1 improves held-out utility", "negative in frozen recipe", "Single task/seed; 28/4; offline objective"],
        ["Camera equivariance is the unique cause of failure", "not supported", "Thought4 localizes a gap but does not establish sufficiency"],
        ["Current Geo-REPA + Pose/Ray recipe restores future utility", "pilot not supported", "G3 misses H1/H2/H3 gates; formal locked"],
        ["Geo-REPA or RayPose is universally ineffective", "not answered", "G4 is not a complete geometry control; G1/G2 absent"],
        ["Future utility is condition/noise-stage dependent", "exploratory hypothesis", "Read-only single-task post-hoc decomposition"],
    ]
    with (TABLE_DIR / "evidence_boundaries.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["claim", "status", "required_boundary"])
        writer.writerows(boundary_rows)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    thought1 = load_json(SOURCES["thought1"])
    thought2 = load_json(SOURCES["thought2"])
    phase1 = load_json(SOURCES["thought3_phase1"])
    phase2 = load_json(SOURCES["thought3_phase2"])
    thought4_evidence = load_json(SOURCES["thought4_evidence"])
    thought4_intervention = load_json(SOURCES["thought4_intervention"])
    thought5_training = load_json(SOURCES["thought5_training"])
    thought5_representation = load_json(SOURCES["thought5_representation"])
    thought5_future_geometry = load_json(SOURCES["thought5_future_geometry"])
    thought5_future_utility = load_json(SOURCES["thought5_future_utility"])
    thought5_rollout = load_json(SOURCES["thought5_rollout"])
    thought5_readonly = load_json(SOURCES["thought5_readonly"])
    sample_rows = aggregate_development_samples(
        load_jsonl(SOURCES["thought3_phase2_a0_samples"]),
        load_jsonl(SOURCES["thought3_phase2_a1_samples"]),
    )

    figure_data = {
        "figure1_research_chain.svg": figure_evidence_chain(
            thought1,
            thought2,
            phase1,
            phase2,
            thought4_evidence,
            thought5_representation,
            thought5_rollout,
            thought5_readonly,
        ),
        "figure2_ood_success.svg": figure_ood_success(thought1),
        "figure3_future_consistency.svg": figure_future_consistency(thought2),
        "figure4_sensitivity_vs_utility.svg": figure_sensitivity_utility(
            phase1, phase2
        ),
        "figure5_camera_equivariance_gap.svg": figure_camera_equivariance_gap(
            thought4_evidence, thought4_intervention
        ),
        "figure6_phase5_failure_decomposition.svg": figure_phase5_failure_decomposition(
            thought5_representation,
            thought5_readonly,
        ),
    }
    write_tables(
        thought1,
        thought2,
        phase1,
        phase2,
        sample_rows,
        thought4_evidence,
        thought4_intervention,
        thought5_training,
        thought5_representation,
        thought5_future_geometry,
        thought5_future_utility,
        thought5_rollout,
        thought5_readonly,
    )

    manifest = {
        "schema_version": "fastwam_ood.paper_figures.v3",
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
