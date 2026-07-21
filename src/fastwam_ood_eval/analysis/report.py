"""Generate a Markdown report without fabricating unrun results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "N/A (no matching results)"
    if percent:
        return f"{float(value) * 100:.2f}%"
    return f"{float(value):.3f}" if isinstance(value, float) else str(value)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "No results were available."
    header = "| " + " | ".join(title for _, title in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append(_fmt(value, percent=key in {"success_rate", "success_ci95_low", "success_ci95_high"}))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def generate_report(experiment_dir: Path, metrics: dict[str, Any] | None = None) -> Path:
    summary_dir = experiment_dir / "summary"
    if metrics is None:
        path = summary_dir / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Aggregate results first; missing {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    config = manifest.get("config", {})
    clean = metrics["clean"]
    ood = metrics["ood"]
    paired = metrics["paired"]
    report = f"""# Fast-WAM OOD Evaluation Report

> This report contains only records found on disk. `N/A` means that the corresponding experiment has not been run or aggregated.

## 1. Experiment setup

- Experiment: `{manifest.get('experiment_id', experiment_dir.name)}`
- Checkpoint: `{config.get('checkpoint', {}).get('path', 'unknown')}`
- Upstream commits: `{json.dumps(manifest.get('provenance', {}), ensure_ascii=False)}`
- Task suite: `{config.get('benchmark', {}).get('suite', 'unknown')}`
- Episodes recorded: `{metrics['all']['episodes']}`
- Seeds: deterministic per `(base seed, suite, task, episode index)`
- GPU environment: `{json.dumps(manifest.get('gpu_environment', {}), ensure_ascii=False)}`

## 2. Clean baseline

- Success rate: **{_fmt(clean['success_rate'], percent=True)}**
- 95% bootstrap CI: [{_fmt(clean['success_ci95_low'], percent=True)}, {_fmt(clean['success_ci95_high'], percent=True)}]
- Attempted / exceptions / skipped: {clean['attempted']} / {clean['exceptions']} / {clean['skipped']}

## 3. OOD results

- Success rate: **{_fmt(ood['success_rate'], percent=True)}**
- 95% bootstrap CI: [{_fmt(ood['success_ci95_low'], percent=True)}, {_fmt(ood['success_ci95_high'], percent=True)}]
- Absolute success drop: **{_fmt(metrics['absolute_success_drop'], percent=True)}**
- Relative success drop: **{_fmt(metrics['relative_success_drop'], percent=True)}**

## 4. Robustness drop by perturbation

{_table(metrics['by_perturbation'], [('perturbation_category', 'Perturbation'), ('attempted', 'N'), ('success_rate', 'Success rate'), ('success_ci95_low', 'CI low'), ('success_ci95_high', 'CI high')])}

## 5. Robustness drop by difficulty

{_table(metrics['by_level'], [('perturbation_level', 'Level'), ('attempted', 'N'), ('success_rate', 'Success rate'), ('success_ci95_low', 'CI low'), ('success_ci95_high', 'CI high')])}

## 6. Task-level results

{_table(metrics['by_task'], [('task_name', 'Task'), ('condition', 'Condition'), ('attempted', 'N'), ('success_rate', 'Success rate')])}

## 7. Latency and memory

- Mean inference latency: {_fmt(metrics['all']['mean_inference_latency_ms'])} ms
- P50 / P95 inference latency: {_fmt(metrics['all']['p50_inference_latency_ms'])} / {_fmt(metrics['all']['p95_inference_latency_ms'])} ms
- Maximum recorded GPU peak memory: {_fmt(metrics['all']['gpu_peak_memory_mb'])} MB
- Mean episode steps: {_fmt(metrics['all']['mean_steps'])}

## 8. Failure cases

- Failures: {metrics['all']['failures']}
- Exceptions: {metrics['all']['exceptions']}
- Skipped incompatible jobs: {metrics['all']['skipped']}
- Paired outcomes: clean-success/OOD-failure={paired['clean_success_ood_failure']}, clean-failure/OOD-success={paired['clean_failure_ood_success']}, both-success={paired['both_success']}, both-failure={paired['both_failure']}.
- Use `fastwam-ood review-failures --experiment-dir {experiment_dir}` for manual taxonomy labels.

## 9. Limitations

- Simulator OOD results do not establish real-robot OOD performance.
- Unpaired or skipped variants weaken causal comparisons; paired-seed counts are shown separately.
- LIBERO-Plus difficulty labels are upstream labels mapped as easy=1–2, medium=3, hard=4–5.
- Manual failure labels are descriptive and are not an automated visual diagnosis.

## 10. Conclusion

This experiment can quantify which tested perturbations Fast-WAM is sensitive to and the measured gap between standard and OOD environments. It cannot establish that explicit future imagination would fix OOD failures, that Fast-WAM lacks world modelling, that every WAM should omit future imagination, or that simulated OOD results equal real-world OOD results.
"""
    output = summary_dir / "report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output

