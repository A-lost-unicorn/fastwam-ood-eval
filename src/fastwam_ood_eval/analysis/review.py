"""Create a standalone, backend-free failure review page."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from fastwam_ood_eval.analysis.aggregate import load_results
from fastwam_ood_eval.analysis.failure_taxonomy import FAILURE_CATEGORIES


def generate_failure_review(experiment_dir: Path) -> Path:
    failures = [
        row
        for row in load_results(experiment_dir)
        if not row.get("success") and row.get("termination_reason") != "skipped"
    ]
    review_dir = experiment_dir / "failure_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    annotations = {
        row["job_id"]: {"failure_category": row.get("failure_category") or "unknown", "notes": row.get("failure_notes") or ""}
        for row in failures
    }
    (review_dir / "annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    cards: list[str] = []
    options = "".join(f'<option value="{html.escape(item)}">{html.escape(item)}</option>' for item in FAILURE_CATEGORIES)
    for row in failures:
        job_id = html.escape(str(row["job_id"]))
        video = row.get("video_path")
        if video:
            try:
                relative = Path(video).resolve().relative_to(review_dir.resolve())
                video_src = str(relative)
            except ValueError:
                video_src = str(Path("..") / Path(video).relative_to(experiment_dir)) if str(video).startswith(str(experiment_dir)) else str(video)
            media = f'<video controls preload="metadata" src="{html.escape(video_src)}"></video>'
        else:
            media = '<p class="missing">No failure video was recorded.</p>'
        cards.append(
            f'''<article data-job="{job_id}"><h2>{html.escape(str(row.get("task_name")))}</h2>
<p>perturbation={html.escape(str(row.get("perturbation_category")))} · level={html.escape(str(row.get("perturbation_level")))} · seed={row.get("episode_seed")} · termination={html.escape(str(row.get("termination_reason")))}</p>
{media}<label>Failure category<select>{options}</select></label><label>Notes<textarea></textarea></label></article>'''
        )
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>Fast-WAM failure review</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1100px}}article{{border:1px solid #ccc;padding:1rem;margin:1rem 0;border-radius:8px}}video{{max-width:100%;max-height:520px}}label{{display:block;margin-top:.8rem}}select,textarea{{display:block;width:100%;padding:.5rem}}textarea{{min-height:5rem}}button{{padding:.7rem 1rem}}</style></head><body>
<h1>Fast-WAM failure review</h1><p>Annotations are kept in browser localStorage. Use Export JSON to save a file that can replace <code>annotations.json</code>.</p>
<button id="export">Export JSON</button>{''.join(cards)}
<script>const initial={json.dumps(annotations, ensure_ascii=False)}; const key='fastwam-review-{html.escape(experiment_dir.name)}'; let data=JSON.parse(localStorage.getItem(key)||JSON.stringify(initial));
document.querySelectorAll('article').forEach(a=>{{let id=a.dataset.job,s=a.querySelector('select'),t=a.querySelector('textarea');s.value=(data[id]||{{}}).failure_category||'unknown';t.value=(data[id]||{{}}).notes||'';function save(){{data[id]={{failure_category:s.value,notes:t.value}};localStorage.setItem(key,JSON.stringify(data));}}s.onchange=save;t.oninput=save;}});
document.querySelector('#export').onclick=()=>{{let b=new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='annotations.json';a.click();URL.revokeObjectURL(u);}};</script></body></html>'''
    output = review_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output

