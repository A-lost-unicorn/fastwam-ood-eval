"""Label-blind review packets for Thought 2 future-consistency videos.

The public packet contains opaque case aliases, task text, and copied media.
Condition, perturbation, outcome, action, metric, seed, and source identifiers
live only in a separate private key directory.  Selection and ordering use
stable diagnostic identity hashes and never read outcome values.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fastwam_ood_eval.diagnostics.aggregate import load_diagnostics


LEGACY_BLIND_REVIEW_SCHEMA = "thought2-future-label-blind-review-v1"
BLIND_REVIEW_SCHEMA = "thought2-future-label-blind-review-v2"
PUBLIC_MANIFEST_NAME = "blind_packet_manifest.json"
PRIVATE_KEY_NAME = "unblinding_key.json"

MEDIA_FIELDS = {
    "current_frame": ("current_frame_path", "current.png"),
    "predicted_future": ("predicted_video_path", "predicted.mp4"),
    "actual_future": ("actual_video_path", "actual.mp4"),
    "aligned_comparison": (
        "side_by_side_video_path",
        "comparison.mp4",
    ),
}

BLIND_ANNOTATION_OPTIONS: dict[str, tuple[str, ...]] = {
    "video_validity": ("valid", "corrupt", "unaligned", "uncertain"),
    "future_goal_progress": (
        "correct",
        "partial",
        "wrong_object",
        "wrong_direction",
        "static",
        "uncertain",
    ),
    "future_physical_plausibility": (
        "plausible",
        "minor_artifact",
        "unphysical",
        "uncertain",
    ),
    "future_actual_agreement": (
        "aligned",
        "partial",
        "conflict",
        "static",
        "uncertain",
    ),
    "action_execution_quality": (
        "realized",
        "stalled",
        "collision",
        "oscillation",
        "uncertain",
    ),
    "confidence": ("low", "medium", "high"),
}

LEGACY_BLIND_ANNOTATION_FIELDS = (
    "case_id",
    "reviewer",
    "review_round",
    *BLIND_ANNOTATION_OPTIONS,
    "notes",
)

BLIND_ANNOTATION_FIELDS = (
    "packet_id",
    *LEGACY_BLIND_ANNOTATION_FIELDS,
)

SENSITIVE_PUBLIC_KEYS = {
    "source",
    "source_dir",
    "source_root",
    "source_record",
    "source_experiment_id",
    "experiment_id",
    "diagnostic_id",
    "probe_id",
    "job_id",
    "task_id",
    "episode_index",
    "episode_seed",
    "initial_state_index",
    "condition",
    "perturbation_category",
    "perturbation_level",
    "perturbation_parameters",
    "success",
    "episode_success",
    "termination_reason",
    "metrics",
    "static_future_flag",
    "action_hash",
    "predicted_actions",
    "executed_actions",
    "replan_index",
    "environment_step",
    "worker_rank",
    "checkpoint",
    "checkpoint_hash",
    "fastwam_commit",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _overlaps(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def _preflight_outputs(
    *,
    packet_dir: Path,
    key_dir: Path,
    input_dirs: Sequence[Path],
) -> None:
    if _overlaps(packet_dir, key_dir):
        raise ValueError(
            "Blind packet and private key directories must be disjoint"
        )
    for source in input_dirs:
        if _overlaps(packet_dir, source) or _overlaps(key_dir, source):
            raise ValueError(
                "Blind-review outputs must be disjoint from every diagnostic "
                f"source: source={source}"
            )
    for target in (packet_dir, key_dir):
        if target.exists():
            raise FileExistsError(
                f"Blind-review output already exists; use a fresh path: {target}"
            )


def _source_inventory(input_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    inventories: list[dict[str, Any]] = []
    for root in input_dirs:
        manifest_path = root / "diagnostic_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Diagnostic manifest does not exist: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid diagnostic manifest: {manifest_path}"
            ) from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("kind") != "future_shadow_diagnostics"
        ):
            raise RuntimeError(
                f"Not a future diagnostic manifest: {manifest_path}"
            )
        source_snapshot = root / "source_manifest.json"
        files = sorted(root.glob("workers/rank_*/diagnostics.jsonl"))
        inventories.append(
            {
                "root": str(root),
                "diagnostic_manifest_path": str(manifest_path),
                "diagnostic_manifest_sha256": _sha256(manifest_path),
                "diagnostic_protocol_fingerprint": manifest.get(
                    "protocol_fingerprint"
                ),
                "diagnostic_experiment_id": manifest.get("experiment_id"),
                "source_snapshot_path": (
                    str(source_snapshot) if source_snapshot.is_file() else None
                ),
                "source_snapshot_sha256": (
                    _sha256(source_snapshot)
                    if source_snapshot.is_file()
                    else None
                ),
                "diagnostic_files": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in files
                ],
            }
        )
    return inventories


def _identity(row: Mapping[str, Any]) -> str:
    return "\x1f".join(
        (
            str(row.get("_diagnostic_root", "")),
            str(row.get("experiment_id", "")),
            str(row.get("diagnostic_id", row.get("probe_id", ""))),
            str(
                (row.get("extra") or {}).get("protocol_fingerprint", "")
                if isinstance(row.get("extra"), Mapping)
                else ""
            ),
        )
    )


def _selection_key(row: Mapping[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"{int(seed)}\x1f{_identity(row)}".encode("utf-8")
    ).hexdigest()


def _source_job_identity(row: Mapping[str, Any]) -> str:
    job_id = row.get("job_id")
    if job_id in (None, ""):
        raise RuntimeError(
            "A per-job blind-review cap requires every diagnostic row to "
            "contain job_id"
        )
    return "\x1f".join(
        (
            str(row.get("_diagnostic_root", "")),
            str(row.get("experiment_id", "")),
            str(job_id),
        )
    )


def _source_job_selection_key(
    source_job_identity: str,
    seed: int,
) -> str:
    return hashlib.sha256(
        (
            f"{int(seed)}\x1fsource-job\x1f"
            f"{source_job_identity}"
        ).encode("utf-8")
    ).hexdigest()


def _artifact_path(
    row: Mapping[str, Any],
    artifact_field: str,
) -> Path:
    artifacts = (
        row.get("artifact_paths")
        if isinstance(row.get("artifact_paths"), Mapping)
        else {}
    )
    value = artifacts.get(artifact_field, row.get(artifact_field))
    if value in (None, ""):
        raise RuntimeError(
            f"Reviewable diagnostic lacks {artifact_field}: "
            f"{row.get('diagnostic_id')}"
        )
    root = Path(str(row.get("_diagnostic_root", ""))).resolve()
    candidate = Path(str(value))
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    if root != path and root not in path.parents:
        raise RuntimeError(
            f"Diagnostic artifact escapes its source root: {path}"
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Diagnostic artifact is missing or empty: {path}"
        )
    return path


def _task_instruction(row: Mapping[str, Any]) -> str:
    value = row.get("task_description") or row.get("task_name")
    if value in (None, ""):
        raise RuntimeError(
            f"Diagnostic lacks task instruction: {row.get('diagnostic_id')}"
        )
    return " ".join(str(value).replace("_", " ").split())


def _annotation_fields_for_schema(schema: str) -> tuple[str, ...]:
    if schema == BLIND_REVIEW_SCHEMA:
        return BLIND_ANNOTATION_FIELDS
    if schema == LEGACY_BLIND_REVIEW_SCHEMA:
        return LEGACY_BLIND_ANNOTATION_FIELDS
    raise RuntimeError(f"Unsupported blind-review schema: {schema!r}")


def _annotation_csv(
    packet_id: str,
    case_ids: Sequence[str],
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=BLIND_ANNOTATION_FIELDS)
    writer.writeheader()
    for case_id in case_ids:
        row = {field: "" for field in BLIND_ANNOTATION_FIELDS}
        row["packet_id"] = packet_id
        row["case_id"] = case_id
        row["review_round"] = "blind"
        writer.writerow(row)
    return buffer.getvalue()


def _review_html(public_manifest: Mapping[str, Any]) -> str:
    cases = public_manifest.get("cases", [])
    options = {
        field: list(values)
        for field, values in BLIND_ANNOTATION_OPTIONS.items()
    }
    cases_json = json.dumps(cases, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    options_json = json.dumps(options, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    packet_id = html.escape(str(public_manifest.get("packet_id", "")))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Future blind review</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem auto;max-width:1180px;padding:0 1rem}}
header{{margin-bottom:1.5rem}} article{{border:1px solid #bbb;border-radius:10px;padding:1rem;margin:1rem 0}}
.media{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}} img,video{{width:100%;max-height:440px;object-fit:contain;background:#111}}
.fields{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.75rem;margin-top:1rem}}
label{{display:block}} select,input,textarea{{box-sizing:border-box;width:100%;padding:.45rem;margin-top:.25rem}}
textarea{{min-height:5rem}} button{{padding:.65rem .9rem;margin:.3rem}} .hint{{color:#555}}
</style>
</head>
<body>
<header>
<h1>Future 一致性盲审</h1>
<p>只依据任务文本与媒体作判断。左侧为预测，右侧为动作执行后的对齐画面。</p>
<label>Reviewer ID <input id="reviewer" autocomplete="off"></label>
<button id="export-json">导出 JSON</button>
<button id="export-csv">导出 CSV</button>
<span class="hint">草稿仅保存在当前浏览器。</span>
</header>
<main id="cases"></main>
<script>
const packetId={json.dumps(packet_id)};
const cases={cases_json};
const options={options_json};
const fields=Object.keys(options);
const storageKey='future-blind-review-'+packetId;
let saved=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
const esc=v=>String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function persist(){{localStorage.setItem(storageKey,JSON.stringify(saved));}}
function render(){{
 const root=document.querySelector('#cases');
 cases.forEach(item=>{{
  const article=document.createElement('article'); article.dataset.caseId=item.case_id;
  article.innerHTML=`<h2>${{esc(item.case_id)}}</h2><p>${{esc(item.task_instruction)}}</p>
  <div class="media"><div><h3>当前帧</h3><img src="${{esc(item.media.current_frame.path)}}"></div>
  <div><h3>完整预测未来</h3><video controls preload="metadata" src="${{esc(item.media.predicted_future.path)}}"></video></div>
  <div><h3>动作执行后实际画面</h3><video controls preload="metadata" src="${{esc(item.media.actual_future.path)}}"></video></div>
  <div><h3>对齐比较：左预测 / 右实际</h3><video controls preload="metadata" src="${{esc(item.media.aligned_comparison.path)}}"></video></div></div>
  <div class="fields"></div><label>Notes<textarea data-field="notes"></textarea></label>`;
  const area=article.querySelector('.fields');
  fields.forEach(field=>{{
   const label=document.createElement('label'); label.textContent=field;
   const select=document.createElement('select'); select.dataset.field=field;
   select.innerHTML='<option value=""></option>'+options[field].map(v=>`<option value="${{esc(v)}}">${{esc(v)}}</option>`).join('');
   label.appendChild(select); area.appendChild(label);
  }});
  const current=saved[item.case_id]||{{}};
  article.querySelectorAll('[data-field]').forEach(control=>{{
   control.value=current[control.dataset.field]||'';
   control.addEventListener('input',()=>{{saved[item.case_id]={{...(saved[item.case_id]||{{}}),[control.dataset.field]:control.value}};persist();}});
  }});
  root.appendChild(article);
 }});
}}
function rows(){{
 const reviewer=document.querySelector('#reviewer').value.trim();
 return cases.map(item=>({{packet_id:packetId,case_id:item.case_id,reviewer,review_round:'blind',...(saved[item.case_id]||{{}})}}));
}}
function download(name,text,type){{
 const url=URL.createObjectURL(new Blob([text],{{type}})); const link=document.createElement('a');
 link.href=url;link.download=name;link.click();URL.revokeObjectURL(url);
}}
document.querySelector('#export-json').onclick=()=>download('blind_annotations.json',JSON.stringify({{packet_id:packetId,annotations:rows()}},null,2),'application/json');
document.querySelector('#export-csv').onclick=()=>{{
 const columns=['packet_id','case_id','reviewer','review_round',...fields,'notes'];
 const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 download('blind_annotations.csv',[columns.map(quote).join(','),...rows().map(row=>columns.map(c=>quote(row[c])).join(','))].join('\\n'),'text/csv');
}};
render();
</script>
</body>
</html>
"""


def _walk_public_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_public_keys(item)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            yield from _walk_public_keys(item)


def prepare_blind_review(
    *,
    packet_dir: Path,
    key_dir: Path,
    input_dirs: Sequence[Path],
    seed: int,
    max_cases: int | None = None,
    max_cases_per_job: int | None = None,
) -> dict[str, Any]:
    """Build a fresh public packet and separate private unblinding key."""

    packet_dir = Path(packet_dir)
    key_dir = Path(key_dir)
    inputs: list[Path] = []
    seen: set[Path] = set()
    for value in input_dirs:
        path = Path(value)
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            inputs.append(path)
    if not inputs:
        raise ValueError("At least one diagnostic input directory is required")
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be positive when provided")
    if max_cases_per_job is not None and max_cases_per_job <= 0:
        raise ValueError(
            "max_cases_per_job must be positive when provided"
        )
    _preflight_outputs(
        packet_dir=packet_dir,
        key_dir=key_dir,
        input_dirs=inputs,
    )

    inventories = _source_inventory(inputs)
    rows = load_diagnostics(inputs[0], inputs[1:])
    reviewable = [
        row
        for row in rows
        if row.get("status") == "completed" and not row.get("error")
    ]
    nonreviewable = [
        {
            "diagnostic_id": row.get("diagnostic_id", row.get("probe_id")),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for row in rows
        if row not in reviewable
    ]
    if not reviewable:
        raise RuntimeError("No completed diagnostic rows are reviewable")
    ordered = sorted(
        reviewable,
        key=lambda row: (_selection_key(row, seed), _identity(row)),
    )
    reviewable_before_job_cap = len(ordered)
    if max_cases_per_job is not None:
        by_source_job: dict[str, list[dict[str, Any]]] = {}
        for row in ordered:
            source_job = _source_job_identity(row)
            by_source_job.setdefault(source_job, []).append(row)
        source_jobs = sorted(
            by_source_job,
            key=lambda value: (
                _source_job_selection_key(value, seed),
                value,
            ),
        )
        capped: list[dict[str, Any]] = []
        # Round-robin probe ranks so a global max_cases limit never gives
        # episodes with more probes a larger inclusion probability.
        for probe_rank in range(max_cases_per_job):
            for source_job in source_jobs:
                probes = by_source_job[source_job]
                if probe_rank < len(probes):
                    capped.append(probes[probe_rank])
        ordered = capped
    eligible_after_job_cap = len(ordered)
    if max_cases is not None:
        ordered = ordered[:max_cases]

    protocol_payload = {
        "schema": BLIND_REVIEW_SCHEMA,
        "seed": int(seed),
        "max_cases": max_cases,
        "max_cases_per_job": max_cases_per_job,
        "input_hashes": [
            {
                "diagnostic_manifest_sha256": item[
                    "diagnostic_manifest_sha256"
                ],
                "diagnostic_files": [
                    file["sha256"] for file in item["diagnostic_files"]
                ],
            }
            for item in inventories
        ],
        "selected_identities": [_identity(row) for row in ordered],
        "implementation_sha256": _sha256(Path(__file__)),
    }
    packet_id = _canonical_sha256(protocol_payload)[:24]

    packet_dir.parent.mkdir(parents=True, exist_ok=True)
    key_dir.parent.mkdir(parents=True, exist_ok=True)
    packet_temp = Path(
        tempfile.mkdtemp(
            prefix=f".{packet_dir.name}.",
            dir=packet_dir.parent,
        )
    )
    key_temp = Path(
        tempfile.mkdtemp(
            prefix=f".{key_dir.name}.",
            dir=key_dir.parent,
        )
    )
    published_key = False
    try:
        public_cases: list[dict[str, Any]] = []
        private_cases: list[dict[str, Any]] = []
        for index, row in enumerate(ordered, start=1):
            case_id = f"case_{index:04d}"
            public_media: dict[str, dict[str, Any]] = {}
            private_media: dict[str, dict[str, Any]] = {}
            for public_name, (source_field, filename) in MEDIA_FIELDS.items():
                source_path = _artifact_path(row, source_field)
                relative = Path("media") / case_id / filename
                target = packet_temp / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
                source_hash = _sha256(source_path)
                copied_hash = _sha256(target)
                if source_hash != copied_hash:
                    raise RuntimeError(
                        f"Media copy hash mismatch: {source_path}"
                    )
                public_media[public_name] = {
                    "path": relative.as_posix(),
                    "sha256": copied_hash,
                    "bytes": target.stat().st_size,
                }
                private_media[public_name] = {
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "public_path": relative.as_posix(),
                }
            public_cases.append(
                {
                    "case_id": case_id,
                    "task_instruction": _task_instruction(row),
                    "media": public_media,
                }
            )
            private_cases.append(
                {
                    "case_id": case_id,
                    "selection_key": _selection_key(row, seed),
                    "source_identity": _identity(row),
                    "source_media": private_media,
                    "source_record": {
                        key: value
                        for key, value in row.items()
                        if key != "_diagnostic_root"
                    },
                    "source_root": row.get("_diagnostic_root"),
                }
            )

        public_manifest = {
            "schema_version": 1,
            "kind": "future_label_blind_review_packet",
            "schema": BLIND_REVIEW_SCHEMA,
            "packet_id": packet_id,
            "blinding_level": "group_outcome_metric_identifier_blind",
            "case_count": len(public_cases),
            "case_order": [case["case_id"] for case in public_cases],
            "cases": public_cases,
            "annotation_fields": list(BLIND_ANNOTATION_FIELDS),
            "annotation_options": {
                key: list(values)
                for key, values in BLIND_ANNOTATION_OPTIONS.items()
            },
            "review_instructions": (
                "Judge only the task text and media. The aligned comparison "
                "shows predicted frames on the left and realized frames on the right."
            ),
        }
        _atomic_json(packet_temp / PUBLIC_MANIFEST_NAME, public_manifest)
        _atomic_text(
            packet_temp / "annotations.csv",
            _annotation_csv(
                packet_id,
                public_manifest["case_order"],
            ),
        )
        _atomic_text(packet_temp / "index.html", _review_html(public_manifest))
        public_manifest_hash = _sha256(
            packet_temp / PUBLIC_MANIFEST_NAME
        )
        private_key = {
            "schema_version": 1,
            "kind": "future_label_blind_review_private_key",
            "schema": BLIND_REVIEW_SCHEMA,
            "packet_id": packet_id,
            "public_manifest_sha256": public_manifest_hash,
            "public_index_sha256": _sha256(packet_temp / "index.html"),
            "selection": {
                "seed": int(seed),
                "max_cases": max_cases,
                "max_cases_per_job": max_cases_per_job,
                "candidate_rows": len(rows),
                "reviewable_rows": reviewable_before_job_cap,
                "eligible_rows_after_job_cap": (
                    eligible_after_job_cap
                ),
                "selected_rows": len(ordered),
                "variables_used": [
                    "diagnostic_identity",
                    "status_completed",
                    "artifact_availability",
                    *(
                        ["source_job_identity"]
                        if max_cases_per_job is not None
                        else []
                    ),
                ],
                "outcome_fields_used": False,
                "condition_fields_used": False,
                "metric_fields_used": False,
                "nonreviewable_rows": nonreviewable,
            },
            "source_inventories": inventories,
            "cases": private_cases,
            "protocol_payload": protocol_payload,
        }
        _atomic_json(key_temp / PRIVATE_KEY_NAME, private_key)
        os.chmod(key_temp / PRIVATE_KEY_NAME, 0o600)

        validate_blind_review_packet(
            packet_temp,
            key_temp,
        )
        os.chmod(key_temp, 0o700)
        os.replace(key_temp, key_dir)
        published_key = True
        os.replace(packet_temp, packet_dir)
    except Exception:
        if published_key and not packet_dir.exists():
            # Keep the private key: it is safer and recoverable, while deleting
            # it could make an already shared packet impossible to unblind.
            pass
        raise
    finally:
        shutil.rmtree(packet_temp, ignore_errors=True)
        shutil.rmtree(key_temp, ignore_errors=True)

    return validate_blind_review_packet(packet_dir, key_dir)


def validate_blind_review_packet(
    packet_dir: Path,
    key_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate public leakage boundaries, media hashes, and optional key."""

    packet_dir = Path(packet_dir)
    manifest_path = packet_dir / PUBLIC_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Blind packet manifest does not exist: {manifest_path}"
        )
    try:
        public = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid blind packet manifest: {manifest_path}"
        ) from exc
    if (
        not isinstance(public, Mapping)
        or public.get("kind") != "future_label_blind_review_packet"
        or public.get("schema")
        not in {BLIND_REVIEW_SCHEMA, LEGACY_BLIND_REVIEW_SCHEMA}
    ):
        raise RuntimeError(f"Invalid blind packet identity: {manifest_path}")
    packet_schema = str(public["schema"])
    leaked_keys = sorted(
        set(_walk_public_keys(public)) & SENSITIVE_PUBLIC_KEYS
    )
    if leaked_keys:
        raise RuntimeError(
            f"Public blind packet contains sensitive keys: {leaked_keys}"
        )
    expected_annotation_fields = _annotation_fields_for_schema(
        packet_schema
    )
    if tuple(public.get("annotation_fields", ())) != (
        expected_annotation_fields
    ):
        raise RuntimeError(
            "Blind packet annotation fields differ from protocol"
        )
    expected_options = {
        field: list(values)
        for field, values in BLIND_ANNOTATION_OPTIONS.items()
    }
    if public.get("annotation_options") != expected_options:
        raise RuntimeError(
            "Blind packet annotation options differ from protocol"
        )

    cases = public.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("Blind packet must contain at least one case")
    case_ids = [str(case.get("case_id")) for case in cases]
    if (
        len(set(case_ids)) != len(case_ids)
        or case_ids != public.get("case_order")
        or len(cases) != int(public.get("case_count", -1))
    ):
        raise RuntimeError("Blind packet case identities/order are inconsistent")
    media_count = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise RuntimeError("Blind packet case must be an object")
        media = case.get("media")
        if not isinstance(media, Mapping) or set(media) != set(MEDIA_FIELDS):
            raise RuntimeError(
                f"Blind case has incomplete media: {case.get('case_id')}"
            )
        for item in media.values():
            if not isinstance(item, Mapping):
                raise RuntimeError("Blind media entry must be an object")
            relative = Path(str(item.get("path", "")))
            path = (packet_dir / relative).resolve()
            packet_root = packet_dir.resolve()
            if packet_root != path and packet_root not in path.parents:
                raise RuntimeError(f"Blind media escapes packet: {relative}")
            if (
                not path.is_file()
                or path.stat().st_size != int(item.get("bytes", -1))
                or _sha256(path) != item.get("sha256")
            ):
                raise RuntimeError(f"Blind media hash/size mismatch: {path}")
            media_count += 1

    annotation_path = packet_dir / "annotations.csv"
    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != expected_annotation_fields:
            raise RuntimeError("Blind annotation CSV schema differs from protocol")
    if [row.get("case_id") for row in rows] != case_ids:
        raise RuntimeError("Blind annotation CSV case order differs from manifest")
    if (
        packet_schema == BLIND_REVIEW_SCHEMA
        and any(row.get("packet_id") != public.get("packet_id") for row in rows)
    ):
        raise RuntimeError(
            "Blind annotation CSV packet_id differs from manifest"
        )
    if not (packet_dir / "index.html").is_file():
        raise FileNotFoundError("Blind review index.html is missing")

    key_verified = False
    if key_dir is not None:
        key_path = Path(key_dir) / PRIVATE_KEY_NAME
        if not key_path.is_file():
            raise FileNotFoundError(
                f"Private unblinding key does not exist: {key_path}"
            )
        try:
            private = json.loads(key_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid private key: {key_path}") from exc
        if (
            not isinstance(private, Mapping)
            or private.get("kind")
            != "future_label_blind_review_private_key"
            or private.get("schema") != packet_schema
            or private.get("packet_id") != public.get("packet_id")
            or private.get("public_manifest_sha256")
            != _sha256(manifest_path)
        ):
            raise RuntimeError("Private unblinding key does not match packet")
        private_cases = private.get("cases")
        if (
            not isinstance(private_cases, list)
            or [str(case.get("case_id")) for case in private_cases] != case_ids
        ):
            raise RuntimeError("Private case mapping differs from public packet")

        sensitive_tokens: set[str] = set()
        for inventory in private.get("source_inventories", []):
            if isinstance(inventory, Mapping):
                for name in ("root", "diagnostic_experiment_id"):
                    token = inventory.get(name)
                    if isinstance(token, str) and len(token) >= 8:
                        sensitive_tokens.add(token)
                        sensitive_tokens.add(Path(token).name)
        for case in private_cases:
            if not isinstance(case, Mapping):
                continue
            record = case.get("source_record")
            if isinstance(record, Mapping):
                for name in (
                    "job_id",
                    "diagnostic_id",
                    "probe_id",
                    "experiment_id",
                    "source_experiment_id",
                ):
                    token = record.get(name)
                    if isinstance(token, str) and len(token) >= 8:
                        sensitive_tokens.add(token)
        public_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (
                manifest_path,
                packet_dir / "annotations.csv",
                packet_dir / "index.html",
            )
        )
        leaked_tokens = sorted(
            token for token in sensitive_tokens if token in public_text
        )
        if leaked_tokens:
            raise RuntimeError(
                "Public blind packet exposes private source identifiers: "
                f"{leaked_tokens[:5]}"
            )
        key_verified = True

    return {
        "packet_id": public.get("packet_id"),
        "packet_schema": packet_schema,
        "annotation_packet_binding": (
            "row_packet_id"
            if packet_schema == BLIND_REVIEW_SCHEMA
            else "legacy_case_id_only"
        ),
        "cases": len(cases),
        "media_files": media_count,
        "public_manifest_sha256": _sha256(manifest_path),
        "private_key_verified": key_verified,
        "sensitive_public_keys": 0,
    }


__all__ = [
    "BLIND_ANNOTATION_FIELDS",
    "BLIND_ANNOTATION_OPTIONS",
    "BLIND_REVIEW_SCHEMA",
    "LEGACY_BLIND_ANNOTATION_FIELDS",
    "LEGACY_BLIND_REVIEW_SCHEMA",
    "PRIVATE_KEY_NAME",
    "PUBLIC_MANIFEST_NAME",
    "prepare_blind_review",
    "validate_blind_review_packet",
]
