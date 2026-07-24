"""Validation and inter-reviewer agreement for label-blind annotations.

This module never opens the private unblinding key.  It validates independent
reviewer exports against the public packet, preserves raw-file hashes, and
reports both all-nonmissing and decisive-only (non-``uncertain``) agreement.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.diagnostics.blind_review import (
    BLIND_ANNOTATION_FIELDS,
    BLIND_ANNOTATION_OPTIONS,
    BLIND_REVIEW_SCHEMA,
    LEGACY_BLIND_ANNOTATION_FIELDS,
    LEGACY_BLIND_REVIEW_SCHEMA,
    PUBLIC_MANIFEST_NAME,
    validate_blind_review_packet,
)


BLIND_ANALYSIS_SCHEMA = "thought2-future-blind-review-analysis-v1"
AGREEMENT_FIELDS = tuple(BLIND_ANNOTATION_OPTIONS)
OUTPUT_FILES = (
    "normalized_annotations.csv",
    "reviewer_completion.csv",
    "pairwise_agreement.csv",
    "agreement_summary.json",
    "agreement_report.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _overlaps(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def _read_public_manifest(packet_dir: Path) -> dict[str, Any]:
    validate_blind_review_packet(packet_dir)
    path = packet_dir / PUBLIC_MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid public packet manifest: {path}")
    return payload


def _read_annotation_rows(
    path: Path,
    *,
    packet_id: str,
    expected_fields: Sequence[str],
) -> tuple[list[Mapping[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise RuntimeError(
                    f"Annotation CSV schema differs from packet: {path}"
                )
            return list(reader), "csv"
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid annotation JSON: {path}") from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("packet_id") != packet_id
            or not isinstance(payload.get("annotations"), list)
        ):
            raise RuntimeError(
                f"Annotation JSON packet identity/schema is invalid: {path}"
            )
        rows = payload["annotations"]
        if not all(isinstance(row, Mapping) for row in rows):
            raise RuntimeError(
                f"Annotation JSON rows must be objects: {path}"
            )
        return list(rows), "json"
    raise ValueError(
        f"Annotation input must be .csv or .json, got: {path}"
    )


def _normalize_annotation_file(
    path: Path,
    *,
    packet_id: str,
    packet_schema: str,
    case_ids: Sequence[str],
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")
    if packet_schema == BLIND_REVIEW_SCHEMA:
        source_fields = BLIND_ANNOTATION_FIELDS
    elif packet_schema == LEGACY_BLIND_REVIEW_SCHEMA:
        source_fields = LEGACY_BLIND_ANNOTATION_FIELDS
    else:
        raise RuntimeError(
            f"Unsupported blind packet schema: {packet_schema!r}"
        )
    rows, source_format = _read_annotation_rows(
        path,
        packet_id=packet_id,
        expected_fields=source_fields,
    )
    if (
        packet_schema == LEGACY_BLIND_REVIEW_SCHEMA
        and source_format == "csv"
    ):
        raise RuntimeError(
            "Legacy v1 annotation CSV has no packet_id binding; use its "
            "packet-bound JSON export or regenerate a v2 packet"
        )
    allowed_keys = set(source_fields)
    normalized: dict[str, dict[str, str]] = {}
    reviewers: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        unknown = set(str(key) for key in raw) - allowed_keys
        if unknown:
            raise RuntimeError(
                f"Unknown annotation fields in {path} row {index}: "
                f"{sorted(unknown)}"
            )
        row = {
            field: (
                ""
                if raw.get(field) is None
                else str(raw.get(field))
            )
            for field in BLIND_ANNOTATION_FIELDS
        }
        if packet_schema == LEGACY_BLIND_REVIEW_SCHEMA:
            row["packet_id"] = packet_id
        for field in BLIND_ANNOTATION_FIELDS:
            if field != "notes":
                row[field] = row[field].strip()
        case_id = row["case_id"]
        reviewer = row["reviewer"]
        if row["packet_id"] != packet_id:
            raise RuntimeError(
                f"packet_id differs from packet in {path} row {index}"
            )
        if not case_id:
            raise RuntimeError(
                f"Missing case_id in {path} row {index}"
            )
        if case_id in normalized:
            raise RuntimeError(
                f"Duplicate case_id {case_id!r} in {path}"
            )
        if not reviewer:
            raise RuntimeError(
                f"Missing reviewer in {path} row {index}"
            )
        if row["review_round"] != "blind":
            raise RuntimeError(
                f"review_round must be 'blind' in {path} row {index}"
            )
        for field, options in BLIND_ANNOTATION_OPTIONS.items():
            value = row[field]
            if value and value not in options:
                raise RuntimeError(
                    f"Invalid {field}={value!r} in {path} row {index}"
                )
        reviewers.add(reviewer)
        normalized[case_id] = row
    if len(reviewers) != 1:
        raise RuntimeError(
            f"Each annotation file must contain exactly one reviewer: {path}"
        )
    expected = set(case_ids)
    actual = set(normalized)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Annotation case coverage differs from packet in {path}: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    reviewer = next(iter(reviewers))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "format": source_format,
        "packet_binding": (
            "row_and_json_packet_id"
            if packet_schema == BLIND_REVIEW_SCHEMA
            and source_format == "json"
            else (
                "row_packet_id"
                if packet_schema == BLIND_REVIEW_SCHEMA
                else "legacy_json_top_level_packet_id"
            )
        ),
        "reviewer": reviewer,
        "rows": [normalized[case_id] for case_id in case_ids],
        "by_case": normalized,
    }


def _cohen_kappa(
    first: Sequence[str],
    second: Sequence[str],
) -> tuple[float | None, str]:
    if len(first) != len(second):
        raise ValueError("Kappa inputs must have equal length")
    count = len(first)
    if count == 0:
        return None, "no_eligible_pairs"
    observed = sum(
        left == right for left, right in zip(first, second)
    ) / count
    categories = sorted(set(first) | set(second))
    expected = 0.0
    for category in categories:
        first_rate = sum(value == category for value in first) / count
        second_rate = sum(value == category for value in second) / count
        expected += first_rate * second_rate
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return None, "degenerate_marginals"
    return (observed - expected) / denominator, "available"


def _agreement_rate(
    first: Sequence[str],
    second: Sequence[str],
) -> float | None:
    if not first:
        return None
    return sum(
        left == right for left, right in zip(first, second)
    ) / len(first)


def _pairwise_field_row(
    *,
    reviewer_a: str,
    reviewer_b: str,
    rows_a: Mapping[str, Mapping[str, str]],
    rows_b: Mapping[str, Mapping[str, str]],
    case_ids: Sequence[str],
    field: str,
) -> dict[str, Any]:
    nonmissing_a: list[str] = []
    nonmissing_b: list[str] = []
    decisive_a: list[str] = []
    decisive_b: list[str] = []
    missing_pairs = 0
    uncertain_pairs = 0
    for case_id in case_ids:
        first = rows_a[case_id][field]
        second = rows_b[case_id][field]
        if not first or not second:
            missing_pairs += 1
            continue
        nonmissing_a.append(first)
        nonmissing_b.append(second)
        if first == "uncertain" or second == "uncertain":
            uncertain_pairs += 1
            continue
        decisive_a.append(first)
        decisive_b.append(second)
    kappa_all, kappa_all_status = _cohen_kappa(
        nonmissing_a,
        nonmissing_b,
    )
    kappa_decisive, kappa_decisive_status = _cohen_kappa(
        decisive_a,
        decisive_b,
    )
    return {
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
        "field": field,
        "packet_cases": len(case_ids),
        "nonmissing_pairs": len(nonmissing_a),
        "missing_pairs": missing_pairs,
        "uncertain_pairs": uncertain_pairs,
        "decisive_pairs": len(decisive_a),
        "exact_agreements_nonmissing": sum(
            left == right
            for left, right in zip(nonmissing_a, nonmissing_b)
        ),
        "agreement_nonmissing": _agreement_rate(
            nonmissing_a,
            nonmissing_b,
        ),
        "cohen_kappa_nonmissing": kappa_all,
        "cohen_kappa_nonmissing_status": kappa_all_status,
        "exact_agreements_decisive": sum(
            left == right
            for left, right in zip(decisive_a, decisive_b)
        ),
        "agreement_decisive": _agreement_rate(
            decisive_a,
            decisive_b,
        ),
        "cohen_kappa_decisive": kappa_decisive,
        "cohen_kappa_decisive_status": kappa_decisive_status,
    }


def _completion_rows(
    annotations: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        reviewer = str(annotation["reviewer"])
        by_case = annotation["by_case"]
        for field in AGREEMENT_FIELDS:
            values = [by_case[case_id][field] for case_id in case_ids]
            annotated = sum(bool(value) for value in values)
            uncertain = sum(value == "uncertain" for value in values)
            rows.append(
                {
                    "reviewer": reviewer,
                    "field": field,
                    "packet_cases": len(case_ids),
                    "annotated": annotated,
                    "missing": len(case_ids) - annotated,
                    "uncertain": uncertain,
                    "decisive": annotated - uncertain,
                }
            )
    return rows


def _field_summaries(
    pairwise_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in AGREEMENT_FIELDS:
        rows = [row for row in pairwise_rows if row["field"] == field]
        kappa_all = [
            float(row["cohen_kappa_nonmissing"])
            for row in rows
            if row["cohen_kappa_nonmissing"] is not None
        ]
        kappa_decisive = [
            float(row["cohen_kappa_decisive"])
            for row in rows
            if row["cohen_kappa_decisive"] is not None
        ]
        nonmissing = sum(int(row["nonmissing_pairs"]) for row in rows)
        decisive = sum(int(row["decisive_pairs"]) for row in rows)
        agreements_all = sum(
            int(row["exact_agreements_nonmissing"]) for row in rows
        )
        agreements_decisive = sum(
            int(row["exact_agreements_decisive"]) for row in rows
        )
        result[field] = {
            "reviewer_pairs": len(rows),
            "pooled_nonmissing_pair_observations": nonmissing,
            "pooled_agreement_nonmissing": (
                agreements_all / nonmissing if nonmissing else None
            ),
            "pairwise_macro_cohen_kappa_nonmissing": (
                sum(kappa_all) / len(kappa_all) if kappa_all else None
            ),
            "available_kappa_nonmissing_pairs": len(kappa_all),
            "pooled_decisive_pair_observations": decisive,
            "pooled_agreement_decisive": (
                agreements_decisive / decisive if decisive else None
            ),
            "pairwise_macro_cohen_kappa_decisive": (
                sum(kappa_decisive) / len(kappa_decisive)
                if kappa_decisive
                else None
            ),
            "available_kappa_decisive_pairs": len(kappa_decisive),
        }
    return result


def _csv_text(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _format_metric(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.4f}"


def _report_text(
    summary: Mapping[str, Any],
    pairwise_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# 阶段二盲审一致性报告",
        "",
        f"- Packet ID：`{summary['packet_id']}`",
        f"- Cases：{summary['case_count']}",
        f"- Reviewers：{', '.join(summary['reviewers'])}",
        f"- Reviewer pairs：{summary['reviewer_pair_count']}",
        "- Private key read：false",
        "- Outcome/condition/metric fields read：false",
        "",
        "## Pairwise agreement",
        "",
        "| Reviewer pair | Field | Nonmissing n | Agreement | Cohen's κ | "
        "Decisive n | Decisive agreement | Decisive κ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise_rows:
        lines.append(
            "| "
            f"{row['reviewer_a']} / {row['reviewer_b']} | "
            f"{row['field']} | {row['nonmissing_pairs']} | "
            f"{_format_metric(row['agreement_nonmissing'])} | "
            f"{_format_metric(row['cohen_kappa_nonmissing'])} | "
            f"{row['decisive_pairs']} | "
            f"{_format_metric(row['agreement_decisive'])} | "
            f"{_format_metric(row['cohen_kappa_decisive'])} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- Nonmissing 口径把 `uncertain` 作为真实类别；decisive 口径排除任一 "
            "reviewer 为 missing/`uncertain` 的 pair。",
            "- κ 在边际分布退化时为 `undefined`，即使表面 agreement=1；不得把 "
            "`undefined` 改写为 1。",
            "- 多于两名 reviewer 时报告的是所有 reviewer pair 的结果与 macro "
            "均值，不是 Fleiss' κ。",
            "- 本报告仍处于 blind space，只衡量标注一致性，不能说明模型 "
            "future 是否正确，也不能替代解盲后的效应分析。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_blind_review(
    *,
    packet_dir: Path,
    annotation_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate reviewer exports and write a fresh blinded agreement analysis."""

    packet_dir = Path(packet_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Blind-review analysis output already exists: {output_dir}"
        )
    if _overlaps(packet_dir, output_dir):
        raise ValueError(
            "Blind-review analysis output must be disjoint from the packet"
        )
    paths = [Path(path) for path in annotation_paths]
    if len(paths) < 2:
        raise ValueError(
            "At least two independent annotation files are required"
        )
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Annotation file paths must be unique")

    public = _read_public_manifest(packet_dir)
    packet_id = str(public["packet_id"])
    packet_schema = str(public["schema"])
    case_ids = [str(value) for value in public["case_order"]]
    annotations = [
        _normalize_annotation_file(
            path,
            packet_id=packet_id,
            packet_schema=packet_schema,
            case_ids=case_ids,
        )
        for path in paths
    ]
    reviewers = [str(item["reviewer"]) for item in annotations]
    if len(set(reviewers)) != len(reviewers):
        raise RuntimeError(
            "Annotation files must use distinct reviewer IDs"
        )

    normalized_rows = [
        annotation["by_case"][case_id]
        for annotation in annotations
        for case_id in case_ids
    ]
    completion_rows = _completion_rows(annotations, case_ids)
    pairwise_rows = [
        _pairwise_field_row(
            reviewer_a=str(first["reviewer"]),
            reviewer_b=str(second["reviewer"]),
            rows_a=first["by_case"],
            rows_b=second["by_case"],
            case_ids=case_ids,
            field=field,
        )
        for first, second in itertools.combinations(annotations, 2)
        for field in AGREEMENT_FIELDS
    ]
    source_annotations = [
        {
            key: item[key]
            for key in (
                "path",
                "sha256",
                "format",
                "packet_binding",
                "reviewer",
            )
        }
        for item in annotations
    ]
    method = {
        "agreement_fields": list(AGREEMENT_FIELDS),
        "kappa": "unweighted_nominal_pairwise_cohen",
        "nonmissing_policy": (
            "exclude pairs with either value missing; retain uncertain as a category"
        ),
        "decisive_policy": (
            "exclude pairs with either value missing or equal to uncertain"
        ),
        "multi_reviewer_policy": (
            "all unordered reviewer pairs; macro mean over defined pairwise kappas"
        ),
        "notes_in_agreement": False,
    }
    identity_payload = {
        "schema": BLIND_ANALYSIS_SCHEMA,
        "packet_id": packet_id,
        "packet_schema": packet_schema,
        "public_manifest_sha256": _sha256(
            packet_dir / PUBLIC_MANIFEST_NAME
        ),
        "annotation_sha256": [
            item["sha256"] for item in source_annotations
        ],
        "reviewers": reviewers,
        "method": method,
        "implementation_sha256": _sha256(Path(__file__)),
    }
    analysis_id = _canonical_sha256(identity_payload)
    summary = {
        "schema_version": 1,
        "kind": "future_blind_review_agreement",
        "schema": BLIND_ANALYSIS_SCHEMA,
        "analysis_id": analysis_id,
        "packet_id": packet_id,
        "packet_schema": packet_schema,
        "case_count": len(case_ids),
        "reviewers": reviewers,
        "reviewer_pair_count": math.comb(len(reviewers), 2),
        "source_annotations": source_annotations,
        "method": method,
        "field_summaries": _field_summaries(pairwise_rows),
        "private_key_read": False,
        "outcome_fields_read": False,
        "condition_fields_read": False,
        "metric_fields_read": False,
        "causal_interpretation_allowed": False,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        (temporary / "normalized_annotations.csv").write_text(
            _csv_text(normalized_rows, BLIND_ANNOTATION_FIELDS),
            encoding="utf-8",
        )
        completion_fields = (
            "reviewer",
            "field",
            "packet_cases",
            "annotated",
            "missing",
            "uncertain",
            "decisive",
        )
        (temporary / "reviewer_completion.csv").write_text(
            _csv_text(completion_rows, completion_fields),
            encoding="utf-8",
        )
        pairwise_fields = tuple(pairwise_rows[0])
        (temporary / "pairwise_agreement.csv").write_text(
            _csv_text(pairwise_rows, pairwise_fields),
            encoding="utf-8",
        )
        (temporary / "agreement_summary.json").write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "agreement_report.md").write_text(
            _report_text(summary, pairwise_rows),
            encoding="utf-8",
        )
        output_hashes = {
            name: _sha256(temporary / name)
            for name in OUTPUT_FILES
        }
        manifest = {
            "schema_version": 1,
            "kind": "future_blind_review_analysis_manifest",
            "schema": BLIND_ANALYSIS_SCHEMA,
            "analysis_id": analysis_id,
            "identity_payload": identity_payload,
            "packet": {
                "directory": str(packet_dir.resolve()),
                "packet_id": packet_id,
                "packet_schema": packet_schema,
                "public_manifest_sha256": identity_payload[
                    "public_manifest_sha256"
                ],
            },
            "source_annotations": source_annotations,
            "outputs": output_hashes,
            "source_files_rewritten": False,
            "private_key_read": False,
            "outcome_fields_read": False,
            "condition_fields_read": False,
            "metric_fields_read": False,
        }
        (temporary / "analysis_manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return validate_blind_review_analysis(output_dir)


def validate_blind_review_analysis(
    output_dir: Path,
) -> dict[str, Any]:
    """Verify analysis identity, source hashes, and every derived artifact."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Blind-review analysis manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid blind-review analysis manifest: {manifest_path}"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("kind")
        != "future_blind_review_analysis_manifest"
        or manifest.get("schema") != BLIND_ANALYSIS_SCHEMA
    ):
        raise RuntimeError("Invalid blind-review analysis identity")
    identity = manifest.get("identity_payload")
    if (
        not isinstance(identity, Mapping)
        or manifest.get("analysis_id") != _canonical_sha256(identity)
    ):
        raise RuntimeError("Blind-review analysis ID is invalid")
    for field in (
        "source_files_rewritten",
        "private_key_read",
        "outcome_fields_read",
        "condition_fields_read",
        "metric_fields_read",
    ):
        if manifest.get(field) is not False:
            raise RuntimeError(
                f"Blind-review analysis safety flag is invalid: {field}"
            )
    packet = manifest.get("packet")
    if not isinstance(packet, Mapping):
        raise RuntimeError("Blind-review analysis packet metadata is invalid")
    packet_manifest = (
        Path(str(packet.get("directory", ""))) / PUBLIC_MANIFEST_NAME
    )
    if (
        not packet_manifest.is_file()
        or _sha256(packet_manifest)
        != packet.get("public_manifest_sha256")
    ):
        raise RuntimeError(
            "Blind-review analysis public packet hash has changed"
        )
    if (
        identity.get("packet_id") != packet.get("packet_id")
        or identity.get("packet_schema") != packet.get("packet_schema")
        or identity.get("public_manifest_sha256")
        != packet.get("public_manifest_sha256")
    ):
        raise RuntimeError(
            "Blind-review analysis identity differs from packet metadata"
        )
    sources = manifest.get("source_annotations")
    if not isinstance(sources, list) or len(sources) < 2:
        raise RuntimeError(
            "Blind-review analysis source annotation list is invalid"
        )
    for source in sources:
        if not isinstance(source, Mapping):
            raise RuntimeError("Invalid source annotation metadata")
        path = Path(str(source.get("path", "")))
        if not path.is_file() or _sha256(path) != source.get("sha256"):
            raise RuntimeError(
                f"Blind-review source annotation hash changed: {path}"
            )
    if (
        identity.get("annotation_sha256")
        != [source.get("sha256") for source in sources]
        or identity.get("reviewers")
        != [source.get("reviewer") for source in sources]
        or identity.get("implementation_sha256") != _sha256(Path(__file__))
    ):
        raise RuntimeError(
            "Blind-review analysis source/method identity has changed"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(OUTPUT_FILES):
        raise RuntimeError("Blind-review analysis output inventory is invalid")
    for name, expected_hash in outputs.items():
        path = output_dir / str(name)
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(
                f"Blind-review analysis output hash changed: {path}"
            )
    summary = json.loads(
        (output_dir / "agreement_summary.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(summary, Mapping)
        or summary.get("analysis_id") != manifest.get("analysis_id")
        or summary.get("packet_id") != packet.get("packet_id")
        or summary.get("packet_schema") != packet.get("packet_schema")
        or summary.get("private_key_read") is not False
    ):
        raise RuntimeError("Blind-review agreement summary is inconsistent")
    return {
        "analysis_id": manifest.get("analysis_id"),
        "packet_id": packet.get("packet_id"),
        "cases": summary.get("case_count"),
        "reviewers": len(summary.get("reviewers", [])),
        "reviewer_pairs": summary.get("reviewer_pair_count"),
        "outputs_verified": len(outputs),
        "private_key_read": False,
        "source_files_rewritten": False,
    }


__all__ = [
    "AGREEMENT_FIELDS",
    "BLIND_ANALYSIS_SCHEMA",
    "analyze_blind_review",
    "validate_blind_review_analysis",
]
