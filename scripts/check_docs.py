#!/usr/bin/env python3
"""Validate the organized documentation tree and its local Markdown links."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MARKDOWN_FILES = [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
OLD_FLAT_RE = re.compile(
    r"docs/(?:thought[123]_[A-Za-z0-9_.-]+\.md|"
    r"(?:architecture|engineering_highlights|environment_setup|experiment_ledger|"
    r"experiment_protocol|research_index|results_schema|troubleshooting|"
    r"upstream_notes)\.md)"
)
HISTORICAL_NONEXISTENT_PATHS = {
    "docs/thought1_protocol.md",
    "docs/thought3_analysis_protocol_FROZEN.md",
}


def destination_path(raw: str) -> str | None:
    target = raw.strip()
    if not target or target.startswith("#"):
        return None
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if any(
        target.startswith(prefix)
        for prefix in ("http://", "https://", "mailto:", "data:")
    ):
        return None
    # This repository does not use titled local destinations.  If one appears,
    # fail explicitly rather than parsing it ambiguously.
    if " " in target:
        return target
    return unquote(target.split("#", 1)[0])


def main() -> int:
    errors: list[str] = []

    root_markdown = sorted(path.name for path in DOCS.glob("*.md"))
    if root_markdown != ["README.md"]:
        errors.append(
            "docs/ root must contain only README.md; found: "
            + ", ".join(root_markdown)
        )

    for source in MARKDOWN_FILES:
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            destination = destination_path(match.group(1))
            if destination is None:
                continue
            if " " in destination:
                errors.append(
                    f"{source.relative_to(ROOT)}:{text.count(chr(10), 0, match.start()) + 1}: "
                    f"ambiguous local link destination {destination!r}"
                )
                continue
            target = (
                ROOT / destination.lstrip("/")
                if destination.startswith("/")
                else source.parent / destination
            )
            if not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{source.relative_to(ROOT)}:{line}: missing target "
                    f"{match.group(1)!r}"
                )

        for match in OLD_FLAT_RE.finditer(text):
            old_path = match.group(0)
            if old_path not in HISTORICAL_NONEXISTENT_PATHS:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{source.relative_to(ROOT)}:{line}: stale flat docs path "
                    f"{old_path!r}"
                )

    if errors:
        print("Documentation validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Documentation validation passed: {len(MARKDOWN_FILES)} Markdown files, "
        "all local links resolve, and docs/ root is clean."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
