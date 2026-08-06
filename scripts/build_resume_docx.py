#!/usr/bin/env python3
"""Render the two Markdown resume sources to styled Word documents.

The Markdown files remain the editable source of truth.  Conversion happens
through a temporary, self-contained HTML document and LibreOffice; no network
resources are used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = ROOT / "docs" / "resume"
SOURCES = (
    RESUME_DIR / "FastWAM项目经历_简略版.md",
    RESUME_DIR / "FastWAM项目经历_详细版.md",
)

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
@page {{ size: A4; margin: 1.45cm 1.55cm; }}
body {{
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  font-size: 10.2pt;
  line-height: 1.48;
  color: #172033;
}}
h1 {{
  color: #173b66;
  font-size: 20pt;
  margin: 0 0 10pt 0;
  border-bottom: 2pt solid #3568b0;
  padding-bottom: 5pt;
}}
h2 {{
  color: #244f7f;
  font-size: 14pt;
  margin: 14pt 0 6pt 0;
  border-bottom: 0.6pt solid #b7c9df;
  padding-bottom: 2pt;
}}
h3 {{ color: #315f8f; font-size: 11.5pt; margin: 10pt 0 4pt 0; }}
p {{ margin: 4pt 0 6pt 0; }}
ul, ol {{ margin: 4pt 0 7pt 18pt; padding-left: 6pt; }}
li {{ margin: 2.5pt 0; }}
strong {{ color: #152f4e; }}
code {{
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.8pt;
  background: #eef3f8;
  color: #24364b;
  padding: 0.5pt 2pt;
}}
pre {{
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.5pt;
  line-height: 1.35;
  background: #f3f6f9;
  border-left: 3pt solid #7d9fc4;
  padding: 7pt;
  white-space: pre-wrap;
}}
table {{ width: 100%; border-collapse: collapse; margin: 7pt 0 10pt 0; }}
th {{ background: #dce8f7; color: #173b66; font-weight: bold; }}
th, td {{ border: 0.6pt solid #9eb2c9; padding: 4pt 5pt; vertical-align: top; }}
blockquote {{
  margin: 6pt 0;
  padding: 5pt 9pt;
  border-left: 3pt solid #3568b0;
  background: #f3f7fb;
}}
a {{ color: #285f9d; text-decoration: none; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def markdown_to_html(source: Path) -> str:
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=("tables", "fenced_code", "sane_lists"),
        output_format="html5",
    )
    return HTML_TEMPLATE.format(title=source.stem, body=body)


def validate_docx(destination: Path) -> int:
    with zipfile.ZipFile(destination) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"Corrupt DOCX member: {corrupt_member}")
        document_xml = archive.read("word/document.xml")
        if len(document_xml) < 1_000:
            raise RuntimeError(f"DOCX content is unexpectedly small: {destination}")

    with tempfile.TemporaryDirectory(prefix="fastwam-resume-check-") as temporary:
        temporary_dir = Path(temporary)
        output_dir = temporary_dir / "output"
        profile_dir = temporary_dir / "libreoffice-profile"
        runtime_dir = temporary_dir / "runtime"
        config_dir = temporary_dir / "config"
        cache_dir = temporary_dir / "cache"
        for directory in (output_dir, profile_dir, config_dir, cache_dir):
            directory.mkdir()
        runtime_dir.mkdir(mode=0o700)
        environment = os.environ.copy()
        environment.update(
            {
                "XDG_RUNTIME_DIR": os.fspath(runtime_dir),
                "XDG_CONFIG_HOME": os.fspath(config_dir),
                "XDG_CACHE_HOME": os.fspath(cache_dir),
                "SAL_USE_VCLPLUGIN": "svp",
            }
        )
        completed = subprocess.run(
            [
                "libreoffice",
                f"-env:UserInstallation={profile_dir.as_uri()}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                os.fspath(output_dir),
                os.fspath(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        pdf_path = output_dir / f"{destination.stem}.pdf"
        if completed.returncode != 0 or not pdf_path.is_file():
            raise RuntimeError(
                "DOCX PDF validation failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        info = subprocess.run(
            ["pdfinfo", os.fspath(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in info.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", maxsplit=1)[1].strip())
        raise RuntimeError(f"pdfinfo did not report a page count for {destination}")


def render_one(source: Path) -> tuple[Path, int]:
    destination = source.with_suffix(".docx")
    with tempfile.TemporaryDirectory(prefix="fastwam-resume-") as temporary:
        temporary_dir = Path(temporary)
        html_path = temporary_dir / f"{source.stem}.html"
        output_dir = temporary_dir / "output"
        profile_dir = temporary_dir / "libreoffice-profile"
        runtime_dir = temporary_dir / "runtime"
        config_dir = temporary_dir / "config"
        cache_dir = temporary_dir / "cache"
        output_dir.mkdir()
        profile_dir.mkdir()
        runtime_dir.mkdir(mode=0o700)
        config_dir.mkdir()
        cache_dir.mkdir()
        html_path.write_text(markdown_to_html(source), encoding="utf-8")

        command = [
            "libreoffice",
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--headless",
            "--convert-to",
            'docx:Office Open XML Text',
            "--outdir",
            os.fspath(output_dir),
            os.fspath(html_path),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "XDG_RUNTIME_DIR": os.fspath(runtime_dir),
                "XDG_CONFIG_HOME": os.fspath(config_dir),
                "XDG_CACHE_HOME": os.fspath(cache_dir),
                "SAL_USE_VCLPLUGIN": "svp",
            }
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        generated = output_dir / f"{source.stem}.docx"
        if completed.returncode != 0 or not generated.is_file():
            raise RuntimeError(
                "LibreOffice conversion failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        replacement = destination.with_suffix(".docx.tmp")
        shutil.copyfile(generated, replacement)
        os.replace(replacement, destination)
    return destination, validate_docx(destination)


def main() -> None:
    for source in SOURCES:
        if not source.is_file():
            raise FileNotFoundError(source)
        rendered, page_count = render_one(source)
        print(f"{rendered.relative_to(ROOT)} ({page_count} pages)")


if __name__ == "__main__":
    main()
