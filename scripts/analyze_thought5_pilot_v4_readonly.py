#!/usr/bin/env python3
"""Create the sealed Pilot v4 read-only failure decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastwam_ood_eval.thought5.readonly_failure_analysis import write_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="outputs/thought5/phase5_camera_equivariant_geo_repa_pilot_v4",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/thought5/"
            "phase5_camera_equivariant_geo_repa_pilot_v4_readonly_failure_v1"
        ),
    )
    args = parser.parse_args()
    output = write_analysis(Path(args.source), Path(args.output))
    print(
        {
            "status": "complete",
            "analysis_role": "posthoc_readonly_exploratory",
            "pilot_decision_preserved": True,
            "formal_unlocked": False,
            "result": str(output / "analysis_result.json"),
            "report": str(output / "report.md"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

