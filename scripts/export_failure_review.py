#!/usr/bin/env python3
"""Generate the static failure-review bundle required by the experiment protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastwam_ood_eval.analysis.review import generate_failure_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    print(generate_failure_review(args.experiment_dir))


if __name__ == "__main__":
    main()

