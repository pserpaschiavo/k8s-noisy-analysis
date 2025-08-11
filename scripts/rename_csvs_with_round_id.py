#!/usr/bin/env python3
"""
Rename existing CSVs in outputs/ to include the round_id in the filename where
appropriate, matching the pipeline's new naming convention.

Targets:
- outputs/.../impact_analysis/csv/impact_analysis_summary.csv -> impact_analysis_summary_<round>.csv
- outputs/.../correlation_analysis/csv/(correlation_summary|covariance_summary).csv -> *_<round>.csv
- outputs/.../causality_analysis/csv/(granger_tidy|transfer_entropy_tidy).csv -> *_<round>.csv
- outputs/.../descriptive_analysis/csv/descriptive_stats_<round>.csv (already contains round)
- outputs/.../phase_comparison_analysis/csv/phase_comparison_stats_<round>.csv (already contains round)

Usage:
  python scripts/rename_csvs_with_round_id.py [--root outputs] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def infer_round_id(path: Path) -> str | None:
    parts = path.parts
    try:
        csv_idx = parts.index("csv")
        return parts[csv_idx - 2]  # .../<round>/<stage>/csv/<file>
    except Exception:
        return None


def target_new_name(path: Path, round_id: str) -> Path | None:
    name = path.name
    parent = path.parent
    if name == "impact_analysis_summary.csv":
        return parent / f"impact_analysis_summary_{round_id}.csv"
    if name == "correlation_summary.csv":
        return parent / f"correlation_summary_{round_id}.csv"
    if name == "covariance_summary.csv":
        return parent / f"covariance_summary_{round_id}.csv"
    if name == "granger_tidy.csv":
        return parent / f"granger_tidy_{round_id}.csv"
    if name == "transfer_entropy_tidy.csv":
        return parent / f"transfer_entropy_tidy_{round_id}.csv"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    candidates = [
        "impact_analysis_summary.csv",
        "correlation_summary.csv",
        "covariance_summary.csv",
        "granger_tidy.csv",
        "transfer_entropy_tidy.csv",
    ]

    renames = 0
    total = 0
    for file in root.rglob("*.csv"):
        if file.name not in candidates:
            continue
        total += 1
        round_id = infer_round_id(file)
        if not round_id:
            print(f"[SKIP] {file}: could not infer round_id")
            continue
        new_path = target_new_name(file, round_id)
        if not new_path:
            print(f"[SKIP] {file}: not a target file")
            continue
        if new_path.exists():
            print(f"[SKIP] {file}: target exists -> {new_path.name}")
            continue
        print(f"[RENAME] {file.name} -> {new_path.name}")
        if not args.dry_run:
            file.rename(new_path)
            renames += 1

    print(f"Processed {total} files; renamed {renames}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
