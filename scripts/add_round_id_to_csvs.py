#!/usr/bin/env python3
"""
Backfill a 'round_id' column into row-based CSV outputs by inferring it from the
directory structure: outputs/<dataset>/<experiment>/<round-id>/<stage>/csv/*.csv

Targets only tidy/row-based CSVs to avoid breaking matrix-shaped CSVs:
  - descriptive_stats_*.csv
  - impact_analysis_summary.csv
  - phase_comparison_stats_*.csv
  - correlation_summary.csv
  - covariance_summary.csv
  - granger_tidy.csv
  - transfer_entropy_tidy.csv

Usage:
  python scripts/add_round_id_to_csvs.py [--root outputs] [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


ROW_BASED_FILENAMES = {
    "impact_analysis_summary.csv",
    "correlation_summary.csv",
    "covariance_summary.csv",
    "granger_tidy.csv",
    "transfer_entropy_tidy.csv",
}


def is_target_csv(path: Path) -> bool:
    name = path.name
    if name in ROW_BASED_FILENAMES:
        return True
    if name.startswith("descriptive_stats_") and name.endswith(".csv"):
        return True
    if name.startswith("phase_comparison_stats_") and name.endswith(".csv"):
        return True
    return False


def infer_round_id(path: Path) -> str | None:
    # Expect .../outputs/<dataset>/<experiment>/<round-id>/<stage>/csv/<file>
    parts = path.parts
    try:
        csv_idx = parts.index("csv")
        stage = parts[csv_idx - 1]
        round_id = parts[csv_idx - 2]
        # Basic sanity check: stage likely ends with "_analysis"
        if not stage.endswith("analysis") and not stage.endswith("_analysis"):
            # still accept, round_id position should be correct regardless
            pass
        return round_id
    except Exception:
        return None


def process_csv(csv_path: Path, dry_run: bool = False) -> tuple[bool, str]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Failed to read {csv_path}: {e}"

    if "round_id" in df.columns:
        return False, "Already has round_id"

    round_id = infer_round_id(csv_path)
    if not round_id:
        return False, "Could not infer round_id from path"

    df["round_id"] = round_id
    if dry_run:
        return True, f"Would add round_id={round_id}"

    try:
        df.to_csv(csv_path, index=False)
        return True, f"Added round_id={round_id}"
    except Exception as e:
        return False, f"Failed to write {csv_path}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs", help="Outputs root directory")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify files; just report actions")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root directory not found: {root}", file=sys.stderr)
        return 2

    count = 0
    updated = 0
    for csv_path in root.rglob("*.csv"):
        if not is_target_csv(csv_path):
            continue
        count += 1
        changed, msg = process_csv(csv_path, dry_run=args.dry_run)
        status = "CHANGED" if changed else "SKIP"
        print(f"[{status}] {csv_path}: {msg}")
        if changed:
            updated += 1

    print(f"Processed {count} CSVs, updated {updated}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
