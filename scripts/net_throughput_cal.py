"""Compute network throughput (receive + transmit) for all tenants and rounds.

This script recursively searches a root directory (default: data/) for pairs of
network_receive.csv and network_transmit.csv files that live in the same folder
(typically a tenant folder inside a phase inside a round) and produces a
network_throughput.csv alongside them containing the summed value per timestamp.

Features:
 - Recursive discovery
 - Skips folders where one of the files is missing
 - Safe overwrite control (--overwrite)
 - Dry-run mode to preview actions (--dry-run)
 - Summary report at the end
 - Minimal memory footprint (can optionally stream in future if needed)

Usage examples:
  python scripts/net_throughput_cal.py                # process under ./data
  python scripts/net_throughput_cal.py --root data/sfi2-long --overwrite
  python scripts/net_throughput_cal.py --root /custom/path --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd


def compute_throughput(receive_path: Path, transmit_path: Path, output_path: Path, overwrite: bool = False) -> str:
    if output_path.exists() and not overwrite:
        return "skip-exists"
    try:
        receive_df = pd.read_csv(receive_path)
        transmit_df = pd.read_csv(transmit_path)
    except Exception as e:  # broad but we log
        logging.warning("Failed reading %s / %s: %s", receive_path, transmit_path, e)
        return "error-read"

    missing_cols = {"timestamp", "value"} - set(receive_df.columns) | {"timestamp", "value"} - set(transmit_df.columns)
    if missing_cols:
        logging.warning("Missing required columns %s in %s or %s", missing_cols, receive_path, transmit_path)
        return "error-schema"

    # Inner join keeps only aligned timestamps; warn if lengths differ
    if len(receive_df) != len(transmit_df):
        logging.debug(
            "Length mismatch (receive=%d transmit=%d) for folder %s", len(receive_df), len(transmit_df), receive_path.parent
        )
    merged_df = pd.merge(receive_df, transmit_df, on="timestamp", suffixes=("_receive", "_transmit"), how="inner")
    if merged_df.empty:
        logging.warning("No overlapping timestamps for %s", receive_path.parent)
        return "empty-merge"

    merged_df["value"] = merged_df["value_receive"] + merged_df["value_transmit"]
    throughput_df = merged_df[["timestamp", "value"]]
    try:
        throughput_df.to_csv(output_path, index=False)
    except Exception as e:  # pragma: no cover (filesystem errors)
        logging.error("Failed writing %s: %s", output_path, e)
        return "error-write"
    return "written"


def find_candidate_folders(root: Path) -> List[Path]:
    # We look for directories containing both network_receive.csv and network_transmit.csv
    receive_name = "network_receive.csv"
    transmit_name = "network_transmit.csv"
    candidates: List[Path] = []
    for path in root.rglob(receive_name):
        folder = path.parent
        if (folder / transmit_name).exists():
            candidates.append(folder)
    return candidates


def process(root: Path, overwrite: bool = False, dry_run: bool = False) -> None:
    folders = find_candidate_folders(root)
    if not folders:
        logging.info("No folders with receive/transmit pair found under %s", root)
        return
    logging.info("Found %d folders with network receive/transmit", len(folders))

    stats = {"written": 0, "skip-exists": 0, "error-read": 0, "error-schema": 0, "empty-merge": 0, "error-write": 0}
    for folder in sorted(folders):
        receive_path = folder / "network_receive.csv"
        transmit_path = folder / "network_transmit.csv"
        output_path = folder / "network_throughput.csv"
        rel = output_path.relative_to(root)
        if dry_run:
            action = "would-skip" if output_path.exists() and not overwrite else "would-write"
            logging.info("[%s] %s", action, rel)
            continue
        result = compute_throughput(receive_path, transmit_path, output_path, overwrite=overwrite)
        if result in stats:
            stats[result] += 1
        logging.debug("Processed %s -> %s", rel, result)

    if dry_run:
        logging.info("Dry-run complete.")
        return

    total = sum(stats.values())
    logging.info(
        "Summary: %d processed | written=%d skip-exists=%d error-read=%d error-schema=%d empty-merge=%d error-write=%d",
        total,
        stats["written"],
        stats["skip-exists"],
        stats["error-read"],
        stats["error-schema"],
        stats["empty-merge"],
        stats["error-write"],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute network throughput for all tenants (receive+transmit)")
    p.add_argument("--root", type=Path, default=Path("data"), help="Root directory to search (default: ./data)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing network_throughput.csv files")
    p.add_argument("--dry-run", action="store_true", help="List actions without writing output")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    root: Path = args.root
    if not root.exists():
        logging.error("Root path %s does not exist", root)
        return
    process(root, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    main()