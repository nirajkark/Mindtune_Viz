from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Sequence, Tuple


def _find_eeg_csvs(root: str) -> List[str]:
    pattern = os.path.join(root, "*", "eeg_rows.csv")
    return sorted(glob.glob(pattern))

def _discover_session_roots() -> List[str]:
    roots: List[str] = []
    for p in sorted(glob.glob("sessions*")):
        if os.path.isdir(p):
            roots.append(p)
    return roots


def _read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _iter_rows(path: str) -> Sequence[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all session eeg_rows.csv files from all session roots (sessions* folders) into one CSV."
        )
    )
    parser.add_argument(
        "--out",
        default="all_sessions_eeg_rows.csv",
        help="Output CSV path (default: all_sessions_eeg_rows.csv)",
    )
    parser.add_argument(
        "--include-sessions2",
        action="store_true",
        help='Deprecated (kept for compatibility). All "sessions*" folders are included automatically.',
    )
    args = parser.parse_args()

    sources: List[Tuple[str, str]] = []
    for root in _discover_session_roots():
        for p in _find_eeg_csvs(root):
            sources.append((root, p))

    if not sources:
        raise SystemExit(
            'No eeg_rows.csv found. Expected "sessions*/*/eeg_rows.csv" under the repo root.'
        )

    # 1) Compute union of all columns across all sources.
    union_cols: List[str] = []
    seen = set()
    for _, path in sources:
        for c in _read_header(path):
            if c not in seen:
                seen.add(c)
                union_cols.append(c)

    # Add traceability columns up front.
    out_cols = ["source_folder", "source_session_path"] + union_cols

    # 2) Stream-write merged CSV so we don't need pandas.
    row_count = 0
    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        for source_folder, path in sources:
            for row in _iter_rows(path):
                row_out = {"source_folder": source_folder, "source_session_path": path}
                row_out.update(row)
                writer.writerow(row_out)
                row_count += 1

    print(f"wrote {args.out} with {row_count} rows from {len(sources)} files")


if __name__ == "__main__":
    main()

