from __future__ import annotations

import argparse
import csv
from typing import Dict, Iterable, List, Optional, Set, Tuple


BANDS = [
    "delta",
    "theta",
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "low_gamma",
    "mid_gamma",
]


def _safe_float(x: Optional[str]) -> float:
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if s == "":
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _safe_int(x: Optional[str]) -> int:
    try:
        if x is None:
            return 0
        s = str(x).strip()
        if s == "":
            return 0
        return int(float(s))
    except Exception:
        return 0


def _derive_label_3class(row: Dict[str, str]) -> str:
    # Same rule as scripts/train_model.py: stressed > calm > neutral
    if _safe_int(row.get("emo_stressed")) == 1:
        return "stressed"
    if _safe_int(row.get("emo_calm")) == 1:
        return "calm"
    return "neutral"


def _compute_band_pcts(row: Dict[str, str], available_bands: List[str]) -> Dict[str, float]:
    vals = {b: _safe_float(row.get(b)) for b in available_bands}
    total = sum(vals.values())
    if total <= 0:
        return {f"{b}_pct": 0.0 for b in available_bands}
    return {f"{b}_pct": vals[b] / total for b in available_bands}


def _iter_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create model-ready CSV exports from all_sessions_eeg_with_markers.csv without pandas."
    )
    parser.add_argument(
        "--in",
        dest="inp",
        default="all_sessions_eeg_with_markers.csv",
        help="Input CSV (default: all_sessions_eeg_with_markers.csv)",
    )
    parser.add_argument(
        "--out",
        default="model_ready.csv",
        help="Output CSV (default: model_ready.csv)",
    )
    parser.add_argument(
        "--out-strict",
        default="model_ready_strict.csv",
        help="Strict output CSV (default: model_ready_strict.csv)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate rows by (source_session_path, timestamp_ms), keeping first seen.",
    )
    parser.add_argument(
        "--include-events",
        action="store_true",
        help="Include marker-derived event activity columns (marker_ev_*_active) as features.",
    )
    args = parser.parse_args()

    # Discover which band columns exist from the header by peeking first row.
    first_row: Optional[Dict[str, str]] = None
    for r in _iter_rows(args.inp):
        first_row = r
        break
    if first_row is None:
        raise SystemExit("Input CSV is empty.")

    available_bands = [b for b in BANDS if b in first_row]

    id_cols = [c for c in ["source_folder", "source_session_path", "session_id", "participant_id", "timestamp_ms"] if c in first_row]
    event_cols: List[str] = []
    if args.include_events:
        for k in first_row.keys():
            if k.startswith("marker_ev_") and k.endswith("_active"):
                event_cols.append(k)
        event_cols = sorted(event_cols)

    feature_cols = (
        [f"{b}_pct" for b in available_bands]
        + [c for c in ["attention", "meditation", "signal_quality"] if c in first_row]
        + event_cols
    )
    out_cols = id_cols + ["label_3class"] + feature_cols

    seen: Set[Tuple[str, str]] = set()

    def _should_keep(row: Dict[str, str]) -> bool:
        if not args.dedupe:
            return True
        key = (row.get("source_session_path", ""), row.get("timestamp_ms", ""))
        if key in seen:
            return False
        seen.add(key)
        return True

    # Write standard + strict in one pass.
    n_std = 0
    n_strict = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f_std, open(
        args.out_strict, "w", encoding="utf-8", newline=""
    ) as f_strict:
        w_std = csv.DictWriter(f_std, fieldnames=out_cols)
        w_strict = csv.DictWriter(f_strict, fieldnames=out_cols)
        w_std.writeheader()
        w_strict.writeheader()

        for row in _iter_rows(args.inp):
            if not _should_keep(row):
                continue

            out: Dict[str, str] = {c: row.get(c, "") for c in id_cols}
            out["label_3class"] = _derive_label_3class(row)

            # band pcts
            pcts = _compute_band_pcts(row, available_bands)
            for k, v in pcts.items():
                out[k] = f"{v:.10f}"

            # numeric passthroughs
            for c in ["attention", "meditation", "signal_quality"]:
                if c in feature_cols:
                    out[c] = row.get(c, "")
            # event passthroughs (already 0/1 strings)
            for c in event_cols:
                out[c] = row.get(c, "")

            w_std.writerow(out)
            n_std += 1

            # strict filter: drop emo_unknown==1 and marker_emotion missing (if present)
            emo_unknown = _safe_int(row.get("emo_unknown"))
            marker_emotion = row.get("marker_emotion")
            if emo_unknown == 1:
                continue
            if "marker_emotion" in first_row and (marker_emotion is None or str(marker_emotion).strip() == ""):
                continue
            w_strict.writerow(out)
            n_strict += 1

    print(f"wrote {args.out} rows={n_std} cols={len(out_cols)}")
    print(f"wrote {args.out_strict} rows={n_strict} cols={len(out_cols)}")


if __name__ == "__main__":
    main()

