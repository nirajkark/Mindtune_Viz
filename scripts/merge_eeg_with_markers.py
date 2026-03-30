from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SessionPaths:
    source_folder: str
    session_dir: str
    eeg_csv: str
    markers_csv: str
    meta_json: Optional[str]

def _discover_session_roots() -> List[str]:
    roots: List[str] = []
    for p in sorted(glob.glob("sessions*")):
        if os.path.isdir(p):
            roots.append(p)
    return roots


def _find_sessions(root: str) -> List[SessionPaths]:
    out: List[SessionPaths] = []
    for session_dir in sorted(glob.glob(os.path.join(root, "*"))):
        eeg_csv = os.path.join(session_dir, "eeg_rows.csv")
        markers_csv = os.path.join(session_dir, "markers.csv")
        if not (os.path.isfile(eeg_csv) and os.path.isfile(markers_csv)):
            continue
        meta_json = os.path.join(session_dir, "session_meta.json")
        if not os.path.isfile(meta_json):
            meta_json = None
        out.append(
            SessionPaths(
                source_folder=root,
                session_dir=session_dir,
                eeg_csv=eeg_csv,
                markers_csv=markers_csv,
                meta_json=meta_json,
            )
        )
    return out


def _read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        try:
            return next(r)
        except StopIteration:
            return []


def _iter_dict_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def _safe_int(x: object, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _load_events(meta_json_path: Optional[str], markers_csv: str) -> List[str]:
    if meta_json_path:
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            events = meta.get("events")
            if isinstance(events, list) and all(isinstance(e, str) for e in events):
                return list(events)
        except Exception:
            pass

    # Fallback: infer from markers.csv
    evs: List[str] = []
    seen = set()
    for row in _iter_dict_rows(markers_csv):
        if (row.get("marker_type") or "").strip() != "event":
            continue
        v = (row.get("field") or "").strip()
        # In your data, field holds the event name (e.g., ev_speaking)
        if v and v not in seen:
            seen.add(v)
            evs.append(v)
    return evs


def _load_sorted_markers(markers_csv: str) -> List[Dict[str, str]]:
    rows = list(_iter_dict_rows(markers_csv))
    rows.sort(key=lambda r: _safe_int(r.get("timestamp_ms"), default=0))
    return rows


def _merge_one_session(sp: SessionPaths, *, writer: csv.DictWriter, out_cols: Sequence[str]) -> int:
    markers = _load_sorted_markers(sp.markers_csv)
    events = _load_events(sp.meta_json, sp.markers_csv)
    active_events: Dict[str, int] = {e: 0 for e in events}

    current_emotion: str = ""
    current_emotion_conf: str = ""

    mi = 0
    n_written = 0

    for eeg_row in _iter_dict_rows(sp.eeg_csv):
        t = _safe_int(eeg_row.get("timestamp_ms"), default=0)

        # Advance marker state up to this eeg timestamp.
        while mi < len(markers) and _safe_int(markers[mi].get("timestamp_ms"), default=0) <= t:
            m = markers[mi]
            mi += 1

            mtype = (m.get("marker_type") or "").strip()
            field = (m.get("field") or "").strip()
            action = (m.get("action") or "").strip()
            value = (m.get("value") or "").strip()

            if mtype == "emotion" and field == "emotion" and action == "set":
                current_emotion = value
                current_emotion_conf = (m.get("confidence") or "").strip()
                continue

            if mtype == "event" and field:
                # field is the event name (ev_speaking, etc.)
                if field not in active_events:
                    active_events[field] = 0
                if action == "start":
                    active_events[field] = 1
                elif action == "end":
                    active_events[field] = 0

        out_row: Dict[str, str] = {}
        out_row["source_folder"] = sp.source_folder
        out_row["source_session_path"] = sp.session_dir

        # Copy all EEG columns through.
        out_row.update(eeg_row)

        # Marker-derived enrichment.
        out_row["marker_emotion"] = current_emotion
        out_row["marker_emotion_confidence"] = current_emotion_conf
        for ev, is_active in active_events.items():
            out_row[f"marker_{ev}_active"] = str(int(is_active))

        # Ensure all columns exist (DictWriter will fill missing with empty).
        writer.writerow({c: out_row.get(c, "") for c in out_cols})
        n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge each session's eeg_rows.csv with markers.csv by time, producing EEG rows enriched with marker-based emotion + active events."
        )
    )
    parser.add_argument(
        "--out",
        default="all_sessions_eeg_with_markers.csv",
        help="Output CSV path (default: all_sessions_eeg_with_markers.csv)",
    )
    parser.add_argument(
        "--include-sessions2",
        action="store_true",
        help='Deprecated (kept for compatibility). All "sessions*" folders are included automatically.',
    )
    args = parser.parse_args()

    sessions: List[SessionPaths] = []
    for root in _discover_session_roots():
        sessions.extend(_find_sessions(root))

    if not sessions:
        raise SystemExit('No sessions found under any "sessions*" folder.')

    # Build output column union: EEG header union + marker enrichment union.
    eeg_union: List[str] = []
    seen = set()
    marker_event_cols: List[str] = []
    marker_seen = set()

    for sp in sessions:
        for c in _read_header(sp.eeg_csv):
            if c and c not in seen:
                seen.add(c)
                eeg_union.append(c)

        events = _load_events(sp.meta_json, sp.markers_csv)
        for ev in events:
            col = f"marker_{ev}_active"
            if col not in marker_seen:
                marker_seen.add(col)
                marker_event_cols.append(col)

    out_cols = (
        ["source_folder", "source_session_path"]
        + eeg_union
        + ["marker_emotion", "marker_emotion_confidence"]
        + marker_event_cols
    )

    total_rows = 0
    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        for sp in sessions:
            total_rows += _merge_one_session(sp, writer=writer, out_cols=out_cols)

    print(f"wrote {args.out} with {total_rows} rows from {len(sessions)} sessions")


if __name__ == "__main__":
    main()

