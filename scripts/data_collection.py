from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from mindtune_stress_model.stream_adapter import CANONICAL_BANDS
from mindtune_stress_model.stream_adapter import CsvPlaybackStreamAdapter
from mindtune_stress_model.types import LabeledWindow
from mindtune_stress_model.windowing import SlidingWindowBuffer


LABELS = ["stressed", "calm", "neutral"]


def _derive_label_from_raw(raw: Dict[str, Any]) -> Optional[str]:
    # Matches your session_meta columns: emo_calm, emo_stressed.
    if "emo_calm" not in raw and "emo_stressed" not in raw:
        return None
    try:
        emo_stressed = int(raw.get("emo_stressed", 0) or 0)
        if emo_stressed == 1:
            return "stressed"
        emo_calm = int(raw.get("emo_calm", 0) or 0)
        if emo_calm == 1:
            return "calm"
    except Exception:
        return None
    return "neutral"


def _label_input_thread(q: "queue.Queue[Tuple[str, float]]", get_current_ts_s, stop_event: threading.Event) -> None:
    """
    Read single-character commands from stdin:
      s => stressed
      c => calm
      n => neutral
      q => quit
    """

    print("Label commands: [s] stressed, [c] calm, [n] neutral, [q] quit", flush=True)
    while not stop_event.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            cmd = line.strip().lower()
            if not cmd:
                continue

            if cmd == "q":
                stop_event.set()
                return

            if cmd not in {"s", "c", "n"}:
                print("Unknown command. Use s/c/n or q.", flush=True)
                continue

            label = {"s": "stressed", "c": "calm", "n": "neutral"}[cmd]
            ts_s = float(get_current_ts_s())
            q.put((label, ts_s))
        except Exception:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mock_csv", choices=["mock_csv"])
    parser.add_argument("--csv-path", default="Downloads/mindtune_full_eeg_data.csv")
    parser.add_argument("--output", default="collected_dataset.csv")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--label-max-delay", type=float, default=2.0, help="Max |window_end - label_time| allowed")
    parser.add_argument("--max-duration-s", type=float, default=60.0, help="Stop after this many seconds (mock stream time)")
    parser.add_argument("--derive-labels-from-csv", action="store_true", help="Auto-label using emo_calm/emo_stressed columns if present")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed for mock_csv")
    parser.add_argument("--no-sleep", action="store_true", help="Disable sleeping during playback")
    args = parser.parse_args()

    if args.backend != "mock_csv":
        raise NotImplementedError("Only mock_csv backend is implemented in this repo version.")

    # Adapter yields per-sample features.
    stream = CsvPlaybackStreamAdapter(
        args.csv_path,
        speed=args.speed,
        sleep=not args.no_sleep,
    )
    buffer = SlidingWindowBuffer(window_seconds=args.window_seconds, step_seconds=None)

    # Keep recent window features for event alignment.
    recent_windows: Deque[Tuple[float, float, Dict[str, float]]] = deque()
    # recent_windows items: (window_end_s, window_start_s, features)

    latest_ts_lock = threading.Lock()
    latest_ts_s = 0.0

    def get_current_ts_s() -> float:
        with latest_ts_lock:
            return latest_ts_s

    label_q: "queue.Queue[Tuple[str, float]]" = queue.Queue()
    stop_event = threading.Event()
    input_thread = None

    if not args.derive_labels_from_csv:
        input_thread = threading.Thread(
            target=_label_input_thread,
            args=(label_q, get_current_ts_s, stop_event),
            daemon=True,
        )
        input_thread.start()

    # Collect rows in-memory then write once; easiest and reliable for consistent column ordering.
    rows: List[Dict[str, Any]] = []
    dataset_columns: Optional[List[str]] = None

    start_real_time = time.time()
    stop_ts_s = None

    for sample in stream:
        with latest_ts_lock:
            latest_ts_s = float(sample.timestamp_s)

        win = buffer.add_sample(sample)
        if win is None:
            # Not enough samples in the current window (or we’re gating by step).
            continue

        # Prune recent windows older than label_max_delay.
        cutoff = win.window_end_s - (args.label_max_delay + 5.0)
        while recent_windows and recent_windows[0][0] < cutoff:
            recent_windows.popleft()

        recent_windows.append((win.window_end_s, win.window_start_s, dict(win.features)))

        # Optional auto-label (useful for dry runs / training).
        if args.derive_labels_from_csv:
            label = _derive_label_from_raw(sample.raw)
            if label is None:
                continue
            row = {
                "window_end_s": win.window_end_s,
                "window_start_s": win.window_start_s,
                "label": label,
                **win.features,
            }
            rows.append(row)
            if dataset_columns is None:
                dataset_columns = list(row.keys())

        # Otherwise, capture human labels from the input queue.
        else:
            while not label_q.empty():
                label, label_ts_s = label_q.get_nowait()
                if label not in LABELS:
                    continue

                # Choose the nearest window_end_s within allowed delay.
                best = None
                best_dist = None
                for w_end, w_start, feats in recent_windows:
                    dist = abs(w_end - label_ts_s)
                    if dist <= args.label_max_delay and (best_dist is None or dist < best_dist):
                        best = (w_end, w_start, feats)
                        best_dist = dist

                if best is None:
                    print(
                        f"Skipped label={label}: no window within {args.label_max_delay}s of t={label_ts_s:.2f}s",
                        flush=True,
                    )
                    continue

                w_end, w_start, feats = best
                row = {
                    "window_end_s": float(w_end),
                    "window_start_s": float(w_start),
                    "label": label,
                    **feats,
                }
                rows.append(row)
                if dataset_columns is None:
                    dataset_columns = list(row.keys())

        # Stop conditions
        if stop_event.is_set():
            break
        if args.max_duration_s is not None and (sample.timestamp_s >= float(args.max_duration_s)):
            stop_event.set()
            break
        if (time.time() - start_real_time) > (float(args.max_duration_s) / max(args.speed, 1e-6) + 10.0):
            # Real-time safety net; prevents hangs if time parsing/sleep mismatch.
            stop_event.set()
            break

    # Write dataset.
    if not rows:
        print("No labeled rows collected. Nothing written.", flush=True)
        return

    out_path = args.output
    out_path_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path_abs) if os.path.dirname(out_path_abs) else ".", exist_ok=True)

    if dataset_columns is None:
        dataset_columns = sorted(list(rows[0].keys()))

    with open(out_path_abs, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dataset_columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote dataset: {out_path_abs} ({len(rows)} rows)", flush=True)
    if not args.derive_labels_from_csv:
        print("Tip: run `python scripts/train_model.py --input <dataset> --input-type dataset ...` next.", flush=True)


if __name__ == "__main__":
    main()

