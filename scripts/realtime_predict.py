from __future__ import annotations

import argparse
import sys
import time
from typing import Dict

import numpy as np

from mindtune_stress_model.modeling import ProbabilitySmoother, load_bundle
from mindtune_stress_model.stream_adapter import CallbackSdkStreamAdapter, CsvPlaybackStreamAdapter
from mindtune_stress_model.windowing import SlidingWindowBuffer


def _build_backend(args):
    if args.backend == "mock_csv":
        return CsvPlaybackStreamAdapter(
            args.csv_path,
            speed=args.speed,
            sleep=not args.no_sleep,
        )

    raise NotImplementedError(
        "SDK backend is not wired without your Mindtune SDK callback details. "
        "Use `--backend mock_csv` first."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--backend", default="mock_csv", choices=["mock_csv", "sdk_callback"])
    parser.add_argument("--csv-path", default="Downloads/mindtune_full_eeg_data.csv")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--ema-alpha", type=float, default=0.6, help="EMA smoothing alpha in (0,1]")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed for mock_csv")
    parser.add_argument("--no-sleep", action="store_true", help="Disable sleeping during mock playback")
    parser.add_argument("--print-every", type=int, default=1, help="Print every N windows")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    smoother = ProbabilitySmoother(alpha=args.ema_alpha)

    stream = _build_backend(args)
    buffer = SlidingWindowBuffer(window_seconds=args.window_seconds, step_seconds=None)

    latest_label = None
    latest_time = None

    for i, window in enumerate(buffer_window_iter(stream, buffer), start=1):
        probs = bundle.model.predict_proba(
            np.array([[window.features.get(col, 0.0) for col in bundle.feature_columns]], dtype=float)
        )[0]

        ema_probs = smoother.update(probs)
        best_idx = int(np.argmax(ema_probs))
        label = str(bundle.label_columns[best_idx])

        if (i % args.print_every) == 0:
            latest_label = label
            latest_time = window.window_end_s
            prob_str = ", ".join([f"{bundle.label_columns[j]}={ema_probs[j]:.2f}" for j in range(len(bundle.label_columns))])
            print(f"[t={window.window_end_s:.2f}s] {label} | {prob_str}")

    if latest_label is not None:
        print(f"Final state: {latest_label} (window_end={latest_time:.2f}s)")


def buffer_window_iter(stream, buffer):
    for sample in stream:
        win = buffer.add_sample(sample)
        if win is not None:
            yield win


if __name__ == "__main__":
    main()

