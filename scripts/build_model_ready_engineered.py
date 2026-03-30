#!/usr/bin/env python3
"""Append engineered columns to model_ready.csv and write model_ready_engineered.csv."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from mindtune_stress_model.model_ready_engineering import engineer_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=os.path.join(_REPO_ROOT, "model_ready.csv"))
    p.add_argument("--output", default=os.path.join(_REPO_ROOT, "model_ready_engineered.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.input)
    before = df.shape[1]
    df = engineer_features(df)
    after = df.shape[1]
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}  rows={len(df)}  cols {before} -> {after} (+{after - before})")


if __name__ == "__main__":
    main()
