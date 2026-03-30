from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

# Allow running this script without installing the package (pip install -e .).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mindtune_stress_model.stream_adapter import CANONICAL_BANDS
from mindtune_stress_model.modeling import LABELS, evaluate_model, fit_model


def _derive_label_from_session_meta(row: pd.Series) -> str:
    # Your data is one-hot-ish across emotions: emo_calm, emo_stressed, emo_delighted, emo_angry, emo_unknown.
    # We compress everything that's not calm/stressed into "neutral".
    if int(row.get("emo_stressed", 0) or 0) == 1:
        return "stressed"
    if int(row.get("emo_calm", 0) or 0) == 1:
        return "calm"
    return "neutral"


def _compute_relative_band_pcts_from_row(row: pd.Series) -> Dict[str, float]:
    # Canonical names used in our model: Delta, Theta, Low_Alpha, High_Alpha, Low_Beta, High_Beta, Low_Gamma, Mid_Gamma.
    # Your CSV uses lowercase variants: delta, low_alpha, etc.
    band_values: Dict[str, float] = {}
    total = 0.0
    for canonical_band in CANONICAL_BANDS:
        # Derive expected lowercase column name.
        # E.g. "Low_Alpha" -> "low_alpha"
        col = canonical_band.lower()
        col = col.replace("__", "_")
        val = float(row.get(col, 0.0) or 0.0)
        band_values[canonical_band] = val
        total += val

    if total <= 0:
        return {f"{b}_pct": 0.0 for b in CANONICAL_BANDS}

    return {f"{b}_pct": float(band_values[b]) / total for b in CANONICAL_BANDS}


def _load_training_dataframe(input_path: str, input_type: str) -> Tuple[pd.DataFrame, List[str], str]:
    if input_type == "dataset":
        df = pd.read_csv(input_path)
        # Accept either `label` or our EDA export column `label_3class`.
        if "label" in df.columns:
            label_col = "label"
        elif "label_3class" in df.columns:
            label_col = "label_3class"
        else:
            raise ValueError("Dataset input must have a `label` or `label_3class` column.")
        # Feature columns: keep only numeric-ish columns, exclude IDs/strings.
        # This makes `model_ready*.csv` work out of the box.
        exclude = {
            "label",
            "label_3class",
            "window_start_s",
            "window_end_s",
            # common IDs / traceability columns
            "source_folder",
            "source_session_path",
            "session_id",
            "participant_id",
            "timestamp_ms",
            "timestamp",
        }

        candidate_cols = [c for c in df.columns if c not in exclude]
        # Keep numeric dtypes; also allow numeric strings by coercing.
        numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
        feature_cols = [c for c in candidate_cols if numeric_df[c].notna().any()]
        # Replace columns with coerced numeric versions so downstream .to_numpy(dtype=float) works.
        df[feature_cols] = numeric_df[feature_cols]
        return df, feature_cols, label_col

    if input_type == "session_meta":
        df = pd.read_csv(input_path)
        label_col = "label"
        df[label_col] = df.apply(_derive_label_from_session_meta, axis=1)

        # Build relative band pct features + include attention/meditation (if present).
        band_pct_cols = [f"{b}_pct" for b in CANONICAL_BANDS]
        feature_cols = band_pct_cols[:]
        if "attention" not in df.columns and "Attention" in df.columns:
            # normalize attention col name
            df = df.rename(columns={"Attention": "attention"})
        if "meditation" not in df.columns and "Meditation" in df.columns:
            df = df.rename(columns={"Meditation": "meditation"})

        # Session_meta columns are lowercase (as seen in your CSV): attention, meditation.
        for bpc in band_pct_cols:
            df[bpc] = 0.0

        # Vectorized compute for band pcts.
        # For robustness: use row-wise for now (dataset size is manageable).
        df[band_pct_cols] = df.apply(lambda r: pd.Series(_compute_relative_band_pcts_from_row(r)), axis=1)

        # Optional features.
        if "attention" in df.columns:
            df["Attention"] = pd.to_numeric(df["attention"], errors="coerce").fillna(0.0)
            feature_cols.append("Attention")
        if "meditation" in df.columns:
            df["Meditation"] = pd.to_numeric(df["meditation"], errors="coerce").fillna(0.0)
            feature_cols.append("Meditation")

        return df, feature_cols, label_col

    raise ValueError("input_type must be one of: dataset, session_meta")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV dataset")
    parser.add_argument("--input-type", default="session_meta", choices=["dataset", "session_meta"])
    parser.add_argument("--model-out", default="model.joblib")
    parser.add_argument("--model-type", default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >1, run K-fold CV (and then train final model on all data).",
    )
    parser.add_argument(
        "--cv-group-by",
        default=None,
        choices=[None, "session_id", "participant_id", "source_session_path"],
        help="Optional group column for GroupKFold to reduce leakage.",
    )
    parser.add_argument("--calibrate-proba", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", default="metrics.json")
    args = parser.parse_args()

    df, feature_cols, label_col = _load_training_dataframe(args.input, args.input_type)
    df = df.dropna(subset=feature_cols + [label_col]).copy()

    def _run_cross_val(
        df_in: pd.DataFrame, *, folds: int, group_by: Optional[str]
    ) -> Dict[str, Any]:
        if folds <= 1:
            raise ValueError("folds must be > 1 for cross-validation")

        y = df_in[label_col].astype(str).to_numpy()
        X_df = df_in[feature_cols + [label_col]].copy()

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        if group_by:
            if group_by not in df_in.columns:
                raise ValueError(f"cv_group_by column not found: {group_by}")
            groups = df_in[group_by].astype(str).to_numpy()
            gkf = GroupKFold(n_splits=folds)
            splits = list(gkf.split(X_df, y, groups=groups))
        else:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
            splits = list(skf.split(X_df, y))

        fold_metrics: List[Dict[str, Any]] = []
        accuracies: List[float] = []

        for i, (train_idx, test_idx) in enumerate(splits, start=1):
            train_df = df_in.iloc[train_idx].copy()
            test_df = df_in.iloc[test_idx].copy()
            bundle = fit_model(
                train_df,
                label_col=label_col,
                feature_cols=feature_cols,
                model_type=args.model_type,
                calibrate=args.calibrate_proba,
                random_state=args.seed,
            )
            m = evaluate_model(test_df, bundle, label_col=label_col)
            m["fold"] = i
            m["train_rows"] = int(len(train_df))
            m["test_rows"] = int(len(test_df))
            fold_metrics.append(m)
            accuracies.append(float(m["accuracy"]))

        macro_f1s: List[float] = []
        for fm in fold_metrics:
            try:
                macro_f1s.append(float(fm["classification_report"]["macro avg"]["f1-score"]))
            except Exception:
                pass

        return {
            "cv": {
                "folds": int(folds),
                "group_by": group_by,
                "accuracies": accuracies,
                "accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
                "accuracy_std": float(np.std(accuracies)) if accuracies else None,
                "macro_f1s": macro_f1s,
                "macro_f1_mean": float(np.mean(macro_f1s)) if macro_f1s else None,
                "macro_f1_std": float(np.std(macro_f1s)) if macro_f1s else None,
                "fold_metrics": fold_metrics,
            }
        }

    # Time-aware split if timestamp_ms/window_end_s exists.
    sort_col = None
    for c in ["timestamp_ms", "window_end_s"]:
        if c in df.columns:
            sort_col = c
            break
    if sort_col is not None:
        df = df.sort_values(sort_col)

    metrics: Dict[str, Any]

    # If requested, run cross-validation first.
    if args.cv_folds and args.cv_folds > 1:
        metrics = _run_cross_val(df, folds=args.cv_folds, group_by=args.cv_group_by)
        # Train final model on all rows for saving.
        bundle = fit_model(
            df,
            label_col=label_col,
            feature_cols=feature_cols,
            model_type=args.model_type,
            calibrate=args.calibrate_proba,
            random_state=args.seed,
        )
    else:
        n_test = max(1, int(len(df) * args.test_fraction))
        train_df = df.iloc[:-n_test]
        test_df = df.iloc[-n_test:]

        bundle = fit_model(
            train_df,
            label_col=label_col,
            feature_cols=feature_cols,
            model_type=args.model_type,
            calibrate=args.calibrate_proba,
            random_state=args.seed,
        )

        metrics = evaluate_model(test_df, bundle, label_col=label_col)

    # Persist.
    out_dir = os.path.dirname(os.path.abspath(args.model_out))
    os.makedirs(out_dir, exist_ok=True)
    bundle.save(args.model_out)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {args.model_out}")
    if "accuracy" in metrics:
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
    elif "cv" in metrics:
        print(
            f"CV accuracy mean±std: {metrics['cv']['accuracy_mean']:.4f}±{metrics['cv']['accuracy_std']:.4f}"
        )
        if metrics["cv"].get("macro_f1_mean") is not None:
            print(
                f"CV macro-F1 mean±std: {metrics['cv']['macro_f1_mean']:.4f}±{metrics['cv']['macro_f1_std']:.4f}"
            )
    print(f"Labels: {bundle.label_columns}")


if __name__ == "__main__":
    main()

