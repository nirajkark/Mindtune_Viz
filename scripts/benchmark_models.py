from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Allow running without installing the package (pip install -e .).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_EXCLUDE = {
    "label",
    "label_3class",
    "window_start_s",
    "window_end_s",
    "source_folder",
    "source_session_path",
    "session_id",
    "participant_id",
    "timestamp_ms",
    "timestamp",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: Any


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns and "label_3class" not in df.columns:
        raise ValueError("Dataset must contain `label` or `label_3class`.")
    return df


def _pick_label_col(df: pd.DataFrame) -> str:
    return "label" if "label" in df.columns else "label_3class"


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    candidate_cols = [c for c in df.columns if c not in DEFAULT_EXCLUDE]
    numeric = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    feature_cols = [c for c in candidate_cols if numeric[c].notna().any()]
    out_df = df.copy()
    out_df[feature_cols] = numeric[feature_cols]
    return out_df, feature_cols


def _make_models(seed: int) -> List[ModelSpec]:
    models: List[ModelSpec] = []

    # Logistic regression baseline (scaled).
    models.append(
        ModelSpec(
            "logreg",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        )
    )

    # RandomForest / ExtraTrees (no scaling needed).
    models.append(
        ModelSpec(
            "rf_300",
            RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            ),
        )
    )
    models.append(
        ModelSpec(
            "extratrees_600",
            ExtraTreesClassifier(
                n_estimators=600,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            ),
        )
    )

    # GradientBoosting (can perform well on tabular; no class_weight support).
    models.append(
        ModelSpec(
            "gboost",
            GradientBoostingClassifier(random_state=seed),
        )
    )

    # Linear SVM + probability calibration (often strong).
    # LinearSVC doesn't output probabilities; calibrate for a fairer comparison/reporting.
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", random_state=seed)),
        ]
    )
    models.append(ModelSpec("linear_svc_calibrated", CalibratedClassifierCV(svm, cv=3, method="sigmoid")))

    # KNN (scaled).
    models.append(
        ModelSpec(
            "knn_25",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=25)),
                ]
            ),
        )
    )

    # Naive Bayes baseline (expects dense numeric).
    models.append(ModelSpec("gaussian_nb", GaussianNB()))

    return models


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark multiple classifiers with cross-validation.")
    p.add_argument("--input", required=True, help="CSV path (e.g., model_ready_events.csv)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--cv-group-by",
        default=None,
        choices=[None, "session_id", "participant_id", "source_session_path"],
        help="Optional group column for GroupKFold to reduce leakage.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--select-metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy"],
        help="Metric used to select the best model.",
    )
    p.add_argument("--report-out", default="benchmark_report.json")
    p.add_argument("--best-model-out", default="best_model.joblib")
    args = p.parse_args()

    df = _load_dataset(args.input)
    label_col = _pick_label_col(df)
    df, feature_cols = _select_features(df)

    # Drop rows with missing features/labels.
    df = df.dropna(subset=feature_cols + [label_col]).copy()

    y = df[label_col].astype(str).to_numpy()
    X = df[feature_cols].to_numpy(dtype=float)

    # Build splits once, reuse for all models.
    if args.cv_folds <= 1:
        raise ValueError("--cv-folds must be > 1")

    splits: List[Tuple[np.ndarray, np.ndarray]]
    if args.cv_group_by:
        groups = df[args.cv_group_by].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=args.cv_folds)
        splits = list(gkf.split(X, y, groups=groups))
    else:
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(X, y))

    report: Dict[str, Any] = {
        "input": args.input,
        "label_col": label_col,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "cv": {"folds": int(args.cv_folds), "group_by": args.cv_group_by},
        "models": [],
        "selected_metric": args.select_metric,
    }

    models = _make_models(args.seed)

    best_name: Optional[str] = None
    best_score = -1.0
    best_model: Any = None

    for spec in models:
        fold_acc: List[float] = []
        fold_macro_f1: List[float] = []
        fold_reports: List[Dict[str, Any]] = []

        for fold_i, (tr, te) in enumerate(splits, start=1):
            m = spec.model
            # Fit a fresh clone for safety (pipelines/classifiers implement get_params/set_params).
            # For simplicity and reliability here, re-instantiate by using sklearn's cloning if possible.
            from sklearn.base import clone

            model = clone(m)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])

            acc = float(accuracy_score(y[te], pred))
            macro_f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))

            fold_acc.append(acc)
            fold_macro_f1.append(macro_f1)
            fold_reports.append(
                {
                    "fold": fold_i,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "confusion_matrix": confusion_matrix(y[te], pred).tolist(),
                    "classification_report": classification_report(
                        y[te], pred, output_dict=True, zero_division=0
                    ),
                }
            )

        model_summary = {
            "name": spec.name,
            "accuracy_mean": float(np.mean(fold_acc)),
            "accuracy_std": float(np.std(fold_acc)),
            "macro_f1_mean": float(np.mean(fold_macro_f1)),
            "macro_f1_std": float(np.std(fold_macro_f1)),
            "folds": fold_reports,
        }
        report["models"].append(model_summary)

        score = model_summary["macro_f1_mean"] if args.select_metric == "macro_f1" else model_summary["accuracy_mean"]
        if score > best_score:
            best_score = float(score)
            best_name = spec.name
            best_model = spec.model

    # Train the best model on all data and save.
    if best_model is None or best_name is None:
        raise RuntimeError("No models evaluated.")

    from sklearn.base import clone

    final_model = clone(best_model)
    final_model.fit(X, y)

    import joblib

    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "label_col": label_col,
            "best_model_name": best_name,
            "selected_metric": args.select_metric,
        },
        args.best_model_out,
    )

    report["best"] = {"name": best_name, "score": best_score}
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Best model:", best_name)
    print(f"Best {args.select_metric}:", best_score)
    print("Saved report:", args.report_out)
    print("Saved best model bundle:", args.best_model_out)


if __name__ == "__main__":
    main()

