"""
train.py
--------
LightGBM training module for flight delay binary classification.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

# Features produced by the FeaturePipeline
FEATURE_COLS = [
    "CRS_DEP_TIME_sin",
    "CRS_DEP_TIME_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Month_sin",
    "Month_cos",
    "departure_hour",
    "route_distance_km",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "OP_CARRIER_te",
    "ORIGIN_te",
    "DEST_te",
    "TAIL_NUM_te",
    "route_delay_rate",
    "airline_delay_rate",
    "is_near_holiday",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature columns that are present in ``df``."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning("Feature columns not found in dataframe: %s", missing)
    logger.info("Using %d feature columns.", len(available))
    return available


def compute_class_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight = count(y=0) / count(y=1)."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        raise ValueError("No positive examples (y=1) found. Cannot compute class weight.")
    weight = n_neg / n_pos
    logger.info("scale_pos_weight = %.3f  (n_neg=%d, n_pos=%d)", weight, n_neg, n_pos)
    return float(weight)


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[dict] = None,
) -> tuple[lgb.Booster, dict]:
    """Train a LightGBM binary classifier.

    Parameters
    ----------
    X_train, y_train : train features and labels
    X_val, y_val : validation features and labels
    params : dict, optional
        LightGBM parameters. Defaults applied when None.

    Returns
    -------
    (model, training_history)
        model : lgb.Booster
        training_history : dict with "pr_auc_val" and "best_iteration"
    """
    scale_pos_weight = compute_class_weight(y_train)

    default_params = {
        "objective":         "binary",
        "metric":            "auc",
        "learning_rate":     0.03,
        "num_leaves":        127,
        "max_depth":         -1,          # unlimited — let num_leaves control complexity
        "min_child_samples": 50,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "lambda_l2":         1.0,
        "scale_pos_weight":  1.0,         # No class weighting — tune threshold instead
        "verbose":           -1,
        "n_jobs":            -1,
        "seed":              42,
    }
    if params is not None:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True, first_metric_only=True),
        lgb.log_evaluation(period=100),
    ]

    logger.info("Starting LightGBM training (num_boost_round=3000, early_stopping=100)...")
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    val_probs = model.predict(X_val)
    pr_auc_val = float(average_precision_score(y_val, val_probs))
    logger.info(
        "Training complete. Best iteration: %d | Val PR-AUC: %.4f",
        model.best_iteration, pr_auc_val,
    )

    history = {
        "best_iteration": model.best_iteration,
        "pr_auc_val": pr_auc_val,
        "params": default_params,
    }
    return model, history


def save_model(
    model: lgb.Booster,
    params: dict,
    metrics: dict,
    artifacts_dir: Path,
    threshold: float = 0.5,
) -> Path:
    """Save model, params, and metrics to ``artifacts_dir``.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path  = artifacts_dir / f"model_{ts}.txt"
    params_path = artifacts_dir / f"params_{ts}.json"

    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    combined = {"threshold": threshold, "metrics": metrics, **params}
    with open(params_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info("Params/metrics saved to %s", params_path)

    # Also write a canonical params.json (latest model wins)
    with open(artifacts_dir / "params.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    return model_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.model.train <data_path.csv> [artifacts_dir]")
        sys.exit(0)

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.loader import load_raw
    from src.data.splitter import temporal_split
    from src.features.pipeline import FeaturePipeline

    data_path     = Path(sys.argv[1])
    artifacts_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("backend/artifacts")

    df = load_raw(data_path)
    train_df, val_df, test_df = temporal_split(df)

    pipeline = FeaturePipeline()
    train_transformed = pipeline.fit_transform(train_df, target_col="y")
    val_transformed   = pipeline.transform(val_df)
    pipeline.save(artifacts_dir)

    feat_cols = get_feature_cols(train_transformed)
    X_train, y_train = train_transformed[feat_cols], train_transformed["y"]
    X_val,   y_val   = val_transformed[feat_cols],   val_transformed["y"]

    model, history = train_lgbm(X_train, y_train, X_val, y_val)
    save_model(model, history.get("params", {}), history, artifacts_dir)
    print("Training complete:", history)
