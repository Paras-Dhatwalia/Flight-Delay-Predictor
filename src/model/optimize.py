"""
optimize.py
-----------
Optuna-based hyperparameter tuning for the LightGBM flight delay classifier.

Primary objective: maximise PR-AUC on the validation set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging; we emit our own INFO log instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> float:
    """Single Optuna trial objective — returns validation PR-AUC.

    Parameters
    ----------
    trial : optuna.Trial
    X_train, y_train : training data
    X_val, y_val : validation data
    scale_pos_weight : float
        Imbalance weight = count(y=0) / count(y=1).

    Returns
    -------
    float
        PR-AUC on validation set (higher is better).
    """
    params = {
        "objective":         "binary",
        "metric":            "binary_logloss",
        "verbose":           -1,
        "n_jobs":            -1,
        "seed":              42,
        "scale_pos_weight":  scale_pos_weight,
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "max_depth":         trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.6, 0.9),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "lambda_l1":         trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2":         trial.suggest_float("lambda_l2", 0.0, 5.0),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    val_probs = model.predict(X_val)
    pr_auc = float(average_precision_score(y_val, val_probs))
    return pr_auc


def run_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    artifacts_dir: Optional[Path] = None,
) -> dict:
    """Run Optuna hyperparameter search and return the best parameters.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    n_trials : int
        Number of Optuna trials (default 50; set to 100 for better coverage).
    artifacts_dir : Path, optional
        If provided, save ``best_params.json`` here.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("scale_pos_weight = %.3f", scale_pos_weight)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="flight_delay_lgbm",
    )

    logger.info("Starting Optuna tuning: %d trials | objective=PR-AUC", n_trials)

    def _objective_wrapper(trial):
        pr_auc = objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight)
        if trial.number % 10 == 0 or trial.number == 0:
            logger.info(
                "Trial %3d | PR-AUC=%.4f | Best so far=%.4f",
                trial.number, pr_auc, study.best_value if study.best_trial else pr_auc,
            )
        return pr_auc

    study.optimize(_objective_wrapper, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = scale_pos_weight
    logger.info(
        "Tuning complete. Best PR-AUC=%.4f at trial %d.",
        study.best_value, study.best_trial.number,
    )
    logger.info("Best params: %s", best_params)

    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_path = artifacts_dir / "best_params.json"
        with open(out_path, "w") as f:
            json.dump({"best_pr_auc": study.best_value, **best_params}, f, indent=2)
        logger.info("Best params saved to %s", out_path)

    return best_params


def get_best_params(artifacts_dir: Path) -> dict:
    """Load best hyperparameters from a previously saved ``best_params.json``.

    Parameters
    ----------
    artifacts_dir : Path
        Directory containing ``best_params.json``.

    Returns
    -------
    dict
        Loaded parameter dictionary.
    """
    path = Path(artifacts_dir) / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"best_params.json not found in {artifacts_dir}. "
            "Run run_tuning() first."
        )
    with open(path) as f:
        params = json.load(f)
    logger.info("Loaded best params from %s: %s", path, params)
    return params


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.model.optimize <data_path.csv> [n_trials] [artifacts_dir]")
        sys.exit(0)

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.loader import load_raw
    from src.data.splitter import temporal_split
    from src.features.pipeline import FeaturePipeline
    from src.model.train import get_feature_cols

    data_path     = Path(sys.argv[1])
    n_trials      = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    artifacts_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("backend/artifacts")

    df = load_raw(data_path)
    train_df, val_df, _ = temporal_split(df)
    pipeline = FeaturePipeline()
    train_t = pipeline.fit_transform(train_df, target_col="y")
    val_t   = pipeline.transform(val_df)

    feat_cols = get_feature_cols(train_t)
    best = run_tuning(
        train_t[feat_cols], train_t["y"],
        val_t[feat_cols],   val_t["y"],
        n_trials=n_trials,
        artifacts_dir=artifacts_dir,
    )
    print("Best params:", best)
