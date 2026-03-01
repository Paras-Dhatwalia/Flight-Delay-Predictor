#!/usr/bin/env python3
"""
run_training.py
===============
End-to-end training pipeline for the Flight Delay Prediction system.

Usage:
    python run_training.py

This script:
  1. Loads all 12 monthly BTS CSV files from Dataset/extracted/
  2. Splits into train (Jan-Aug) / val (Sep-Oct) / test (Nov-Dec)
  3. Runs the full feature engineering pipeline
  4. Runs leakage audit
  5. Trains LightGBM with default params (or Optuna-tuned params if available)
  6. Evaluates on test set, saves all plots and metrics
  7. Saves model + pipeline artifacts to backend/artifacts/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.loader import load_multiple
from src.data.splitter import temporal_split, get_split_stats
from src.features.pipeline import FeaturePipeline
from src.model.train import train_lgbm, save_model, get_feature_cols
from src.model.evaluate import evaluate_model, plot_shap_summary
from src.utils.leakage_check import run_leakage_audit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "Dataset" / "extracted"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("run_training")


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  FLIGHT DELAY PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading BTS data from %s", DATA_DIR)
    csv_files = sorted(DATA_DIR.glob("month_*.csv"))
    if not csv_files:
        logger.error("No month_*.csv files found in %s", DATA_DIR)
        sys.exit(1)
    logger.info("Found %d CSV files: %s", len(csv_files), [f.name for f in csv_files])

    df = load_multiple(csv_files)
    logger.info("Combined dataset: %d rows, %d columns", len(df), df.shape[1])
    logger.info("Columns: %s", list(df.columns))

    # ------------------------------------------------------------------
    # Step 2: Temporal split
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2] Temporal train/val/test split")
    train_df, val_df, test_df = temporal_split(df)
    stats = get_split_stats(train_df, val_df, test_df)
    for name, s in stats.items():
        logger.info("  %s: %d rows, delay rate: %.4f", name, s["count"], s["delay_rate"])

    # Free memory
    del df

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3] Feature engineering pipeline")
    pipeline = FeaturePipeline()
    train_feat = pipeline.fit_transform(train_df, target_col="y")
    val_feat = pipeline.transform(val_df)
    test_feat = pipeline.transform(test_df)

    # Save pipeline to artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.save(ARTIFACTS_DIR)
    logger.info("Pipeline saved to %s", ARTIFACTS_DIR)

    # ------------------------------------------------------------------
    # Step 4: Leakage audit
    # ------------------------------------------------------------------
    logger.info("\n[STEP 4] Leakage audit")
    audit = run_leakage_audit(train_feat, target_col="y")
    if not audit["all_passed"]:
        logger.error("LEAKAGE DETECTED — aborting training.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Prepare feature matrices
    # ------------------------------------------------------------------
    logger.info("\n[STEP 5] Preparing feature matrices")
    feat_cols = get_feature_cols(train_feat)
    logger.info("Feature columns (%d): %s", len(feat_cols), feat_cols)

    X_train = train_feat[feat_cols]
    y_train = train_feat["y"]
    X_val = val_feat[feat_cols]
    y_val = val_feat["y"]
    X_test = test_feat[feat_cols]
    y_test = test_feat["y"]

    logger.info("X_train: %s, X_val: %s, X_test: %s", X_train.shape, X_val.shape, X_test.shape)

    # Check for NaN features
    nan_counts = X_train.isna().sum()
    if nan_counts.any():
        logger.warning("NaN counts in training features:\n%s", nan_counts[nan_counts > 0])
        # Fill NaN with median for training
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)
        X_test = X_test.fillna(medians)
        logger.info("Filled NaN with training medians.")

    # ------------------------------------------------------------------
    # Step 6: Train LightGBM
    # ------------------------------------------------------------------
    logger.info("\n[STEP 6] Training LightGBM")

    # Check if Optuna best_params exist
    best_params_path = ARTIFACTS_DIR / "best_params.json"
    params = None
    if best_params_path.exists():
        with open(best_params_path) as f:
            params = json.load(f)
        # Remove non-LightGBM keys
        params.pop("best_pr_auc", None)
        logger.info("Using Optuna best params: %s", params)

    model, history = train_lgbm(X_train, y_train, X_val, y_val, params=params)

    # ------------------------------------------------------------------
    # Step 7: Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("\n[STEP 7] Evaluating on test set")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use optimal threshold from PR curve
    from sklearn.metrics import precision_recall_curve
    y_prob_val = model.predict(X_val)
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob_val)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
    optimal_threshold = float(thresholds[np.argmax(f1_scores)])
    logger.info("Optimal threshold (F1-maximizing on val): %.4f", optimal_threshold)

    metrics = evaluate_model(
        model, X_test, y_test,
        threshold=optimal_threshold,
        output_dir=REPORTS_DIR,
    )
    logger.info("Test PR-AUC: %.4f", metrics["pr_auc"])
    logger.info("Test ROC-AUC: %.4f", metrics["roc_auc"])
    logger.info("Test F1: %.4f", metrics["f1"])
    logger.info("Brier Score: %.4f", metrics["brier_score"])

    # Save metrics
    with open(REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Step 8: SHAP analysis
    # ------------------------------------------------------------------
    logger.info("\n[STEP 8] SHAP analysis (on 5000-row sample)")
    sample_size = min(5000, len(X_test))
    X_shap = X_test.iloc[:sample_size]
    try:
        plot_shap_summary(model, X_shap, output_dir=REPORTS_DIR, max_display=17)
    except Exception as exc:
        logger.warning("SHAP plotting failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 9: Save model
    # ------------------------------------------------------------------
    logger.info("\n[STEP 9] Saving model artifacts")
    model_path = save_model(
        model,
        params=history.get("params", {}),
        metrics=metrics,
        artifacts_dir=ARTIFACTS_DIR,
        threshold=optimal_threshold,
    )
    logger.info("Model saved to %s", model_path)

    # Save feature columns list for backend reference
    with open(ARTIFACTS_DIR / "feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE — %.1f minutes", elapsed / 60)
    logger.info("  Artifacts: %s", ARTIFACTS_DIR)
    logger.info("  Reports:   %s", REPORTS_DIR)
    logger.info("  PR-AUC:    %.4f", metrics["pr_auc"])
    logger.info("  Threshold: %.4f", optimal_threshold)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
