"""
evaluate.py
-----------
Comprehensive model evaluation for flight delay binary classification.

Primary metric: PR-AUC (Precision-Recall AUC), chosen over ROC-AUC because
flight delays are an imbalanced class (~20% positive rate).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> dict:
    """Evaluate a LightGBM model and return a full metrics dict.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM booster.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True binary labels.
    threshold : float
        Decision threshold for binary predictions.
    output_dir : Path, optional
        If provided, save all plots here.

    Returns
    -------
    dict
        Keys: pr_auc, roc_auc, f1, precision_at_50_recall,
              recall_at_80_precision, brier_score, confusion_matrix,
              classification_report, optimal_threshold.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= threshold).astype(int)

    # --- Core metrics ---
    pr_auc  = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    f1      = float(f1_score(y_test, y_pred))
    brier   = float(brier_score_loss(y_test, y_prob))
    cm      = confusion_matrix(y_test, y_pred).tolist()
    cr      = classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"])

    # --- Precision @ 50% Recall ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    precision_at_50r = float(np.interp(0.5, recalls[::-1], precisions[::-1]))

    # --- Recall @ 80% Precision ---
    valid_mask = precisions >= 0.80
    recall_at_80p = float(recalls[valid_mask].max()) if valid_mask.any() else 0.0

    # --- Optimal threshold (maximises F1) ---
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / np.maximum(precisions[:-1] + recalls[:-1], 1e-8)
    )
    optimal_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[optimal_idx])

    metrics = {
        "pr_auc":                 round(pr_auc, 6),
        "roc_auc":                round(roc_auc, 6),
        "f1":                     round(f1, 6),
        "precision_at_50_recall": round(precision_at_50r, 6),
        "recall_at_80_precision": round(recall_at_80p, 6),
        "brier_score":            round(brier, 6),
        "confusion_matrix":       cm,
        "classification_report":  cr,
        "optimal_threshold":      round(optimal_threshold, 4),
        "threshold_used":         threshold,
    }

    logger.info(
        "Evaluation results: PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | Brier=%.4f | "
        "Optimal threshold=%.3f",
        pr_auc, roc_auc, f1, brier, optimal_threshold,
    )
    logger.info("\n%s", cr)

    if output_dir is not None:
        plot_precision_recall_curve(y_test, y_prob, output_dir)
        plot_roc_curve(y_test, y_prob, output_dir)
        plot_confusion_matrix(y_test, y_pred, output_dir)
        plot_calibration_curve(y_test, y_prob, output_dir)

    return metrics


def plot_precision_recall_curve(
    y_true, y_prob, output_dir: Optional[Path] = None
) -> None:
    """Plot and save the Precision-Recall curve."""
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, name=f"LightGBM (PR-AUC={pr_auc:.3f})", ax=ax,
    )
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "pr_curve.png", dpi=150)
        logger.info("Saved pr_curve.png")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, output_dir: Optional[Path] = None) -> None:
    """Plot and save the ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "roc_curve.png", dpi=150)
        logger.info("Saved roc_curve.png")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, output_dir: Optional[Path] = None) -> None:
    """Plot and save the confusion matrix heatmap."""
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                       for row_v, row_p in zip(cm, cm_pct)])
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", ax=ax,
        xticklabels=["On-Time", "Delayed"],
        yticklabels=["On-Time", "Delayed"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "confusion_matrix.png", dpi=150)
        logger.info("Saved confusion_matrix.png")
    plt.close(fig)


def plot_calibration_curve(
    y_true, y_prob, output_dir: Optional[Path] = None, n_bins: int = 10
) -> None:
    """Plot and save the calibration curve."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", label="LightGBM")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / "calibration_curve.png", dpi=150)
        logger.info("Saved calibration_curve.png")
    plt.close(fig)


def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    output_dir: Optional[Path] = None,
    max_display: int = 20,
) -> np.ndarray:
    """Generate SHAP summary (beeswarm) and bar plots.

    Returns
    -------
    np.ndarray
        SHAP values array of shape (n_samples, n_features).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # LightGBM binary may return list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, max_display=max_display, show=False)
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / "shap_summary.png", dpi=150, bbox_inches="tight")
        logger.info("Saved shap_summary.png")
    plt.close()

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_test, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / "shap_bar.png", dpi=150, bbox_inches="tight")
        logger.info("Saved shap_bar.png")
    plt.close()

    return sv


def get_top_shap_factors(
    model, X_row: pd.DataFrame, n: int = 5
) -> list[dict]:
    """Compute top-n SHAP factors for a single prediction row.

    Parameters
    ----------
    model : lgb.Booster
    X_row : pd.DataFrame
        Single-row feature DataFrame.
    n : int
        Number of top factors to return.

    Returns
    -------
    list[dict]
        ``[{"feature": name, "impact": float}, ...]`` sorted by |impact| desc.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):
        sv = np.array(shap_vals[1][0])
    else:
        sv = np.array(shap_vals[0])

    feature_names = list(X_row.columns)
    pairs = sorted(zip(feature_names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": name, "impact": round(impact, 6)} for name, impact in pairs[:n]]


def plot_partial_dependence(
    model,
    X_test: pd.DataFrame,
    top_n: int = 3,
    output_dir: Optional[Path] = None,
) -> None:
    """Plot partial dependence plots for the top-n SHAP-important features."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [X_test.columns[i] for i in top_idx]

    for feat in top_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(feat, sv, X_test, ax=ax, show=False)
        ax.set_title(f"Partial Dependence: {feat}")
        plt.tight_layout()
        if output_dir:
            safe_name = feat.replace("/", "_").replace(" ", "_")
            fig.savefig(Path(output_dir) / f"pdp_{safe_name}.png", dpi=150)
            logger.info("Saved pdp_%s.png", safe_name)
        plt.close(fig)
