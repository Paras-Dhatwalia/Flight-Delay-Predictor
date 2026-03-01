"""
target_encoding.py
------------------
K-Fold smoothed target encoding for high-cardinality categorical features.

Encoding formula (Laplace / James-Stein smoothing):
    TE(c) = (n_c * p_c  +  λ * p_global) / (n_c + λ)

Where:
    p_c      = delay rate of category c in the training fold
    p_global = overall delay rate in the training fold
    n_c      = number of observations in category c
    λ        = smoothing factor (higher → stronger shrinkage toward global mean)

Using out-of-fold predictions prevents target leakage: each row's encoding
is derived only from folds that did NOT include that row.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

DEFAULT_ENCODE_COLS: List[str] = ["OP_CARRIER", "ORIGIN", "DEST", "TAIL_NUM"]


class KFoldTargetEncoder:
    """Out-of-fold smoothed target encoder for categorical columns.

    Parameters
    ----------
    cols : list[str]
        Categorical column names to encode.
    n_splits : int
        Number of cross-validation folds (default 5).
    smoothing : float
        Laplace smoothing factor λ (default 20.0).
    random_state : int
        Random seed (used only when shuffle=True, kept False here for
        temporal ordering).
    """

    def __init__(
        self,
        cols: List[str] = None,
        n_splits: int = 5,
        smoothing: float = 20.0,
        random_state: int = 42,
    ) -> None:
        self.cols = cols if cols is not None else DEFAULT_ENCODE_COLS
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

        # Populated during fit_transform; used by transform()
        self._global_means: Dict[str, float] = {}
        self._category_means: Dict[str, Dict[str, float]] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth_encode(
        self,
        series: pd.Series,
        target: pd.Series,
        global_mean: float,
    ) -> Dict[str, float]:
        """Compute smoothed target-encoding mapping for a single column on one fold."""
        stats = (
            pd.DataFrame({"cat": series, "target": target})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smoothed = (
            (stats["count"] * stats["mean"] + self.smoothing * global_mean)
            / (stats["count"] + self.smoothing)
        )
        return smoothed.to_dict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
        """Compute out-of-fold target encodings and add them to ``df``.

        KFold with shuffle=False preserves temporal ordering.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe containing ``target_col`` and all cols in ``self.cols``.
        target_col : str
            Binary (0/1) delay target column (default ``"y"``).

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with ``{col}_te`` columns appended.
        """
        missing_cols = [c for c in self.cols if c not in df.columns]
        if missing_cols:
            logger.warning("Target encoder: columns not found, skipping: %s", missing_cols)
        encode_cols = [c for c in self.cols if c in df.columns]

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")

        df = df.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        target = df[target_col].astype(float)
        idx = np.arange(len(df))

        for col in encode_cols:
            df[f"{col}_te"] = np.nan

        self._global_means = {}
        self._category_means = {col: {} for col in encode_cols}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(idx)):
            train_target = target.iloc[train_idx]
            global_mean = float(train_target.mean())

            for col in encode_cols:
                encoding_map = self._smooth_encode(
                    df[col].iloc[train_idx], train_target, global_mean
                )
                encoded_vals = df[col].iloc[val_idx].map(encoding_map).fillna(global_mean)
                df.loc[df.index[val_idx], f"{col}_te"] = encoded_vals.values
                logger.debug(
                    "Fold %d | col='%s' | categories=%d | global_mean=%.4f",
                    fold_idx, col, len(encoding_map), global_mean,
                )

        # Fit production encodings on FULL training set for val/test/inference
        full_global_mean = float(target.mean())
        for col in encode_cols:
            self._global_means[col] = full_global_mean
            self._category_means[col] = self._smooth_encode(df[col], target, full_global_mean)

        self._is_fitted = True
        logger.info(
            "KFoldTargetEncoder fit_transform complete. Cols: %s. Folds: %d. λ=%.1f.",
            encode_cols, self.n_splits, self.smoothing,
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encodings to val/test/production data.

        Unseen categories fall back to the global training mean.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with ``{col}_te`` columns appended.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "KFoldTargetEncoder.transform() called before fit_transform()."
            )
        df = df.copy()
        for col, category_map in self._category_means.items():
            if col not in df.columns:
                logger.warning("Column '%s' not found during transform; skipping.", col)
                continue
            global_mean = self._global_means[col]
            df[f"{col}_te"] = df[col].map(category_map).fillna(global_mean)
        return df

    def save(self, path: str | Path) -> None:
        """Persist the fitted encoder to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("KFoldTargetEncoder saved to '%s'.", path)

    @classmethod
    def load(cls, path: str | Path) -> "KFoldTargetEncoder":
        """Load a saved encoder from disk."""
        encoder: "KFoldTargetEncoder" = joblib.load(Path(path))
        logger.info("KFoldTargetEncoder loaded from '%s'.", path)
        return encoder
