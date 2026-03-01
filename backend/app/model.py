"""
model.py
--------
LightGBM model loading and inference with SHAP explanations.

Provides a singleton FlightDelayModel that lazy-loads from the artifacts
directory on first use.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("lightgbm is required: pip install lightgbm") from exc

try:
    import joblib
except ImportError as exc:
    raise ImportError("joblib is required: pip install joblib") from exc

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed; SHAP factors will be unavailable.")

RISK_LOW_THRESHOLD  = 0.3
RISK_HIGH_THRESHOLD = 0.6
TOP_N_FACTORS = 5


class FlightDelayModel:
    """Wraps a trained LightGBM model + feature pipeline for inference.

    Typical artifacts directory layout::

        artifacts/
            model_20260101_120000.txt     # LightGBM text model (latest used)
            pipeline_20260101_120000.pkl  # Feature pipeline (latest used)
            params.json                   # {"threshold": 0.45, ...}
    """

    _MODEL_GLOB    = "model_*.txt"
    _PIPELINE_GLOB = "pipeline*.pkl"

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.threshold: float = 0.5
        self.version: str = "unknown"
        self.is_loaded: bool = False

        self._model: Optional[lgb.Booster]  = None
        self._pipeline                       = None
        self._explainer                      = None
        self._feature_names: list[str]       = []

        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _latest_file(self, glob: str) -> Optional[Path]:
        candidates = sorted(
            self.artifacts_dir.glob(glob),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _load(self) -> None:
        """Load all artifacts. Sets is_loaded=True only on full success."""
        if not self.artifacts_dir.exists():
            logger.error("Artifacts directory not found: %s", self.artifacts_dir)
            return

        # params.json (optional — provides threshold override)
        params_path = self.artifacts_dir / "params.json"
        if params_path.exists():
            try:
                with params_path.open() as fh:
                    params = json.load(fh)
                self.threshold = float(params.get("threshold", self.threshold))
                logger.info("Loaded params.json; threshold=%.3f", self.threshold)
            except Exception as exc:
                logger.warning("Could not read params.json: %s", exc)

        # LightGBM model
        model_path = self._latest_file(self._MODEL_GLOB)
        if model_path is None:
            logger.error("No model file matching '%s' in %s", self._MODEL_GLOB, self.artifacts_dir)
            return
        try:
            self._model = lgb.Booster(model_file=str(model_path))
            self._feature_names = self._model.feature_name()
            ts_match = re.search(r"model_(.+)\.txt$", model_path.name)
            self.version = ts_match.group(1) if ts_match else model_path.stem
            logger.info("Loaded LightGBM model: %s", model_path.name)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return

        # Feature pipeline
        pipeline_path = self._latest_file(self._PIPELINE_GLOB)
        if pipeline_path is None:
            logger.error("No pipeline file matching '%s' in %s", self._PIPELINE_GLOB, self.artifacts_dir)
            return
        try:
            self._pipeline = joblib.load(pipeline_path)
            logger.info("Loaded pipeline: %s", pipeline_path.name)
        except Exception as exc:
            logger.error("Failed to load pipeline: %s", exc)
            return

        # SHAP explainer (non-fatal if unavailable)
        if _SHAP_AVAILABLE:
            try:
                self._explainer = shap.TreeExplainer(self._model)
                logger.info("SHAP TreeExplainer initialised.")
            except Exception as exc:
                logger.warning("SHAP explainer init failed: %s", exc)

        self.is_loaded = True
        logger.info("FlightDelayModel ready | version=%s | threshold=%.3f", self.version, self.threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, raw_df: pd.DataFrame
    ) -> tuple[float, int, str, list[dict]]:
        """Run inference on a raw (pre-pipeline) feature DataFrame.

        Returns
        -------
        (probability, prediction, risk_level, top_factors)
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Check artifacts directory.")

        # Apply feature pipeline
        try:
            transformed = self._pipeline.transform(raw_df)
        except Exception as exc:
            raise ValueError(f"Feature pipeline transform failed: {exc}") from exc

        # Select and align feature columns
        feat_df = self._align_features(transformed)

        # Probability
        raw_probs = self._model.predict(feat_df)
        probability = float(np.clip(raw_probs[0], 0.0, 1.0))

        prediction = int(probability >= self.threshold)
        risk_level = self._risk_level(probability)
        top_factors = self._shap_factors(feat_df)

        return probability, prediction, risk_level, top_factors

    def get_version(self) -> str:
        return self.version

    # ------------------------------------------------------------------
    # Private inference helpers
    # ------------------------------------------------------------------

    def _align_features(self, transformed: pd.DataFrame) -> pd.DataFrame:
        """Ensure the transformed DataFrame has exactly the model's feature columns."""
        if self._feature_names and all(f in transformed.columns for f in self._feature_names):
            return transformed[self._feature_names]
        if self._feature_names and len(transformed.columns) == len(self._feature_names):
            df = transformed.copy()
            df.columns = self._feature_names
            return df
        logger.warning(
            "Feature count mismatch: transformed=%d, model expects=%d.",
            len(transformed.columns), len(self._feature_names),
        )
        return transformed

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < RISK_LOW_THRESHOLD:
            return "Low"
        if probability <= RISK_HIGH_THRESHOLD:
            return "Medium"
        return "High"

    def _shap_factors(self, feat_df: pd.DataFrame) -> list[dict]:
        if self._explainer is None:
            return []
        try:
            shap_vals = self._explainer.shap_values(feat_df)
            if isinstance(shap_vals, list):
                sv = np.array(shap_vals[1][0])
            else:
                sv = np.array(shap_vals[0])
            names = list(feat_df.columns) if len(feat_df.columns) else [f"f{i}" for i in range(len(sv))]
            pairs = sorted(zip(names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
            return [{"feature": n, "impact": round(v, 6)} for n, v in pairs[:TOP_N_FACTORS]]
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_model_instance: Optional[FlightDelayModel] = None


def get_model() -> FlightDelayModel:
    """Return the global FlightDelayModel singleton (lazy-loaded)."""
    global _model_instance
    if _model_instance is None:
        artifacts_dir = os.environ.get("ARTIFACTS_DIR", "/app/artifacts")
        logger.info("Loading model from: %s", artifacts_dir)
        _model_instance = FlightDelayModel(artifacts_dir=artifacts_dir)
    return _model_instance
