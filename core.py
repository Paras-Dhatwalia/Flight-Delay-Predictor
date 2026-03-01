"""
core.py — Model loading, inference pipeline, and utility functions.

Depends only on config.py (pure data). Houses all ML-related logic
and the @st.cache_resource model loader.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import RISK_LOW, RISK_HIGH, _ROUTE_DISTANCES, _AIRPORT_COORDS

# ─── Paths & System ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Pure Helpers ────────────────────────────────────────────

def sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles."""
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a)) * 0.621371


def _lookup_distance(origin: str, dest: str) -> Optional[float]:
    """Look up or compute route distance in miles."""
    key = frozenset({origin, dest})
    if key in _ROUTE_DISTANCES:
        return float(_ROUTE_DISTANCES[key])
    if origin in _AIRPORT_COORDS and dest in _AIRPORT_COORDS:
        c1, c2 = _AIRPORT_COORDS[origin], _AIRPORT_COORDS[dest]
        return round(_haversine_miles(c1[0], c1[1], c2[0], c2[1]), 1)
    return None


def _estimate_elapsed(distance_mi: Optional[float]) -> Optional[float]:
    """Rough elapsed-time estimate from distance."""
    if distance_mi is None:
        return None
    return round((distance_mi / 500.0) * 60.0 + 30.0, 1)


# ─── DataFrame Construction ──────────────────────────────────

def build_raw_df(
    airline: str, origin: str, dest: str,
    dep_dt: datetime, tail: str = "UNKNOWN",
) -> pd.DataFrame:
    """Construct the raw single-row DataFrame for the feature pipeline."""
    d = _lookup_distance(origin, dest)
    return pd.DataFrame([{
        "OP_CARRIER": airline.upper().strip(),
        "ORIGIN":     origin.upper().strip(),
        "DEST":       dest.upper().strip(),
        "TAIL_NUM":   tail.upper().strip() or "UNKNOWN",
        "FL_DATE":    pd.Timestamp(dep_dt.date()),
        "Month":      dep_dt.month,
        "DayofMonth": dep_dt.day,
        "DayOfWeek":  dep_dt.isoweekday(),
        "CRS_DEP_TIME": dep_dt.hour * 100 + dep_dt.minute,
        "DISTANCE":       d,
        "CRS_ELAPSED_TIME": _estimate_elapsed(d),
    }])


# ─── Model Loading (cached) ─────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load LightGBM + pipeline + SHAP + params (cached across reruns).

    Returns
    -------
    tuple : (model, pipeline, explainer, feat_names, threshold, params)
    """
    import lightgbm as lgb

    model_files = sorted(ARTIFACTS_DIR.glob("model_*.txt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_files:
        st.error("No model found in backend/artifacts/"); st.stop()
    model = lgb.Booster(model_file=str(model_files[0]))
    feat_names = model.feature_name()

    pipe_files = sorted(ARTIFACTS_DIR.glob("pipeline*.pkl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not pipe_files:
        st.error("No pipeline found in backend/artifacts/"); st.stop()
    pipeline = joblib.load(pipe_files[0])

    params: dict = {}
    threshold = 0.5
    pp = ARTIFACTS_DIR / "params.json"
    if pp.exists():
        with pp.open() as f:
            params = json.load(f)
        threshold = float(params.get("threshold", threshold))

    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        logger.warning("SHAP unavailable: %s", e)

    return model, pipeline, explainer, feat_names, threshold, params


# ─── Inference ───────────────────────────────────────────────

def run_prediction(
    model, pipeline, explainer,
    feat_names: List[str], threshold: float,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Full inference pipeline.

    Returns dict with keys: probability, prediction, risk_level,
    all_shap, base_value, route_rate, airline_rate, threshold
    """
    transformed = pipeline.transform(raw_df)

    # Extract contextual rates before feature selection
    route_rate = (float(transformed["route_delay_rate"].iloc[0])
                  if "route_delay_rate" in transformed.columns else None)
    airline_rate = (float(transformed["airline_delay_rate"].iloc[0])
                    if "airline_delay_rate" in transformed.columns else None)

    # Align features
    if feat_names and all(f in transformed.columns for f in feat_names):
        feat_df = transformed[feat_names]
    elif feat_names and len(transformed.columns) == len(feat_names):
        feat_df = transformed.copy(); feat_df.columns = feat_names
    else:
        feat_df = transformed

    prob = float(np.clip(model.predict(feat_df)[0], 0.0, 1.0))
    pred = int(prob >= threshold)
    risk = "Low" if prob < RISK_LOW else ("High" if prob > RISK_HIGH else "Medium")

    # SHAP — values in log-odds space (additivity holds)
    all_shap: Dict[str, float] = {}
    base_value: Optional[float] = None
    if explainer is not None:
        try:
            bv = explainer.expected_value
            base_value = float(bv[1]) if isinstance(bv, (list, np.ndarray)) else float(bv)
            sv = explainer.shap_values(feat_df)
            sv = np.array(sv[1][0]) if isinstance(sv, list) else np.array(sv[0])
            all_shap = {n: float(v) for n, v in zip(feat_df.columns, sv)}
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

    return {
        "probability": prob, "prediction": pred, "risk_level": risk,
        "all_shap": all_shap, "base_value": base_value,
        "route_rate": route_rate, "airline_rate": airline_rate,
        "threshold": threshold,
    }
