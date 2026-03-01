"""
Flight Delay Predictor — Production Streamlit Application
==========================================================
Component-driven, state-managed Streamlit app with aviation-themed
design system. Predicts ≥15 min arrival delay using LightGBM with
SHAP waterfall explanations grounded in historical baselines.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Paths & System ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════

RISK_LOW = 0.3
RISK_HIGH = 0.6

AIRLINES: Dict[str, str] = {
    "AA": "American Airlines",  "DL": "Delta Air Lines",
    "UA": "United Airlines",    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",      "HA": "Hawaiian Airlines",
    "SY": "Sun Country Airlines",
}

AIRPORTS: Dict[str, str] = {
    "ATL": "Atlanta (ATL)",           "AUS": "Austin (AUS)",
    "BNA": "Nashville (BNA)",         "BOS": "Boston (BOS)",
    "CLT": "Charlotte (CLT)",         "DEN": "Denver (DEN)",
    "DFW": "Dallas/Fort Worth (DFW)", "DTW": "Detroit (DTW)",
    "EWR": "Newark (EWR)",           "HNL": "Honolulu (HNL)",
    "IAH": "Houston (IAH)",          "JFK": "New York JFK (JFK)",
    "LAS": "Las Vegas (LAS)",        "LAX": "Los Angeles (LAX)",
    "LGA": "New York LaGuardia (LGA)","MCO": "Orlando (MCO)",
    "MDW": "Chicago Midway (MDW)",    "MIA": "Miami (MIA)",
    "MSP": "Minneapolis (MSP)",       "MSY": "New Orleans (MSY)",
    "ORD": "Chicago O'Hare (ORD)",    "PDX": "Portland (PDX)",
    "PHL": "Philadelphia (PHL)",      "PHX": "Phoenix (PHX)",
    "SAN": "San Diego (SAN)",         "SEA": "Seattle (SEA)",
    "SFO": "San Francisco (SFO)",     "SLC": "Salt Lake City (SLC)",
    "STL": "St. Louis (STL)",         "TPA": "Tampa (TPA)",
}

FEATURE_DISPLAY: Dict[str, str] = {
    "CRS_DEP_TIME_sin":  "Departure time pattern",
    "CRS_DEP_TIME_cos":  "Departure time pattern",
    "DayOfWeek_sin":     "Day of week effect",
    "DayOfWeek_cos":     "Day of week effect",
    "Month_sin":         "Seasonal trend",
    "Month_cos":         "Seasonal trend",
    "departure_hour":    "Hour of departure",
    "route_distance_km": "Route distance",
    "DISTANCE":          "Flight distance",
    "CRS_ELAPSED_TIME":  "Scheduled duration",
    "OP_CARRIER_te":     "Airline reliability",
    "ORIGIN_te":         "Origin airport tendency",
    "DEST_te":           "Destination airport tendency",
    "TAIL_NUM_te":       "Aircraft history",
    "route_delay_rate":  "Route delay history",
    "airline_delay_rate":"Airline delay history",
    "is_near_holiday":   "Holiday proximity",
}

AIRLINE_BRANDS: Dict[str, Dict[str, str]] = {
    "AA": {"color": "#0078D2", "accent": "rgba(0,120,210,0.12)",
            "logo": "https://logo.clearbit.com/aa.com"},
    "DL": {"color": "#C01933", "accent": "rgba(192,25,51,0.12)",
            "logo": "https://logo.clearbit.com/delta.com"},
    "UA": {"color": "#005DAA", "accent": "rgba(0,93,170,0.12)",
            "logo": "https://logo.clearbit.com/united.com"},
    "WN": {"color": "#F9B612", "accent": "rgba(249,182,18,0.12)",
            "logo": "https://logo.clearbit.com/southwest.com"},
    "B6": {"color": "#003876", "accent": "rgba(0,56,118,0.12)",
            "logo": "https://logo.clearbit.com/jetblue.com"},
    "AS": {"color": "#01426A", "accent": "rgba(1,66,106,0.12)",
            "logo": "https://logo.clearbit.com/alaskaair.com"},
    "NK": {"color": "#FFE500", "accent": "rgba(255,229,0,0.12)",
            "logo": "https://logo.clearbit.com/spirit.com"},
    "F9": {"color": "#00B140", "accent": "rgba(0,177,64,0.12)",
            "logo": "https://logo.clearbit.com/flyfrontier.com"},
    "G4": {"color": "#003B70", "accent": "rgba(0,59,112,0.12)",
            "logo": "https://logo.clearbit.com/allegiantair.com"},
    "HA": {"color": "#6B2FA0", "accent": "rgba(107,47,160,0.12)",
            "logo": "https://logo.clearbit.com/hawaiianairlines.com"},
    "SY": {"color": "#F7931E", "accent": "rgba(247,147,30,0.12)",
            "logo": "https://logo.clearbit.com/suncountry.com"},
}
_DEFAULT_BRAND = {"color": "#38bdf8", "accent": "rgba(56,189,248,0.12)", "logo": ""}

_ROUTE_DISTANCES: Dict[frozenset, float] = {
    frozenset({"JFK","LAX"}):2475, frozenset({"JFK","SFO"}):2586,
    frozenset({"JFK","ORD"}):740,  frozenset({"JFK","MIA"}):1089,
    frozenset({"JFK","BOS"}):187,  frozenset({"JFK","ATL"}):760,
    frozenset({"JFK","DFW"}):1391, frozenset({"JFK","SEA"}):2422,
    frozenset({"JFK","DEN"}):1626, frozenset({"JFK","LAS"}):2243,
    frozenset({"LAX","SFO"}):337,  frozenset({"LAX","ORD"}):1745,
    frozenset({"LAX","MIA"}):2342, frozenset({"LAX","ATL"}):1946,
    frozenset({"LAX","DFW"}):1235, frozenset({"LAX","SEA"}):954,
    frozenset({"LAX","DEN"}):862,  frozenset({"LAX","LAS"}):236,
    frozenset({"LAX","PHX"}):370,  frozenset({"ORD","ATL"}):606,
    frozenset({"ORD","DFW"}):802,  frozenset({"ORD","MIA"}):1197,
    frozenset({"ORD","SEA"}):1720, frozenset({"ORD","DEN"}):920,
    frozenset({"ORD","BOS"}):867,  frozenset({"ATL","DFW"}):732,
    frozenset({"ATL","MIA"}):661,  frozenset({"ATL","BOS"}):946,
    frozenset({"ATL","SEA"}):2182, frozenset({"DFW","MIA"}):1121,
    frozenset({"DFW","SEA"}):1660, frozenset({"DFW","DEN"}):641,
    frozenset({"DFW","LAS"}):1055, frozenset({"DFW","PHX"}):868,
    frozenset({"SFO","SEA"}):679,  frozenset({"SFO","DEN"}):967,
    frozenset({"SFO","LAS"}):414,  frozenset({"SFO","PHX"}):651,
    frozenset({"SEA","DEN"}):1024, frozenset({"SEA","LAS"}):867,
    frozenset({"DEN","LAS"}):598,  frozenset({"DEN","PHX"}):586,
    frozenset({"LAS","PHX"}):256,  frozenset({"MIA","BOS"}):1258,
    frozenset({"BOS","ORD"}):867,
}

_AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    "ATL":(33.64,-84.43),  "BOS":(42.37,-71.01),  "CLT":(35.21,-80.94),
    "DEN":(39.86,-104.67), "DFW":(32.90,-97.04),  "DTW":(42.21,-83.35),
    "EWR":(40.69,-74.17),  "IAH":(29.99,-95.34),  "JFK":(40.64,-73.78),
    "LAX":(33.94,-118.41), "LAS":(36.08,-115.15), "LGA":(40.78,-73.87),
    "MCO":(28.43,-81.31),  "MDW":(41.79,-87.75),  "MIA":(25.80,-80.29),
    "MSP":(44.88,-93.22),  "ORD":(41.97,-87.91),  "PDX":(45.59,-122.60),
    "PHL":(39.87,-75.24),  "PHX":(33.44,-112.01), "SAN":(32.73,-117.19),
    "SEA":(47.45,-122.31), "SFO":(37.62,-122.38), "SLC":(40.79,-111.98),
    "STL":(38.75,-90.37),  "TPA":(27.98,-82.53),  "AUS":(30.20,-97.67),
    "BNA":(36.12,-86.68),  "HNL":(21.32,-157.92), "MSY":(29.99,-90.26),
}


# ═══════════════════════════════════════════════════════════════
#  CSS DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════

DESIGN_CSS = """<style>
/* ─── Status Indicator ─── */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 14px; background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3); border-radius: 20px;
    font-size: 0.75rem; color: #4ade80;
}
.status-dot {
    width: 8px; height: 8px; background: #22c55e;
    border-radius: 50%; animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%,100%{opacity:1} 50%{opacity:0.4}
}

/* ─── Header ─── */
.app-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.app-title {
    font-size: 2rem; font-weight: 700; color: #f1f5f9; margin: 0;
}
.app-subtitle {
    font-size: 0.85rem; color: #64748b; margin-top: 4px;
}

/* ─── Hero Empty State ─── */
.hero-container {
    text-align: center; padding: 3.5rem 2rem; margin: 1.5rem 0;
    background: linear-gradient(135deg, rgba(56,189,248,0.04) 0%, rgba(139,92,246,0.04) 100%);
    border: 1px solid #1e293b; border-radius: 16px;
}
.hero-icon {
    font-size: 3.5rem; display: block; margin-bottom: 0.8rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)}
}
.hero-title { font-size: 1.4rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem; }
.hero-text { color: #64748b; font-size: 0.9rem; max-width: 480px; margin: 0 auto; line-height: 1.6; }

/* ─── Boarding Pass ─── */
.boarding-pass {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 16px;
    overflow: hidden; margin: 1rem 0; position: relative;
}
.boarding-pass::before, .boarding-pass::after {
    content: ''; position: absolute; width: 20px; height: 20px;
    background: var(--background-color, #0f172a); border-radius: 50%;
    top: 50%; transform: translateY(-50%);
}
.boarding-pass::before { left: -10px; }
.boarding-pass::after  { right: -10px; }
.bp-header {
    background: rgba(56,189,248,0.06); border-bottom: 1px dashed #334155;
    padding: 10px 24px; display: flex; align-items: center; justify-content: space-between;
}
.bp-header-left { display: flex; align-items: center; gap: 8px; }
.bp-logo { font-size: 1.1rem; }
.bp-tag {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 2px;
    color: #38bdf8; text-transform: uppercase;
}
.bp-airline-tag {
    font-size: 0.7rem; color: #94a3b8;
    background: rgba(148,163,184,0.08); padding: 3px 10px; border-radius: 10px;
}
.bp-body { padding: 18px 24px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.bp-route { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 200px; }
.bp-airport { font-size: 2rem; font-weight: 800; color: #f1f5f9; letter-spacing: 2px; }
.bp-city { font-size: 0.65rem; color: #64748b; margin-top: 2px; }
.bp-route-line { flex: 1; text-align: center; position: relative; }
.bp-route-line::before {
    content:''; position: absolute; top: 50%; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, #334155, #38bdf8, #334155);
}
.bp-route-plane {
    position: relative; z-index: 1; font-size: 1rem;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 0 8px; color: #38bdf8;
}
.bp-info { display: flex; gap: 18px; flex-wrap: wrap; }
.bp-field { min-width: 65px; }
.bp-label {
    font-size: 0.55rem; font-weight: 600; letter-spacing: 1.5px;
    color: #64748b; text-transform: uppercase; margin-bottom: 3px;
}
.bp-value { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; }

/* ─── Risk Banner ─── */
.risk-banner { padding: 18px 22px; border-radius: 12px; margin: 0.5rem 0; border: 1px solid; }
.risk-banner-low    { background: rgba(34,197,94,0.06);  border-color: rgba(34,197,94,0.2); }
.risk-banner-medium { background: rgba(234,179,8,0.06);  border-color: rgba(234,179,8,0.2); }
.risk-banner-high   { background: rgba(239,68,68,0.06);  border-color: rgba(239,68,68,0.2); }
.risk-verdict { font-size: 1.2rem; font-weight: 700; margin-bottom: 6px; }
.risk-verdict-low    { color: #4ade80; }
.risk-verdict-medium { color: #facc15; }
.risk-verdict-high   { color: #f87171; }
.risk-detail { font-size: 0.82rem; color: #94a3b8; line-height: 1.6; }
.risk-tip {
    font-size: 0.78rem; color: #64748b; margin-top: 10px;
    padding-top: 10px; border-top: 1px solid rgba(71,85,105,0.3);
}

/* ─── Metric Cards ─── */
div[data-testid="stMetric"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 14px;
}
div[data-testid="stMetric"] label {
    color: #64748b !important; font-size: 0.65rem !important;
    text-transform: uppercase; letter-spacing: 1px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}

/* ─── Sidebar Metric ─── */
.sb-metric {
    background: rgba(30,41,59,0.5); border: 1px solid #334155;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
}
.sb-metric-label {
    font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px;
}
.sb-metric-value { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-top: 2px; }

/* ─── SHAP Legend ─── */
.shap-legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.73rem; color: #94a3b8; }
.legend-dot  { width: 10px; height: 10px; border-radius: 3px; }

/* ─── Route Preview ─── */
.route-preview {
    text-align: center; margin: 0.8rem 0 0.3rem; padding: 12px;
    background: rgba(56,189,248,0.04); border-radius: 10px; border: 1px solid #1e293b;
}
.route-code {
    font-size: 1.8rem; font-weight: 800; color: #f1f5f9; letter-spacing: 3px;
}
.route-arrow { color: #38bdf8; margin: 0 12px; font-size: 1.1rem; }

/* ─── Footer ─── */
.app-footer { text-align: center; padding: 1.5rem 0; font-size: 0.7rem; color: #475569; }
</style>"""


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a)) * 0.621371


def _lookup_distance(origin: str, dest: str) -> Optional[float]:
    key = frozenset({origin, dest})
    if key in _ROUTE_DISTANCES:
        return float(_ROUTE_DISTANCES[key])
    if origin in _AIRPORT_COORDS and dest in _AIRPORT_COORDS:
        c1, c2 = _AIRPORT_COORDS[origin], _AIRPORT_COORDS[dest]
        return round(_haversine_miles(c1[0], c1[1], c2[0], c2[1]), 1)
    return None


def _estimate_elapsed(distance_mi: Optional[float]) -> Optional[float]:
    if distance_mi is None:
        return None
    return round((distance_mi / 500.0) * 60.0 + 30.0, 1)


def build_raw_df(airline: str, origin: str, dest: str,
                 dep_dt: datetime, tail: str = "UNKNOWN") -> pd.DataFrame:
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


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING & INFERENCE
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """Load LightGBM + pipeline + SHAP + params (cached across reruns)."""
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


def run_prediction(model, pipeline, explainer,
                   feat_names: list, threshold: float,
                   raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Full inference pipeline → dict of results."""
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

    # SHAP
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


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def create_gauge(prob: float) -> go.Figure:
    pct = prob * 100
    c = "#22c55e" if prob < 0.3 else ("#eab308" if prob <= 0.6 else "#ef4444")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": c, "family": "sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "dtick": 25,
                     "tickfont": {"color": "#64748b", "size": 11},
                     "tickcolor": "#475569", "tickwidth": 1},
            "bar": {"color": c, "thickness": 0.8},
            "bgcolor": "#1e293b", "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "rgba(34,197,94,0.08)"},
                {"range": [30, 60], "color": "rgba(234,179,8,0.08)"},
                {"range": [60,100], "color": "rgba(239,68,68,0.08)"},
            ],
        },
        title={"text": "Delay Probability", "font": {"size": 14, "color": "#64748b"}},
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=55, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1f5f9"},
    )
    return fig


def create_waterfall(all_shap: Dict[str, float], base_value: float) -> go.Figure:
    """SHAP additive-attribution waterfall chart."""
    sorted_feats = sorted(all_shap.items(), key=lambda x: abs(x[1]))
    n_show = min(7, len(sorted_feats))
    top = sorted_feats[-n_show:]          # ascending |impact|
    others = sorted_feats[:-n_show] if len(sorted_feats) > n_show else []
    other_sum = sum(v for _, v in others)

    names, vals, measures, labels = [], [], [], []

    # Base
    bp = sigmoid(base_value)
    names.append(f"Baseline ({bp*100:.0f}%)")
    vals.append(base_value); measures.append("absolute")
    labels.append(f"{bp*100:.0f}%")

    # Other bucket
    if abs(other_sum) > 0.005:
        names.append("Other factors")
        vals.append(other_sum); measures.append("relative")
        labels.append(f"{other_sum:+.3f}")

    # Features (smallest → largest so largest ends up near the top)
    for fname, sv in top:
        names.append(FEATURE_DISPLAY.get(fname, fname))
        vals.append(sv); measures.append("relative")
        labels.append(f"{sv:+.3f}")

    # Final
    fp = sigmoid(base_value + sum(all_shap.values()))
    names.append(f"Prediction ({fp*100:.1f}%)")
    vals.append(0); measures.append("total")
    labels.append(f"{fp*100:.1f}%")

    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=vals, measure=measures,
        text=labels, textposition="outside",
        textfont={"color": "#94a3b8", "size": 11},
        increasing={"marker": {"color": "rgba(239,68,68,0.65)",
                                "line": {"color": "#ef4444", "width": 1}}},
        decreasing={"marker": {"color": "rgba(59,130,246,0.65)",
                                "line": {"color": "#3b82f6", "width": 1}}},
        totals={"marker": {"color": "rgba(139,92,246,0.65)",
                            "line": {"color": "#8b5cf6", "width": 1}}},
        connector={"line": {"color": "#334155", "width": 1, "dash": "dot"}},
    ))
    fig.update_layout(
        height=max(260, (n_show + 3) * 40),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": True, "gridcolor": "rgba(51,65,85,0.3)",
               "zeroline": True, "zerolinecolor": "#475569",
               "tickfont": {"color": "#64748b", "size": 10},
               "title": {"text": "Model output (log-odds)",
                         "font": {"color": "#475569", "size": 10}}},
        yaxis={"tickfont": {"color": "#cbd5e1", "size": 12},
               "autorange": "reversed"},
        font={"color": "#f1f5f9"}, showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  3D GLOBE
# ═══════════════════════════════════════════════════════════════

def render_route_globe(origin: str, dest: str, airline: str) -> None:
    """Render an interactive 3D globe with the flight route arc."""
    import streamlit.components.v1 as components

    o_coords = _AIRPORT_COORDS.get(origin)
    d_coords = _AIRPORT_COORDS.get(dest)
    if not o_coords or not d_coords:
        st.caption("Globe unavailable — airport coordinates not found.")
        return

    o_lat, o_lng = o_coords
    d_lat, d_lng = d_coords
    mid_lat = (o_lat + d_lat) / 2
    mid_lng = (o_lng + d_lng) / 2

    brand = AIRLINE_BRANDS.get(airline, _DEFAULT_BRAND)
    arc_color = brand["color"]

    html = f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background: transparent; overflow: hidden; }}
      #globe-container {{
        width: 100%; height: 480px; border-radius: 14px;
        overflow: hidden; position: relative;
        background: radial-gradient(ellipse at center, #0a1628 0%, #000 100%);
      }}
    </style></head>
    <body>
    <div id="globe-container"></div>

    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/globe.gl@2.33.0/dist/globe.gl.min.js"></script>
    <script>
    (function() {{
      const container = document.getElementById('globe-container');
      const w = container.clientWidth;
      const h = container.clientHeight;

      const globe = Globe()
        (container)
        .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-night.jpg')
        .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
        .backgroundImageUrl('')
        .backgroundColor('rgba(0,0,0,0)')
        .atmosphereColor('#38bdf8')
        .atmosphereAltitude(0.18)
        .width(w)
        .height(h);

      // Arc config
      const DASH_LEN  = 0.6;
      const DASH_GAP  = 0.3;
      const CYCLE_MS  = 2500;
      const DASH_UNIT = DASH_LEN + DASH_GAP;

      const arcData = [{{
        startLat: {o_lat}, startLng: {o_lng},
        endLat: {d_lat},   endLng: {d_lng},
        color: ['{arc_color}', '#38bdf8']
      }}];
      globe
        .arcsData(arcData)
        .arcColor('color')
        .arcAltitudeAutoScale(0.45)
        .arcStroke(1.5)
        .arcDashLength(DASH_LEN)
        .arcDashGap(DASH_GAP)
        .arcDashAnimateTime(CYCLE_MS);

      // Airport markers
      const points = [
        {{ lat: {o_lat}, lng: {o_lng}, label: '{origin}', size: 0.6, color: '#22c55e' }},
        {{ lat: {d_lat}, lng: {d_lng}, label: '{dest}',   size: 0.6, color: '#ef4444' }}
      ];
      globe
        .pointsData(points)
        .pointColor('color')
        .pointAltitude(0.01)
        .pointRadius('size')
        .pointsMerge(false);

      // Airport IATA labels
      globe
        .labelsData(points)
        .labelLat('lat')
        .labelLng('lng')
        .labelText('label')
        .labelSize(1.8)
        .labelDotRadius(0.4)
        .labelColor(() => '#f1f5f9')
        .labelAltitude(0.02)
        .labelResolution(2);

      // Great-circle interpolation for plane animation
      const toRad = d => d * Math.PI / 180;
      const toDeg = r => r * 180 / Math.PI;
      const lat1 = toRad({o_lat}), lng1 = toRad({o_lng});
      const lat2 = toRad({d_lat}), lng2 = toRad({d_lng});

      function gcInterp(t) {{
        const d = 2 * Math.asin(Math.sqrt(
          Math.pow(Math.sin((lat2-lat1)/2),2) +
          Math.cos(lat1)*Math.cos(lat2)*Math.pow(Math.sin((lng2-lng1)/2),2)
        ));
        if (d < 1e-10) return {{ lat: {o_lat}, lng: {o_lng} }};
        const A = Math.sin((1-t)*d) / Math.sin(d);
        const B = Math.sin(t*d) / Math.sin(d);
        const x = A*Math.cos(lat1)*Math.cos(lng1) + B*Math.cos(lat2)*Math.cos(lng2);
        const y = A*Math.cos(lat1)*Math.sin(lng1) + B*Math.cos(lat2)*Math.sin(lng2);
        const z = A*Math.sin(lat1) + B*Math.sin(lat2);
        return {{ lat: toDeg(Math.atan2(z, Math.sqrt(x*x+y*y))), lng: toDeg(Math.atan2(y, x)) }};
      }}

      // Bearing for rotation
      function bearing(t) {{
        const p1 = gcInterp(Math.max(0, t - 0.01));
        const p2 = gcInterp(Math.min(1, t + 0.01));
        const dLng = toRad(p2.lng - p1.lng);
        const la1 = toRad(p1.lat), la2 = toRad(p2.lat);
        const bx = Math.sin(dLng) * Math.cos(la2);
        const by = Math.cos(la1)*Math.sin(la2) - Math.sin(la1)*Math.cos(la2)*Math.cos(dLng);
        return toDeg(Math.atan2(bx, by));
      }}

      // Animated plane
      let planeEl = null;
      const planeData = [{{ lat: {o_lat}, lng: {o_lng} }}];
      globe
        .htmlElementsData(planeData)
        .htmlLat('lat')
        .htmlLng('lng')
        .htmlAltitude(0.065)
        .htmlElement(d => {{
          const el = document.createElement('div');
          el.style.cssText = 'font-size:20px;pointer-events:none;'
            + 'filter:drop-shadow(0 0 8px {arc_color});transition:transform 0.1s linear;';
          el.textContent = '✈';
          planeEl = el;
          return el;
        }});

      // Animation loop — plane rides in the dash gap
      const startTime = performance.now();
      function animatePlane() {{
        const elapsed = performance.now() - startTime;
        const dashProgress = (elapsed % CYCLE_MS) / CYCLE_MS;
        // Place plane in the middle of the first gap
        const t = (dashProgress * DASH_UNIT + DASH_LEN + DASH_GAP / 2) % 1;
        const pos = gcInterp(t);
        planeData[0].lat = pos.lat;
        planeData[0].lng = pos.lng;
        globe.htmlElementsData(planeData);

        if (planeEl) {{
          const angle = bearing(t);
          planeEl.style.transform = 'rotate(' + (angle + 120) + 'deg)';
        }}
        requestAnimationFrame(animatePlane);
      }}
      requestAnimationFrame(animatePlane);

      // Camera — frame both airports
      globe.pointOfView({{ lat: {mid_lat}, lng: {mid_lng}, altitude: 2.0 }}, 1000);

      // Subtle auto-rotate
      globe.controls().autoRotate = true;
      globe.controls().autoRotateSpeed = 0.3;
      globe.controls().enableZoom = true;

      // Responsive
      window.addEventListener('resize', () => {{
        globe.width(container.clientWidth).height(container.clientHeight);
      }});
    }})();
    </script>
    </body></html>
    """
    components.html(html, height=490)


# ═══════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════

def render_header(loaded: bool) -> None:
    badge = ('<span class="status-badge"><span class="status-dot"></span>'
             'Model Loaded &amp; Calibrated</span>') if loaded else ''
    st.markdown(f"""
    <div class="app-header">
        <div style="display:flex;justify-content:center;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="font-size:1.8rem;">✈️</span>
            <span class="app-title">Flight Delay Predictor</span>
        </div>
        <div class="app-subtitle">ML-powered delay risk analysis with explainable predictions</div>
        <div style="margin-top:10px;">{badge}</div>
    </div>""", unsafe_allow_html=True)


def render_sidebar(params: dict) -> None:
    with st.sidebar:
        st.markdown("## About")
        st.markdown(
            "Predicts whether a US domestic flight will arrive "
            "**≥15 minutes late** using only pre-departure data "
            "available **24 hours before departure**."
        )
        st.markdown("---")
        st.markdown("### Model Evaluation")
        metrics = params.get("metrics", {})
        for label, key, fmt in [
            ("ROC-AUC", "roc_auc", ".3f"), ("PR-AUC", "pr_auc", ".3f"),
            ("Brier Score", "brier_score", ".3f"), ("F1 Score", "f1", ".3f"),
        ]:
            v = metrics.get(key)
            if v is not None:
                st.markdown(f'<div class="sb-metric">'
                            f'<div class="sb-metric-label">{label}</div>'
                            f'<div class="sb-metric-value">{v:{fmt}}</div>'
                            f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Data Provenance")
        st.markdown(
            "- **Source:** BTS On-Time Reporting\n"
            "- **Records:** ~7 M US domestic flights\n"
            "- **Period:** Jan–Dec 2024\n"
            "- **Features:** 17 pre-departure signals\n"
            "- **Algorithm:** LightGBM (237 rounds)"
        )
        st.markdown("---")
        st.warning(
            "**Disclaimer:** This tool provides probabilistic estimates "
            "based on historical data. It does not account for real-time "
            "weather, ATC delays, or mechanical issues. Not financial or "
            "travel advice."
        )


def render_hero() -> None:
    st.markdown("""
    <div class="hero-container">
        <span class="hero-icon">✈️</span>
        <div class="hero-title">Enter flight details to analyze delay risk</div>
        <div class="hero-text">
            Get an AI-powered probability estimate with SHAP explanations
            showing exactly what drives the prediction for your specific flight.
        </div>
    </div>""", unsafe_allow_html=True)


def render_flight_form() -> Tuple[bool, Dict[str, Any]]:
    """Input form with example-flight injection. Returns (submitted, data)."""
    akeys = list(AIRLINES.keys())
    pkeys = list(AIRPORTS.keys())

    # Example injection buttons
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎯 Try: High-Risk Flight", use_container_width=True):
            st.session_state.update(_def_al="NK", _def_or="EWR", _def_de="ORD",
                _def_dt=date.today()+timedelta(days=22), _def_tm=time(19,45), _def_tl="")
            st.rerun()
    with c2:
        if st.button("✅ Try: Low-Risk Flight", use_container_width=True):
            st.session_state.update(_def_al="DL", _def_or="ATL", _def_de="JFK",
                _def_dt=date.today()+timedelta(days=3), _def_tm=time(10,15), _def_tl="")
            st.rerun()

    # Resolve defaults
    da = st.session_state.get("_def_al", "AA")
    do = st.session_state.get("_def_or", "JFK")
    dd = st.session_state.get("_def_de", "LAX")
    ddt = st.session_state.get("_def_dt", date.today() + timedelta(days=1))
    dtm = st.session_state.get("_def_tm", time(14, 30))
    dtl = st.session_state.get("_def_tl", "")

    # Route preview
    st.markdown(f"""
    <div class="route-preview">
        <span class="route-code">{do}</span>
        <span class="route-arrow">── ✈ ──</span>
        <span class="route-code">{dd}</span>
    </div>""", unsafe_allow_html=True)

    with st.form("flight_form"):
        col1, col2 = st.columns(2)
        with col1:
            airline = st.selectbox("✈️ Airline", akeys,
                format_func=lambda x: f"{x} — {AIRLINES[x]}",
                index=akeys.index(da) if da in akeys else 0)
            origin = st.selectbox("📍 Origin Airport", pkeys,
                format_func=lambda x: AIRPORTS[x],
                index=pkeys.index(do) if do in pkeys else 0)
            dep_date = st.date_input("📅 Departure Date", value=ddt,
                                     min_value=date.today())
        with col2:
            dest = st.selectbox("📍 Destination Airport", pkeys,
                format_func=lambda x: AIRPORTS[x],
                index=pkeys.index(dd) if dd in pkeys else 0)
            dep_time = st.time_input("🕒 Departure Time", value=dtm)
            tail = st.text_input("🔖 Tail Number (optional)", value=dtl,
                                 placeholder="e.g. N123AA", max_chars=10)

        submitted = st.form_submit_button("🔮 Analyze Delay Risk",
                                          use_container_width=True, type="primary")

    return submitted, dict(airline=airline, origin=origin, destination=dest,
                           dep_date=dep_date, dep_time=dep_time, tail=tail)


def render_boarding_pass(fd: dict, dist: Optional[float], elapsed: Optional[float]) -> None:
    al = fd["airline"]; ori = fd["origin"]; dst = fd["destination"]
    dt = datetime.combine(fd["dep_date"], fd["dep_time"])
    al_name = AIRLINES.get(al, al)
    ori_city = AIRPORTS.get(ori, ori).split("(")[0].strip()
    dst_city = AIRPORTS.get(dst, dst).split("(")[0].strip()
    d_s = f"{dist:,.0f} mi" if dist else "—"
    e_s = f"~{elapsed:.0f} min" if elapsed else "—"

    brand = AIRLINE_BRANDS.get(al, _DEFAULT_BRAND)
    bc = brand["color"]
    logo_url = brand["logo"]
    accent_bg = brand["accent"]

    # Logo HTML — small image with fallback to text
    logo_html = (
        f'<img src="{logo_url}" alt="{al}" '
        f'style="width:28px;height:28px;border-radius:6px;object-fit:contain;'
        f'background:#fff;padding:2px;" onerror="this.style.display=\'none\'">'
        if logo_url else ""
    )

    st.markdown(f"""
    <div class="boarding-pass" style="border-left:4px solid {bc};">
        <div class="bp-header" style="background:{accent_bg};">
            <div class="bp-header-left">
                {logo_html}
                <span class="bp-tag">Flight Delay Analysis</span>
            </div>
            <span class="bp-airline-tag" style="border:1px solid {bc}40;color:{bc};">{al} · {al_name}</span>
        </div>
        <div class="bp-body">
            <div class="bp-route">
                <div><div class="bp-airport">{ori}</div><div class="bp-city">{ori_city}</div></div>
                <div class="bp-route-line"><span class="bp-route-plane" style="color:{bc};">✈</span></div>
                <div><div class="bp-airport">{dst}</div><div class="bp-city">{dst_city}</div></div>
            </div>
            <div class="bp-info">
                <div class="bp-field"><div class="bp-label">Date</div><div class="bp-value">{dt.strftime('%b %d, %Y')}</div></div>
                <div class="bp-field"><div class="bp-label">Time</div><div class="bp-value">{dt.strftime('%H:%M')}</div></div>
                <div class="bp-field"><div class="bp-label">Distance</div><div class="bp-value">{d_s}</div></div>
                <div class="bp-field"><div class="bp-label">Duration</div><div class="bp-value">{e_s}</div></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_risk_banner(risk: str, prob: float, route_rate: Optional[float],
                       airline_rate: Optional[float], threshold: float) -> None:
    lv = risk.lower()
    verdicts = {
        "low":    ("✅ Low delay risk",
                   "Your flight parameters suggest a low probability of significant delay."),
        "medium": ("⏳ Moderate delay risk",
                   "There is a moderate chance of arrival delay based on historical patterns."),
        "high":   ("⚠️ High delay risk detected",
                   "Historical data suggests an elevated probability of arrival delay."),
    }
    tips = {
        "low":    "No action needed. This flight profile historically performs well.",
        "medium": "Consider monitoring your flight status closer to departure. "
                  "Allow buffer time for connections.",
        "high":   "Plan for potential delays. Avoid tight connections and consider "
                  "travel insurance.",
    }
    title, desc = verdicts.get(lv, verdicts["medium"])
    tip = tips.get(lv, "")

    ctx = []
    if route_rate is not None:
        ctx.append(f"Flights on this route arrive on time <strong>{(1-route_rate)*100:.0f}%</strong> of the time")
    if airline_rate is not None:
        ctx.append(f"this airline's on-time rate is <strong>{(1-airline_rate)*100:.0f}%</strong>")
    ctx_html = (". ".join(ctx) + ".") if ctx else ""

    st.markdown(f"""
    <div class="risk-banner risk-banner-{lv}">
        <div class="risk-verdict risk-verdict-{lv}">{title}</div>
        <div class="risk-detail">
            {desc} The model estimates a <strong>{prob*100:.1f}%</strong> probability
            of ≥15 min arrival delay (threshold: {threshold*100:.0f}%).
        </div>
        {"<div class='risk-detail' style='margin-top:6px;'>" + ctx_html + "</div>" if ctx_html else ""}
        <div class="risk-tip">💡 {tip}</div>
    </div>""", unsafe_allow_html=True)


def render_shap_section(all_shap: Dict[str, float],
                        base_value: Optional[float], prob: float) -> None:
    if not all_shap or base_value is None:
        st.info("SHAP explanations are unavailable for this prediction.")
        return

    st.markdown("#### 🔍 What's driving this prediction?")

    fig = create_waterfall(all_shap, base_value)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="shap-legend">
        <div class="legend-item"><div class="legend-dot" style="background:rgba(239,68,68,0.7);"></div>Increases delay risk</div>
        <div class="legend-item"><div class="legend-dot" style="background:rgba(59,130,246,0.7);"></div>Decreases delay risk</div>
        <div class="legend-item"><div class="legend-dot" style="background:rgba(139,92,246,0.7);"></div>Baseline / Prediction</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("ℹ️ How does the model calculate this?"):
        bp = sigmoid(base_value)
        st.markdown(f"""
The model uses **SHAP (SHapley Additive exPlanations)** to decompose
each prediction into individual feature contributions.

**Reading the waterfall chart:**
- The **baseline** ({bp*100:.0f}%) is the average delay rate across all training flights
- Each bar shows how a specific factor **pushes the prediction up or down**
  from that baseline
- **Red bars** = factors increasing delay risk for your flight
- **Blue bars** = factors decreasing delay risk
- The **final prediction** ({prob*100:.1f}%) is the cumulative result
  of all contributions

This ensures full transparency — you can see exactly *why* the model
made its prediction, not just the final number.
        """)


def render_footer() -> None:
    st.markdown("""
    <div class="app-footer">
        Binary classification · ≥15 min arrival delay · T-24h pre-departure features only ·
        LightGBM + SHAP · Trained on BTS 2024 data
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Flight Delay Predictor",
        page_icon="✈️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Inject design system
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)

    # Session state init
    if "prediction_run" not in st.session_state:
        st.session_state.prediction_run = False

    # Load model
    with st.spinner("Initializing model and SHAP explainer..."):
        model, pipeline, explainer, feat_names, threshold, params = load_model()

    # Shell
    render_header(loaded=True)
    render_sidebar(params)

    st.markdown("---")

    # Form
    submitted, fd = render_flight_form()

    # Handle submission
    if submitted:
        if fd["origin"] == fd["destination"]:
            st.error("Origin and destination cannot be the same airport.")
            st.stop()

        dep_dt = datetime.combine(fd["dep_date"], fd["dep_time"])

        with st.spinner("Analyzing historical route distributions and computing SHAP values..."):
            raw_df = build_raw_df(fd["airline"], fd["origin"], fd["destination"],
                                  dep_dt, fd["tail"] or "UNKNOWN")
            try:
                results = run_prediction(model, pipeline, explainer,
                                         feat_names, threshold, raw_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.session_state.prediction_run = True
        st.session_state.results = results
        st.session_state.form_data = fd
        st.session_state.distance = _lookup_distance(fd["origin"], fd["destination"])
        st.session_state.elapsed = _estimate_elapsed(st.session_state.distance)

    # Render results or empty state
    if st.session_state.prediction_run:
        res = st.session_state.results
        fd = st.session_state.form_data

        st.markdown("---")

        # Boarding pass
        render_boarding_pass(fd, st.session_state.distance, st.session_state.elapsed)

        # 3D Route Globe
        st.markdown("#### 🌍 Flight Route")
        render_route_globe(fd["origin"], fd["destination"], fd["airline"])

        # Gauge + Risk
        g_col, r_col = st.columns([1, 1.2])
        with g_col:
            st.plotly_chart(create_gauge(res["probability"]),
                            use_container_width=True, config={"displayModeBar": False})
        with r_col:
            render_risk_banner(res["risk_level"], res["probability"],
                               res["route_rate"], res["airline_rate"], res["threshold"])

        # Probability / Threshold metrics
        m1, m2 = st.columns(2)
        m1.metric("Delay Probability", f'{res["probability"]*100:.1f}%')
        m2.metric("Decision Threshold", f'{res["threshold"]*100:.0f}%')

        # SHAP
        st.markdown("---")
        render_shap_section(res["all_shap"], res["base_value"], res["probability"])

        # Reset
        st.markdown("---")
        if st.button("🔄 Analyze Another Flight", use_container_width=True):
            st.session_state.prediction_run = False
            for k in ("results", "form_data", "distance", "elapsed"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        render_hero()

    render_footer()


if __name__ == "__main__":
    main()
