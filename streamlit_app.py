"""
Flight Delay Predictor — Streamlit App
=======================================
Single-file Streamlit application that loads a trained LightGBM model
and predicts whether a flight will arrive ≥15 minutes late, using only
pre-departure (T-24h) features.

Deploy to Streamlit Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io  →  connect repo
    3. Set main file path:  streamlit_app.py
    4. Set requirements:    requirements_streamlit.txt
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"

RISK_LOW = 0.3
RISK_HIGH = 0.6
TOP_N_FACTORS = 5

# Major US airlines
AIRLINES: Dict[str, str] = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "SY": "Sun Country Airlines",
}

# Major US airports
AIRPORTS: Dict[str, str] = {
    "ATL": "Atlanta (ATL)",
    "AUS": "Austin (AUS)",
    "BNA": "Nashville (BNA)",
    "BOS": "Boston (BOS)",
    "CLT": "Charlotte (CLT)",
    "DEN": "Denver (DEN)",
    "DFW": "Dallas/Fort Worth (DFW)",
    "DTW": "Detroit (DTW)",
    "EWR": "Newark (EWR)",
    "HNL": "Honolulu (HNL)",
    "IAH": "Houston (IAH)",
    "JFK": "New York JFK (JFK)",
    "LAS": "Las Vegas (LAS)",
    "LAX": "Los Angeles (LAX)",
    "LGA": "New York LaGuardia (LGA)",
    "MCO": "Orlando (MCO)",
    "MDW": "Chicago Midway (MDW)",
    "MIA": "Miami (MIA)",
    "MSP": "Minneapolis (MSP)",
    "MSY": "New Orleans (MSY)",
    "ORD": "Chicago O'Hare (ORD)",
    "PDX": "Portland (PDX)",
    "PHL": "Philadelphia (PHL)",
    "PHX": "Phoenix (PHX)",
    "SAN": "San Diego (SAN)",
    "SEA": "Seattle (SEA)",
    "SFO": "San Francisco (SFO)",
    "SLC": "Salt Lake City (SLC)",
    "STL": "St. Louis (STL)",
    "TPA": "Tampa (TPA)",
}

# Route distances (statute miles)
_ROUTE_DISTANCES: Dict[frozenset, float] = {
    frozenset({"JFK", "LAX"}): 2475.0,  frozenset({"JFK", "SFO"}): 2586.0,
    frozenset({"JFK", "ORD"}): 740.0,   frozenset({"JFK", "MIA"}): 1089.0,
    frozenset({"JFK", "BOS"}): 187.0,   frozenset({"JFK", "ATL"}): 760.0,
    frozenset({"JFK", "DFW"}): 1391.0,  frozenset({"JFK", "SEA"}): 2422.0,
    frozenset({"JFK", "DEN"}): 1626.0,  frozenset({"JFK", "LAS"}): 2243.0,
    frozenset({"LAX", "SFO"}): 337.0,   frozenset({"LAX", "ORD"}): 1745.0,
    frozenset({"LAX", "MIA"}): 2342.0,  frozenset({"LAX", "ATL"}): 1946.0,
    frozenset({"LAX", "DFW"}): 1235.0,  frozenset({"LAX", "SEA"}): 954.0,
    frozenset({"LAX", "DEN"}): 862.0,   frozenset({"LAX", "LAS"}): 236.0,
    frozenset({"LAX", "PHX"}): 370.0,   frozenset({"ORD", "ATL"}): 606.0,
    frozenset({"ORD", "DFW"}): 802.0,   frozenset({"ORD", "MIA"}): 1197.0,
    frozenset({"ORD", "SEA"}): 1720.0,  frozenset({"ORD", "DEN"}): 920.0,
    frozenset({"ORD", "BOS"}): 867.0,   frozenset({"ATL", "DFW"}): 732.0,
    frozenset({"ATL", "MIA"}): 661.0,   frozenset({"ATL", "BOS"}): 946.0,
    frozenset({"ATL", "SEA"}): 2182.0,  frozenset({"DFW", "MIA"}): 1121.0,
    frozenset({"DFW", "SEA"}): 1660.0,  frozenset({"DFW", "DEN"}): 641.0,
    frozenset({"DFW", "LAS"}): 1055.0,  frozenset({"DFW", "PHX"}): 868.0,
    frozenset({"SFO", "SEA"}): 679.0,   frozenset({"SFO", "DEN"}): 967.0,
    frozenset({"SFO", "LAS"}): 414.0,   frozenset({"SFO", "PHX"}): 651.0,
    frozenset({"SEA", "DEN"}): 1024.0,  frozenset({"SEA", "LAS"}): 867.0,
    frozenset({"DEN", "LAS"}): 598.0,   frozenset({"DEN", "PHX"}): 586.0,
    frozenset({"LAS", "PHX"}): 256.0,   frozenset({"MIA", "BOS"}): 1258.0,
    frozenset({"BOS", "ORD"}): 867.0,
}

_AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    "ATL": (33.6407, -84.4277),  "BOS": (42.3656, -71.0096),
    "CLT": (35.2140, -80.9431),  "DEN": (39.8561, -104.6737),
    "DFW": (32.8998, -97.0403),  "DTW": (42.2124, -83.3534),
    "EWR": (40.6895, -74.1745),  "IAH": (29.9902, -95.3368),
    "JFK": (40.6413, -73.7781),  "LAX": (33.9425, -118.4081),
    "LAS": (36.0840, -115.1537), "LGA": (40.7769, -73.8740),
    "MCO": (28.4312, -81.3081),  "MDW": (41.7868, -87.7522),
    "MIA": (25.7959, -80.2870),  "MSP": (44.8848, -93.2223),
    "ORD": (41.9742, -87.9073),  "PDX": (45.5898, -122.5951),
    "PHL": (39.8744, -75.2424),  "PHX": (33.4373, -112.0078),
    "SAN": (32.7338, -117.1933), "SEA": (47.4502, -122.3088),
    "SFO": (37.6213, -122.3790), "SLC": (40.7884, -111.9778),
    "STL": (38.7487, -90.3700),  "TPA": (27.9755, -82.5332),
    "AUS": (30.1975, -97.6664),  "BNA": (36.1245, -86.6782),
    "HNL": (21.3187, -157.9224), "MSY": (29.9934, -90.2580),
}

_CRUISE_SPEED_MPH = 500.0
_OVERHEAD_MINUTES = 30.0


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions (mirroring backend/app/features.py)
# ═══════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _lookup_distance(origin: str, dest: str) -> Optional[float]:
    key = frozenset({origin, dest})
    if key in _ROUTE_DISTANCES:
        return _ROUTE_DISTANCES[key]
    if origin in _AIRPORT_COORDS and dest in _AIRPORT_COORDS:
        lat1, lon1 = _AIRPORT_COORDS[origin]
        lat2, lon2 = _AIRPORT_COORDS[dest]
        km = _haversine_km(lat1, lon1, lat2, lon2)
        return round(km * 0.621371, 1)
    return None


def _estimate_elapsed(distance_mi: Optional[float]) -> Optional[float]:
    if distance_mi is None:
        return None
    return round((distance_mi / _CRUISE_SPEED_MPH) * 60.0 + _OVERHEAD_MINUTES, 1)


def build_raw_df(
    airline: str,
    origin: str,
    destination: str,
    dep_datetime: datetime,
    tail_number: str = "UNKNOWN",
) -> pd.DataFrame:
    """Convert user inputs to a single-row raw DataFrame for the pipeline."""
    distance_mi = _lookup_distance(origin, destination)
    crs_elapsed = _estimate_elapsed(distance_mi)

    row = {
        "OP_CARRIER":       airline.upper().strip(),
        "ORIGIN":           origin.upper().strip(),
        "DEST":             destination.upper().strip(),
        "TAIL_NUM":         tail_number.upper().strip() if tail_number else "UNKNOWN",
        "FL_DATE":          pd.Timestamp(dep_datetime.date()),
        "Month":            dep_datetime.month,
        "DayofMonth":       dep_datetime.day,
        "DayOfWeek":        dep_datetime.isoweekday(),  # 1=Mon … 7=Sun
        "CRS_DEP_TIME":     dep_datetime.hour * 100 + dep_datetime.minute,
        "DISTANCE":         distance_mi,
        "CRS_ELAPSED_TIME": crs_elapsed,
    }
    return pd.DataFrame([row])


# ═══════════════════════════════════════════════════════════════════════════
# Model loading (cached so it survives Streamlit reruns)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load LightGBM model, feature pipeline, SHAP explainer, and params."""
    import lightgbm as lgb

    # Find latest model file
    model_files = sorted(ARTIFACTS_DIR.glob("model_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_files:
        st.error("No model file found in backend/artifacts/")
        st.stop()
    model = lgb.Booster(model_file=str(model_files[0]))
    feature_names = model.feature_name()
    logger.info("Loaded model: %s", model_files[0].name)

    # Pipeline
    pipeline_files = sorted(ARTIFACTS_DIR.glob("pipeline*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pipeline_files:
        st.error("No pipeline file found in backend/artifacts/")
        st.stop()
    pipeline = joblib.load(pipeline_files[0])
    logger.info("Loaded pipeline: %s", pipeline_files[0].name)

    # Params
    threshold = 0.5
    params_path = ARTIFACTS_DIR / "params.json"
    if params_path.exists():
        with params_path.open() as f:
            params = json.load(f)
        threshold = float(params.get("threshold", threshold))

    # SHAP explainer
    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer ready.")
    except Exception as e:
        logger.warning("SHAP unavailable: %s", e)

    return model, pipeline, explainer, feature_names, threshold


def predict_delay(
    model, pipeline, explainer, feature_names: list, threshold: float,
    raw_df: pd.DataFrame,
) -> Tuple[float, int, str, list]:
    """Run inference and return (probability, prediction, risk_level, top_factors)."""
    # Feature pipeline transform
    transformed = pipeline.transform(raw_df)

    # Align features
    if feature_names and all(f in transformed.columns for f in feature_names):
        feat_df = transformed[feature_names]
    elif feature_names and len(transformed.columns) == len(feature_names):
        feat_df = transformed.copy()
        feat_df.columns = feature_names
    else:
        feat_df = transformed

    # Predict
    raw_probs = model.predict(feat_df)
    probability = float(np.clip(raw_probs[0], 0.0, 1.0))
    prediction = int(probability >= threshold)

    # Risk level
    if probability < RISK_LOW:
        risk_level = "Low"
    elif probability <= RISK_HIGH:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # SHAP factors
    top_factors = []
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(feat_df)
            if isinstance(shap_vals, list):
                sv = np.array(shap_vals[1][0])
            else:
                sv = np.array(shap_vals[0])
            names = list(feat_df.columns)
            pairs = sorted(zip(names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
            top_factors = [{"feature": n, "impact": round(v, 6)} for n, v in pairs[:TOP_N_FACTORS]]
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

    return probability, prediction, risk_level, top_factors


# ═══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clean_feature_name(name: str) -> str:
    """Human-readable feature names."""
    return (
        name
        .replace("_te", " (encoded)")
        .replace("_sin", " (sin)")
        .replace("_cos", " (cos)")
        .replace("_", " ")
        .title()
    )


def create_gauge(probability: float) -> go.Figure:
    """Create a Plotly gauge chart showing delay probability."""
    pct = probability * 100

    if probability < 0.3:
        bar_color = "#22c55e"  # green
    elif probability <= 0.6:
        bar_color = "#eab308"  # yellow
    else:
        bar_color = "#ef4444"  # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569", "dtick": 25},
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "#334155",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(34, 197, 94, 0.15)"},
                {"range": [30, 60], "color": "rgba(234, 179, 8, 0.15)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#f1f5f9", "width": 2},
                "thickness": 0.8,
                "value": pct,
            },
        },
        title={"text": "Delay Probability", "font": {"size": 16, "color": "#94a3b8"}},
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
    )
    return fig


def create_factors_chart(factors: list) -> go.Figure:
    """Create a horizontal bar chart of SHAP factors."""
    if not factors:
        return None

    # Reverse so top factor is at top
    factors_sorted = list(reversed(factors))
    names = [_clean_feature_name(f["feature"]) for f in factors_sorted]
    values = [f["impact"] for f in factors_sorted]
    colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 12},
    ))

    fig.update_layout(
        height=max(180, len(factors) * 42),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(71, 85, 105, 0.3)",
            "zeroline": True,
            "zerolinecolor": "#475569",
            "tickfont": {"color": "#94a3b8"},
        },
        yaxis={
            "tickfont": {"color": "#cbd5e1", "size": 13},
        },
        font={"color": "#f1f5f9"},
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit App
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }
    .risk-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 1px;
    }
    .risk-low {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .risk-medium {
        background: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .legend-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("# ✈️ Flight Delay Predictor")
st.caption("LightGBM · SHAP Explanations · T-24h Rule — Predicts ≥15 min arrival delay")
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.markdown("## About")
    st.info(
        "This app predicts whether your flight will arrive **≥15 minutes late** "
        "using only information available **24 hours before departure**."
    )
    st.markdown("### How it works")
    st.markdown(
        "1. Enter your flight details\n"
        "2. Click **Predict**\n"
        "3. View delay probability & top contributing factors"
    )
    st.markdown("### Model Info")
    st.markdown(
        "- **Algorithm**: LightGBM (237 boosting rounds)\n"
        "- **Features**: 17 pre-departure features\n"
        "- **Training data**: ~7M US domestic flights (2024)\n"
        "- **Explainability**: SHAP TreeExplainer"
    )
    st.markdown("---")
    st.markdown(
        "**Note:** This is a machine learning prediction, not a guarantee. "
        "Weather and real-time operations are not included."
    )

# ── Load model ──
model, pipeline, explainer, feature_names, threshold = load_model()

# ── Input Form ──
with st.form("flight_form"):
    st.markdown("### ✏️ Flight Details")

    col1, col2 = st.columns(2)

    with col1:
        airline_display = st.selectbox(
            "Airline",
            options=list(AIRLINES.keys()),
            format_func=lambda x: f"{x} — {AIRLINES[x]}",
            index=0,
        )

        origin_display = st.selectbox(
            "Origin Airport",
            options=list(AIRPORTS.keys()),
            format_func=lambda x: AIRPORTS[x],
            index=list(AIRPORTS.keys()).index("JFK"),
        )

        dep_date = st.date_input(
            "Departure Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
        )

    with col2:
        destination_display = st.selectbox(
            "Destination Airport",
            options=list(AIRPORTS.keys()),
            format_func=lambda x: AIRPORTS[x],
            index=list(AIRPORTS.keys()).index("LAX"),
        )

        dep_time = st.time_input(
            "Departure Time",
            value=time(14, 30),
        )

        tail_number = st.text_input(
            "Tail Number (optional)",
            placeholder="e.g. N123AA",
            max_chars=10,
        )

    submitted = st.form_submit_button("🔮 Predict Delay", use_container_width=True, type="primary")


# ── Prediction ──
if submitted:
    # Validate same origin/dest
    if origin_display == destination_display:
        st.error("Origin and destination cannot be the same airport.")
        st.stop()

    # Build datetime
    dep_datetime = datetime.combine(dep_date, dep_time)

    with st.spinner("Running prediction..."):
        raw_df = build_raw_df(
            airline=airline_display,
            origin=origin_display,
            destination=destination_display,
            dep_datetime=dep_datetime,
            tail_number=tail_number or "UNKNOWN",
        )

        try:
            probability, prediction, risk_level, top_factors = predict_delay(
                model, pipeline, explainer, feature_names, threshold, raw_df
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # ── Results layout ──
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        fig_gauge = create_gauge(probability)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        # Risk badge
        risk_class = f"risk-{risk_level.lower()}"
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")
        st.markdown(
            f'<div style="text-align:center; margin-top:20px;">'
            f'<p style="color:#94a3b8; margin-bottom:8px;">Risk Level</p>'
            f'<span class="risk-badge {risk_class}">{risk_emoji} {risk_level.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Metric cards
        st.markdown("")
        m1, m2 = st.columns(2)
        m1.metric("Probability", f"{probability * 100:.1f}%")
        m2.metric("Threshold", f"{threshold * 100:.0f}%")

        # Descriptive text
        descriptions = {
            "High": "⚠️ High chance of a 15+ min arrival delay.",
            "Medium": "⏳ Moderate delay risk. Monitor before departure.",
            "Low": "✅ Low delay risk based on historical patterns.",
        }
        st.info(descriptions.get(risk_level, ""))

    # ── SHAP Factors ──
    if top_factors:
        st.markdown("---")
        st.markdown("### 🔍 Key Factors")
        st.caption("Top features contributing to this prediction (SHAP values)")

        fig_factors = create_factors_chart(top_factors)
        if fig_factors:
            st.plotly_chart(fig_factors, use_container_width=True)

        # Legend
        st.markdown(
            '<span class="legend-dot" style="background:#ef4444;"></span> Increases delay risk &nbsp;&nbsp;&nbsp;'
            '<span class="legend-dot" style="background:#3b82f6;"></span> Decreases delay risk',
            unsafe_allow_html=True,
        )

    # ── Flight summary ──
    with st.expander("📋 Flight Summary"):
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.markdown(f"**Airline:** {airline_display} ({AIRLINES[airline_display]})")
        summary_col2.markdown(f"**Route:** {origin_display} → {destination_display}")
        summary_col3.markdown(f"**Departure:** {dep_datetime.strftime('%b %d, %Y %H:%M')}")

        dist = _lookup_distance(origin_display, destination_display)
        if dist:
            elapsed = _estimate_elapsed(dist)
            st.markdown(f"**Distance:** {dist:.0f} mi &nbsp;|&nbsp; **Est. Duration:** {elapsed:.0f} min")

# ── Footer ──
st.markdown("---")
st.caption(
    "Binary classification · ≥15 min arrival delay · No post-departure data used · "
    "Built with LightGBM & Streamlit"
)
