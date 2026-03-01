"""Plotly visualizations: gauge, SHAP waterfall, and explanation section."""
from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from config import FEATURE_DISPLAY
from core import sigmoid


def create_gauge(prob: float) -> go.Figure:
    """Semi-circle gauge chart for delay probability."""
    color = "#22c55e" if prob < 0.3 else ("#eab308" if prob < 0.6 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 42}},
        title={"text": "Delay Probability", "font": {"color": "#94a3b8", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#334155",
                     "tickfont": {"color": "#475569"}, "dtick": 20},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "#1e293b", "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "rgba(34,197,94,0.1)"},
                {"range": [30, 60],  "color": "rgba(234,179,8,0.1)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.1)"},
            ],
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1f5f9"},
    )
    return fig


def create_waterfall(all_shap: Dict[str, float], base_value: float) -> go.Figure:
    """SHAP additive-attribution waterfall chart (log-odds space)."""
    sorted_feats = sorted(all_shap.items(), key=lambda x: abs(x[1]))
    n_show = min(7, len(sorted_feats))
    top = sorted_feats[-n_show:]
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


def render_shap_section(
    all_shap: Dict[str, float],
    base_value: Optional[float],
    prob: float,
) -> None:
    """Full SHAP explanation section: waterfall chart + legend + expander."""
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
