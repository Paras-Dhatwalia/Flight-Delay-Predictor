"""Header and footer UI components."""
from __future__ import annotations

import streamlit as st


def render_header(loaded: bool) -> None:
    """Top header with app title and optional 'Model Loaded' status badge."""
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


def render_footer() -> None:
    """Bottom footer with model provenance info."""
    st.markdown("""
    <div class="app-footer">
        Binary classification · ≥15 min arrival delay · T-24h pre-departure features only ·
        LightGBM + SHAP · Trained on BTS 2024 data
    </div>""", unsafe_allow_html=True)
