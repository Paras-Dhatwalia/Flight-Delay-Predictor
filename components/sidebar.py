"""Sidebar with model evaluation metrics and data provenance."""
from __future__ import annotations

import streamlit as st


def render_sidebar(params: dict) -> None:
    """Render sidebar with model metrics, data provenance, and disclaimer."""
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
