"""
Flight Delay Predictor — Streamlit Application
================================================
Slim orchestrator. Configuration lives in config.py, ML logic in core.py,
and all UI components in components/.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from core import (
    load_model,
    build_raw_df,
    run_prediction,
    _lookup_distance,
    _estimate_elapsed,
)
from components import (
    render_header,
    render_footer,
    render_sidebar,
    render_hero,
    render_flight_form,
    render_boarding_pass,
    render_risk_banner,
    render_shap_section,
    render_route_globe,
    create_gauge,
)


def main() -> None:
    st.set_page_config(
        page_title="Flight Delay Predictor",
        page_icon="✈️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # ── Inject centralized CSS ────────────────────────────────
    _css = (Path(__file__).resolve().parent / "assets" / "style.css").read_text()
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

    # ── Session state init ────────────────────────────────────
    if "prediction_run" not in st.session_state:
        st.session_state.prediction_run = False

    # ── Load model (cached across reruns) ─────────────────────
    with st.spinner("Initializing model and SHAP explainer..."):
        model, pipeline, explainer, feat_names, threshold, params = load_model()

    # ── Shell ─────────────────────────────────────────────────
    render_header(loaded=True)
    render_sidebar(params)
    st.markdown("---")

    # ── Form ──────────────────────────────────────────────────
    submitted, fd = render_flight_form()

    # ── Handle submission with granular status ────────────────
    if submitted:
        if fd["origin"] == fd["destination"]:
            st.error("Origin and destination cannot be the same airport.")
            st.stop()

        dep_dt = datetime.combine(fd["dep_date"], fd["dep_time"])

        with st.status("Analyzing flight...", expanded=True) as status:
            st.write("Building input features...")
            raw_df = build_raw_df(
                fd["airline"], fd["origin"], fd["destination"],
                dep_dt, fd["tail"] or "UNKNOWN",
            )

            st.write("Running model inference...")
            try:
                results = run_prediction(
                    model, pipeline, explainer,
                    feat_names, threshold, raw_df,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            st.write("Computing SHAP explanations...")
            status.update(label="Analysis complete!", state="complete")

        st.session_state.prediction_run = True
        st.session_state.results = results
        st.session_state.form_data = fd
        st.session_state.distance = _lookup_distance(fd["origin"], fd["destination"])
        st.session_state.elapsed = _estimate_elapsed(st.session_state.distance)

    # ── Render results or empty state ─────────────────────────
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
            st.plotly_chart(
                create_gauge(res["probability"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with r_col:
            render_risk_banner(
                res["risk_level"], res["probability"],
                res["route_rate"], res["airline_rate"], res["threshold"],
            )

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
