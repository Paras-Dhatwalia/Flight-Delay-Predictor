"""Flight input form and hero empty state."""
from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any, Dict, Tuple

import streamlit as st

from config import AIRLINES, AIRPORTS


def render_hero() -> None:
    """Zero-state hero card shown before any prediction."""
    st.markdown("""
    <div class="hero-container">
        <span class="hero-icon">✈️</span>
        <div class="hero-title">Enter flight details to analyze delay risk</div>
        <div class="hero-text">
            Get an AI-powered probability estimate with SHAP explanations
            showing exactly what drives the prediction for your specific flight.
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Example injection callbacks ─────────────────────────────

def _load_example(**kwargs: Any) -> None:
    """on_click callback: inject example flight params into session state."""
    for key, value in kwargs.items():
        st.session_state[key] = value


def render_flight_form() -> Tuple[bool, Dict[str, Any]]:
    """
    Input form with example-flight injection buttons.

    Returns
    -------
    (submitted: bool, form_data: dict)
        form_data keys: airline, origin, destination, dep_date, dep_time, tail
    """
    akeys = list(AIRLINES.keys())
    pkeys = list(AIRPORTS.keys())

    # Example injection buttons — explicit on_click callbacks
    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "🎯 Try: High-Risk Flight", use_container_width=True,
            on_click=_load_example,
            kwargs=dict(
                _def_al="NK", _def_or="EWR", _def_de="ORD",
                _def_dt=date.today() + timedelta(days=22),
                _def_tm=time(19, 45), _def_tl="",
            ),
        )
    with c2:
        st.button(
            "✅ Try: Low-Risk Flight", use_container_width=True,
            on_click=_load_example,
            kwargs=dict(
                _def_al="DL", _def_or="ATL", _def_de="JFK",
                _def_dt=date.today() + timedelta(days=3),
                _def_tm=time(10, 15), _def_tl="",
            ),
        )

    # Resolve defaults from session state
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

        submitted = st.form_submit_button("Analyze Delay Risk",
                                          use_container_width=True, type="primary")

    return submitted, dict(airline=airline, origin=origin, destination=dest,
                           dep_date=dep_date, dep_time=dep_time, tail=tail)
