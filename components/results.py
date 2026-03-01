"""Post-prediction result components: boarding pass and risk banner."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

from config import AIRLINES, AIRPORTS, AIRLINE_BRANDS, _DEFAULT_BRAND


def render_boarding_pass(
    fd: dict,
    dist: Optional[float],
    elapsed: Optional[float],
) -> None:
    """Airline-branded boarding pass card."""
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


def render_risk_banner(
    risk: str,
    prob: float,
    route_rate: Optional[float],
    airline_rate: Optional[float],
    threshold: float,
) -> None:
    """Color-coded risk verdict with contextual airline/route statistics."""
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
        "medium": "Consider buffer time for connections. Monitor flight status closer to departure.",
        "high":   "Build extra buffer time. Consider flexible booking options.",
    }
    title, desc = verdicts.get(lv, verdicts["medium"])
    tip = tips.get(lv, tips["medium"])

    ctx: list = []
    if route_rate is not None:
        pct = (1 - route_rate) * 100
        ctx.append(f"This route historically has a <strong>{pct:.0f}%</strong> on-time rate")
    if airline_rate is not None:
        pct = (1 - airline_rate) * 100
        ctx.append(f"this airline has a <strong>{pct:.0f}%</strong> on-time rate")
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
