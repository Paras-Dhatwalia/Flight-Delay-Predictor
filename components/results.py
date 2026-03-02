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
    """Airline-branded boarding pass card matching real boarding pass styles."""
    al = fd["airline"]; ori = fd["origin"]; dst = fd["destination"]
    dt = datetime.combine(fd["dep_date"], fd["dep_time"])
    al_name = AIRLINES.get(al, al)
    ori_city = AIRPORTS.get(ori, ori).split("(")[0].strip()
    dst_city = AIRPORTS.get(dst, dst).split("(")[0].strip()
    d_s = f"{dist:,.0f} mi" if dist else "—"
    e_s = f"~{elapsed:.0f} min" if elapsed else "—"

    brand = AIRLINE_BRANDS.get(al, _DEFAULT_BRAND)
    bc = brand["color"]
    bc2 = brand.get("color2", bc)
    tc = brand.get("text", "#fff")
    logo_url = brand["logo"]

    # Determine if header background is light (for dark text) or dark (for white text)
    is_light_bg = tc != "#fff"

    logo_html = (
        f'<img src="{logo_url}" alt="{al}" '
        f'style="width:32px;height:32px;border-radius:6px;object-fit:contain;'
        f'background:#fff;padding:2px;" onerror="this.style.display=\'none\'">'
        if logo_url else ""
    )

    bp_html = (
        f'<div style="border-radius:14px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.13);margin:18px 0 10px;max-width:540px;border:1.5px solid {bc}30;">'
        # Header band
        f'<div style="background:linear-gradient(135deg, {bc}, {bc2});padding:16px 22px;display:flex;align-items:center;justify-content:space-between;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'{logo_html}'
        f'<span style="font-size:18px;font-weight:800;color:{tc};letter-spacing:0.5px;text-transform:uppercase;">{al_name}</span>'
        f'</div>'
        f'<span style="font-size:11px;font-weight:600;color:{tc};opacity:0.8;letter-spacing:1px;text-transform:uppercase;">Delay Analysis</span>'
        f'</div>'
        # Route section
        f'<div style="padding:20px 22px 8px;background:#fff;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<div style="text-align:left;">'
        f'<div style="font-size:32px;font-weight:800;color:#1a1a1a;letter-spacing:2px;">{ori}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:2px;">{ori_city}</div>'
        f'</div>'
        f'<div style="flex:1;text-align:center;position:relative;margin:0 18px;">'
        f'<div style="border-top:2px dashed {bc}60;width:100%;position:absolute;top:50%;left:0;"></div>'
        f'<span style="font-size:22px;position:relative;z-index:1;background:#fff;padding:0 8px;color:{bc};">✈</span>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:32px;font-weight:800;color:#1a1a1a;letter-spacing:2px;">{dst}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:2px;">{dst_city}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        # Details strip
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;padding:14px 22px 18px;background:#fff;border-top:1px dashed #e0e0e0;">'
        f'<div><div style="font-size:9px;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1px;">Date</div>'
        f'<div style="font-size:14px;font-weight:700;color:#1a1a1a;margin-top:3px;">{dt.strftime("%b %d")}</div></div>'
        f'<div><div style="font-size:9px;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1px;">Departs</div>'
        f'<div style="font-size:14px;font-weight:700;color:#1a1a1a;margin-top:3px;">{dt.strftime("%H:%M")}</div></div>'
        f'<div><div style="font-size:9px;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1px;">Distance</div>'
        f'<div style="font-size:14px;font-weight:700;color:#1a1a1a;margin-top:3px;">{d_s}</div></div>'
        f'<div><div style="font-size:9px;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1px;">Duration</div>'
        f'<div style="font-size:14px;font-weight:700;color:#1a1a1a;margin-top:3px;">{e_s}</div></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(bp_html, unsafe_allow_html=True)


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
