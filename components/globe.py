"""Interactive 3D globe visualization using globe.gl."""
from __future__ import annotations

import json

import streamlit as st

from config import _AIRPORT_COORDS, AIRLINE_BRANDS, _DEFAULT_BRAND


def render_route_globe(origin: str, dest: str, airline: str) -> None:
    """
    Render an interactive 3D globe with the flight route arc.

    Uses JSON serialization for safe data injection into the JS template.
    globe.gl + Three.js loaded from CDN.
    """
    import streamlit.components.v1 as components

    o_coords = _AIRPORT_COORDS.get(origin)
    d_coords = _AIRPORT_COORDS.get(dest)
    if not o_coords or not d_coords:
        st.caption("Globe unavailable — airport coordinates not found.")
        return

    brand = AIRLINE_BRANDS.get(airline, _DEFAULT_BRAND)

    # Serialize config as JSON for safe injection (no NaN / quote risks)
    cfg = json.dumps({
        "origin":   {"lat": float(o_coords[0]), "lng": float(o_coords[1]), "code": origin},
        "dest":     {"lat": float(d_coords[0]), "lng": float(d_coords[1]), "code": dest},
        "midLat":   float((o_coords[0] + d_coords[0]) / 2),
        "midLng":   float((o_coords[1] + d_coords[1]) / 2),
        "arcColor": brand["color"],
    })

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
      const CFG = JSON.parse('{cfg}');
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

      // Arc
      globe
        .arcsData([{{
          startLat: CFG.origin.lat, startLng: CFG.origin.lng,
          endLat: CFG.dest.lat,     endLng: CFG.dest.lng,
          color: [CFG.arcColor, '#38bdf8']
        }}])
        .arcColor('color')
        .arcAltitudeAutoScale(0.45)
        .arcStroke(1.5)
        .arcDashLength(0.6)
        .arcDashGap(0.3)
        .arcDashAnimateTime(2500);

      // Airport markers
      const points = [
        {{ lat: CFG.origin.lat, lng: CFG.origin.lng, label: CFG.origin.code, size: 0.6, color: '#22c55e' }},
        {{ lat: CFG.dest.lat,   lng: CFG.dest.lng,   label: CFG.dest.code,   size: 0.6, color: '#ef4444' }}
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

      // Camera — frame both airports
      globe.pointOfView({{ lat: CFG.midLat, lng: CFG.midLng, altitude: 2.0 }}, 1000);

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
