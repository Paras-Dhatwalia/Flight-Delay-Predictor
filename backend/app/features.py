"""
features.py
-----------
Online feature engineering for single-row inference.

Converts a PredictRequest (Pydantic schema) into a raw pandas DataFrame row
that mirrors the training data schema expected by the feature pipeline.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd

from .schemas import PredictRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-computed route distances (statute miles) for common US domestic routes
# ---------------------------------------------------------------------------
_ROUTE_DISTANCES: dict[frozenset, float] = {
    frozenset({"JFK", "LAX"}): 2475.0,
    frozenset({"JFK", "SFO"}): 2586.0,
    frozenset({"JFK", "ORD"}): 740.0,
    frozenset({"JFK", "MIA"}): 1089.0,
    frozenset({"JFK", "BOS"}): 187.0,
    frozenset({"JFK", "ATL"}): 760.0,
    frozenset({"JFK", "DFW"}): 1391.0,
    frozenset({"JFK", "SEA"}): 2422.0,
    frozenset({"JFK", "DEN"}): 1626.0,
    frozenset({"JFK", "LAS"}): 2243.0,
    frozenset({"LAX", "SFO"}): 337.0,
    frozenset({"LAX", "ORD"}): 1745.0,
    frozenset({"LAX", "MIA"}): 2342.0,
    frozenset({"LAX", "ATL"}): 1946.0,
    frozenset({"LAX", "DFW"}): 1235.0,
    frozenset({"LAX", "SEA"}): 954.0,
    frozenset({"LAX", "DEN"}): 862.0,
    frozenset({"LAX", "LAS"}): 236.0,
    frozenset({"LAX", "PHX"}): 370.0,
    frozenset({"ORD", "ATL"}): 606.0,
    frozenset({"ORD", "DFW"}): 802.0,
    frozenset({"ORD", "MIA"}): 1197.0,
    frozenset({"ORD", "SEA"}): 1720.0,
    frozenset({"ORD", "DEN"}): 920.0,
    frozenset({"ORD", "BOS"}): 867.0,
    frozenset({"ATL", "DFW"}): 732.0,
    frozenset({"ATL", "MIA"}): 661.0,
    frozenset({"ATL", "BOS"}): 946.0,
    frozenset({"ATL", "SEA"}): 2182.0,
    frozenset({"DFW", "MIA"}): 1121.0,
    frozenset({"DFW", "SEA"}): 1660.0,
    frozenset({"DFW", "DEN"}): 641.0,
    frozenset({"DFW", "LAS"}): 1055.0,
    frozenset({"DFW", "PHX"}): 868.0,
    frozenset({"SFO", "SEA"}): 679.0,
    frozenset({"SFO", "DEN"}): 967.0,
    frozenset({"SFO", "LAS"}): 414.0,
    frozenset({"SFO", "PHX"}): 651.0,
    frozenset({"SEA", "DEN"}): 1024.0,
    frozenset({"SEA", "LAS"}): 867.0,
    frozenset({"DEN", "LAS"}): 598.0,
    frozenset({"DEN", "PHX"}): 586.0,
    frozenset({"LAS", "PHX"}): 256.0,
    frozenset({"MIA", "BOS"}): 1258.0,
    frozenset({"BOS", "ORD"}): 867.0,
}

# Approximate coordinates for haversine fallback
_AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.6407,  -84.4277),
    "BOS": (42.3656,  -71.0096),
    "CLT": (35.2140,  -80.9431),
    "DEN": (39.8561, -104.6737),
    "DFW": (32.8998,  -97.0403),
    "DTW": (42.2124,  -83.3534),
    "EWR": (40.6895,  -74.1745),
    "IAH": (29.9902,  -95.3368),
    "JFK": (40.6413,  -73.7781),
    "LAX": (33.9425, -118.4081),
    "LAS": (36.0840, -115.1537),
    "LGA": (40.7769,  -73.8740),
    "MCO": (28.4312,  -81.3081),
    "MDW": (41.7868,  -87.7522),
    "MIA": (25.7959,  -80.2870),
    "MSP": (44.8848,  -93.2223),
    "ORD": (41.9742,  -87.9073),
    "PDX": (45.5898, -122.5951),
    "PHL": (39.8744,  -75.2424),
    "PHX": (33.4373, -112.0078),
    "SAN": (32.7338, -117.1933),
    "SEA": (47.4502, -122.3088),
    "SFO": (37.6213, -122.3790),
    "SLC": (40.7884, -111.9778),
    "STL": (38.7487,  -90.3700),
    "TPA": (27.9755,  -82.5332),
    "AUS": (30.1975,  -97.6664),
    "BNA": (36.1245,  -86.6782),
    "HNL": (21.3187, -157.9224),
    "MSY": (29.9934,  -90.2580),
}

_CRUISE_SPEED_MPH = 500.0
_OVERHEAD_MINUTES = 30.0


def request_to_raw_df(request: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest to a single-row raw DataFrame.

    Column mapping
    --------------
    OP_CARRIER       <- request.airline
    ORIGIN           <- request.origin
    DEST             <- request.destination
    TAIL_NUM         <- request.tail_number ("UNKNOWN" when None)
    FL_DATE          <- date portion of scheduled_departure
    Month            <- scheduled_departure.month
    DayofMonth       <- scheduled_departure.day
    DayOfWeek        <- ISO weekday 1-7 (Mon=1, Sun=7)
    CRS_DEP_TIME     <- HHMM integer (e.g. 14:30 → 1430)
    DISTANCE         <- statute miles (lookup or haversine fallback)
    CRS_ELAPSED_TIME <- estimated block time in minutes
    """
    dep: datetime = request.scheduled_departure
    distance_mi = _lookup_distance(request.origin, request.destination)
    crs_elapsed = _estimate_elapsed(distance_mi)

    row = {
        "OP_CARRIER":     request.airline,
        "ORIGIN":         request.origin,
        "DEST":           request.destination,
        "TAIL_NUM":       request.tail_number or "UNKNOWN",
        "FL_DATE":        pd.Timestamp(dep.date()),
        "Month":          dep.month,
        "DayofMonth":     dep.day,
        "DayOfWeek":      dep.isoweekday(),  # 1=Mon … 7=Sun
        "CRS_DEP_TIME":   dep.hour * 100 + dep.minute,
        "DISTANCE":       distance_mi,
        "CRS_ELAPSED_TIME": crs_elapsed,
    }

    logger.debug("request_to_raw_df: %s", row)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _lookup_distance(origin: str, dest: str) -> Optional[float]:
    """Lookup statute-mile distance for an O-D pair."""
    key = frozenset({origin, dest})
    if key in _ROUTE_DISTANCES:
        return _ROUTE_DISTANCES[key]
    if origin in _AIRPORT_COORDS and dest in _AIRPORT_COORDS:
        lat1, lon1 = _AIRPORT_COORDS[origin]
        lat2, lon2 = _AIRPORT_COORDS[dest]
        km = _haversine_km(lat1, lon1, lat2, lon2)
        miles = km * 0.621371
        logger.debug("Haversine distance %s→%s: %.1f mi", origin, dest, miles)
        return round(miles, 1)
    logger.warning("Unknown route %s→%s; DISTANCE will be None.", origin, dest)
    return None


def _estimate_elapsed(distance_mi: Optional[float]) -> Optional[float]:
    """Estimate scheduled block time in minutes from distance."""
    if distance_mi is None:
        return None
    return round((distance_mi / _CRUISE_SPEED_MPH) * 60.0 + _OVERHEAD_MINUTES, 1)
