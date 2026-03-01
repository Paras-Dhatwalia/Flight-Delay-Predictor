"""
geospatial.py
-------------
Geospatial features for flight delay prediction.

Provides:
  - AIRPORT_COORDS: lookup dictionary of major US airport coordinates.
  - haversine_km(): vectorised great-circle distance computation.
  - add_route_distance(): adds ``route_distance_km`` to a flight dataframe.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Airport coordinate dictionary  (IATA -> (latitude_deg, longitude_deg))
# ---------------------------------------------------------------------------
AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    # Tier-1 hubs
    "ATL": (33.6407,  -84.4277),
    "LAX": (33.9425, -118.4081),
    "ORD": (41.9742,  -87.9073),
    "DFW": (32.8998,  -97.0403),
    "DEN": (39.8561, -104.6737),
    "JFK": (40.6413,  -73.7781),
    "SFO": (37.6213, -122.3790),
    "SEA": (47.4502, -122.3088),
    "LAS": (36.0840, -115.1537),
    "MCO": (28.4312,  -81.3081),
    # Major hubs
    "CLT": (35.2140,  -80.9431),
    "PHX": (33.4373, -112.0078),
    "MIA": (25.7959,  -80.2870),
    "IAH": (29.9902,  -95.3368),
    "EWR": (40.6895,  -74.1745),
    "MSP": (44.8848,  -93.2223),
    "BOS": (42.3656,  -71.0096),
    "DTW": (42.2124,  -83.3534),
    "PHL": (39.8744,  -75.2424),
    "LGA": (40.7769,  -73.8740),
    # Secondary hubs
    "MDW": (41.7868,  -87.7522),
    "FLL": (26.0726,  -80.1527),
    "BWI": (39.1754,  -76.6683),
    "SLC": (40.7884, -111.9778),
    "SAN": (32.7338, -117.1933),
    "TPA": (27.9755,  -82.5332),
    "PDX": (45.5898, -122.5951),
    "STL": (38.7487,  -90.3700),
    "HOU": (29.6454,  -95.2789),
    "OAK": (37.7213, -122.2208),
    "MCI": (39.2976,  -94.7139),
    "RDU": (35.8776,  -78.7875),
    "MKE": (42.9472,  -87.8966),
    "AUS": (30.1975,  -97.6664),
    "SMF": (38.6954, -121.5908),
    "SNA": (33.6757, -117.8682),
    "MSY": (29.9934,  -90.2580),
    "SJC": (37.3626, -121.9290),
    "DAL": (32.8471,  -96.8518),
    "HNL": (21.3187, -157.9224),
    "BNA": (36.1245,  -86.6782),
    "IND": (39.7173,  -86.2944),
    "PIT": (40.4915,  -80.2329),
    "CVG": (39.0488,  -84.6678),
    "CMH": (39.9980,  -82.8919),
    "SAT": (29.5337,  -98.4698),
    "RSW": (26.5362,  -81.7552),
    "BUR": (34.2007, -118.3590),
    "ONT": (34.0559, -117.6012),
    "ABQ": (35.0402, -106.6090),
    "TUL": (36.1984,  -95.8881),
    "OKC": (35.3931,  -97.6007),
    "OMA": (41.3032,  -95.8941),
    "RIC": (37.5052,  -77.3197),
    "ORF": (36.8976,  -76.0178),
    "BDL": (41.9389,  -72.6832),
    "ALB": (42.7483,  -73.8017),
    "PVD": (41.7272,  -71.4282),
    "BUF": (42.9405,  -78.7322),
    "ROC": (43.1189,  -77.6724),
    "SYR": (43.1112,  -76.1063),
    "MHT": (42.9326,  -71.4357),
    "PWM": (43.6462,  -70.3093),
    "GRR": (42.8808,  -85.5228),
    "CLE": (41.4117,  -81.8498),
    "TYS": (35.8110,  -83.9940),
    "BHM": (33.5629,  -86.7535),
    "MEM": (35.0424,  -89.9767),
    "GSP": (34.8957,  -82.2189),
    "SAV": (32.1276,  -81.2021),
    "JAX": (30.4941,  -81.6879),
    "PNS": (30.4734,  -87.1866),
    "ELP": (31.8072, -106.3779),
    "LBB": (33.6636, -101.8228),
    "AMA": (35.2194, -101.7059),
    "MAF": (31.9425, -102.2019),
    "XNA": (36.2819,  -94.3068),
    "SGF": (37.2457,  -93.3886),
    "ICT": (37.6499,  -97.4331),
    "BOI": (43.5644, -116.2228),
    "GEG": (47.6199, -117.5339),
    "FAT": (36.7762, -119.7182),
    "SBA": (34.4262, -119.8404),
    "LGB": (33.8177, -118.1516),
    "TUS": (32.1161, -110.9410),
    "MFR": (42.3742, -122.8735),
    "EUG": (44.1246, -123.2119),
    "JAC": (43.6073, -110.7377),
    "BZN": (45.7777, -111.1531),
    "MSO": (46.9163, -114.0906),
    "FCA": (48.3105, -114.2560),
    "GTF": (47.4820, -111.3707),
    "BIL": (45.8077, -108.5428),
    "PIH": (42.9098, -112.5958),
    "IDA": (43.5146, -112.0702),
    "DAY": (39.9024,  -84.2194),
    "CAK": (40.9161,  -81.4422),
    "HSV": (34.6372,  -86.7751),
    "MOB": (30.6912,  -88.2428),
    "BTR": (30.5332,  -91.1496),
    "SHV": (32.4466,  -93.8256),
    "LIT": (34.7294,  -92.2243),
    "CHA": (35.0353,  -85.2038),
    "GSO": (36.0978,  -79.9373),
    "AVL": (35.4362,  -82.5418),
    "ILM": (34.2706,  -77.9026),
    "FAY": (34.9912,  -78.8803),
    "AGS": (33.3699,  -81.9645),
    "TLH": (30.3965,  -84.3503),
    "VPS": (30.4832,  -86.5254),
    "CRP": (27.7704,  -97.5012),
}

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute great-circle distance(s) in kilometres.

    Fully vectorised via NumPy; accepts scalars or broadcast-compatible arrays.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return _EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_route_distance(
    df: pd.DataFrame,
    airport_coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Add ``route_distance_km`` column to a flight dataframe.

    Lookup order:
    1. Haversine from ``airport_coords`` when both airports are known.
    2. Fallback: ``DISTANCE`` column (statute miles) × 1.60934.
    3. NaN if both above are unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ORIGIN`` and ``DEST`` string columns.
    airport_coords : dict, optional
        IATA → (lat, lon). Defaults to built-in ``AIRPORT_COORDS``.

    Returns
    -------
    pd.DataFrame
        Same dataframe with ``route_distance_km`` appended.
    """
    if airport_coords is None:
        airport_coords = AIRPORT_COORDS
    if "ORIGIN" not in df.columns or "DEST" not in df.columns:
        raise KeyError("Dataframe must contain 'ORIGIN' and 'DEST' columns.")

    _MILES_TO_KM = 1.60934

    # Build vectorized coordinate lookup via pd.Series.map (fast for millions of rows)
    lat_map = {code: coord[0] for code, coord in airport_coords.items()}
    lon_map = {code: coord[1] for code, coord in airport_coords.items()}

    origin_series = df["ORIGIN"]
    dest_series   = df["DEST"]

    lat1 = origin_series.map(lat_map).values.astype(float)
    lon1 = origin_series.map(lon_map).values.astype(float)
    lat2 = dest_series.map(lat_map).values.astype(float)
    lon2 = dest_series.map(lon_map).values.astype(float)

    both_known = ~(np.isnan(lat1) | np.isnan(lat2))
    distances  = np.full(len(df), np.nan)
    if both_known.any():
        distances[both_known] = haversine_km(
            lat1[both_known], lon1[both_known],
            lat2[both_known], lon2[both_known],
        )

    needs_fallback = np.isnan(distances)
    n_fallback = int(needs_fallback.sum())
    if n_fallback > 0:
        if "DISTANCE" in df.columns:
            fallback_idx = np.where(needs_fallback)[0]
            distances[needs_fallback] = (
                df["DISTANCE"].iloc[fallback_idx].astype(float).values * _MILES_TO_KM
            )
            logger.debug("%d rows used DISTANCE-column fallback.", n_fallback)
        else:
            logger.warning(
                "%d rows have unknown coordinates and no DISTANCE fallback; "
                "route_distance_km will be NaN.", n_fallback,
            )

    df["route_distance_km"] = distances
    n_haversine = int(both_known.sum())
    logger.info(
        "route_distance_km added: %d haversine, %d fallback, %d NaN.",
        n_haversine, n_fallback if "DISTANCE" in df.columns else 0,
        int(np.isnan(distances).sum()),
    )
    return df
