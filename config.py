"""
config.py — Application-wide constants and lookup data.

Pure data module with zero external dependencies. Importable by any module
without triggering Streamlit initialization or heavy ML imports.
"""
from __future__ import annotations

from typing import Dict, Tuple

# ─── Risk Thresholds ─────────────────────────────────────────
RISK_LOW: float = 0.3
RISK_HIGH: float = 0.6

# ─── Airlines ────────────────────────────────────────────────
AIRLINES: Dict[str, str] = {
    "AA": "American Airlines",  "DL": "Delta Air Lines",
    "UA": "United Airlines",    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",      "HA": "Hawaiian Airlines",
    "SY": "Sun Country Airlines",
}

# ─── Airports ────────────────────────────────────────────────
AIRPORTS: Dict[str, str] = {
    "ATL": "Atlanta (ATL)",           "AUS": "Austin (AUS)",
    "BNA": "Nashville (BNA)",         "BOS": "Boston (BOS)",
    "CLT": "Charlotte (CLT)",         "DEN": "Denver (DEN)",
    "DFW": "Dallas/Fort Worth (DFW)", "DTW": "Detroit (DTW)",
    "EWR": "Newark (EWR)",           "HNL": "Honolulu (HNL)",
    "IAH": "Houston (IAH)",          "JFK": "New York JFK (JFK)",
    "LAS": "Las Vegas (LAS)",        "LAX": "Los Angeles (LAX)",
    "LGA": "New York LaGuardia (LGA)","MCO": "Orlando (MCO)",
    "MDW": "Chicago Midway (MDW)",    "MIA": "Miami (MIA)",
    "MSP": "Minneapolis (MSP)",       "MSY": "New Orleans (MSY)",
    "ORD": "Chicago O'Hare (ORD)",    "PDX": "Portland (PDX)",
    "PHL": "Philadelphia (PHL)",      "PHX": "Phoenix (PHX)",
    "SAN": "San Diego (SAN)",         "SEA": "Seattle (SEA)",
    "SFO": "San Francisco (SFO)",     "SLC": "Salt Lake City (SLC)",
    "STL": "St. Louis (STL)",         "TPA": "Tampa (TPA)",
}

# ─── Feature Name Translation ────────────────────────────────
FEATURE_DISPLAY: Dict[str, str] = {
    "CRS_DEP_TIME_sin":  "Departure time pattern",
    "CRS_DEP_TIME_cos":  "Departure time pattern",
    "DayOfWeek_sin":     "Day of week effect",
    "DayOfWeek_cos":     "Day of week effect",
    "Month_sin":         "Seasonal trend",
    "Month_cos":         "Seasonal trend",
    "departure_hour":    "Hour of departure",
    "route_distance_km": "Route distance",
    "DISTANCE":          "Flight distance",
    "CRS_ELAPSED_TIME":  "Scheduled duration",
    "OP_CARRIER_te":     "Airline reliability",
    "ORIGIN_te":         "Origin airport tendency",
    "DEST_te":           "Destination airport tendency",
    "TAIL_NUM_te":       "Aircraft history",
    "route_delay_rate":  "Route delay history",
    "airline_delay_rate":"Airline delay history",
    "is_near_holiday":   "Holiday proximity",
}

# ─── Airline Visual Branding ─────────────────────────────────
AIRLINE_BRANDS: Dict[str, Dict[str, str]] = {
    "AA": {"color": "#0078D2", "accent": "rgba(0,120,210,0.12)",
            "logo": "https://logo.clearbit.com/aa.com"},
    "DL": {"color": "#C01933", "accent": "rgba(192,25,51,0.12)",
            "logo": "https://logo.clearbit.com/delta.com"},
    "UA": {"color": "#005DAA", "accent": "rgba(0,93,170,0.12)",
            "logo": "https://logo.clearbit.com/united.com"},
    "WN": {"color": "#F9B612", "accent": "rgba(249,182,18,0.12)",
            "logo": "https://logo.clearbit.com/southwest.com"},
    "B6": {"color": "#003876", "accent": "rgba(0,56,118,0.12)",
            "logo": "https://logo.clearbit.com/jetblue.com"},
    "AS": {"color": "#01426A", "accent": "rgba(1,66,106,0.12)",
            "logo": "https://logo.clearbit.com/alaskaair.com"},
    "NK": {"color": "#FFE500", "accent": "rgba(255,229,0,0.12)",
            "logo": "https://logo.clearbit.com/spirit.com"},
    "F9": {"color": "#00B140", "accent": "rgba(0,177,64,0.12)",
            "logo": "https://logo.clearbit.com/flyfrontier.com"},
    "G4": {"color": "#003B70", "accent": "rgba(0,59,112,0.12)",
            "logo": "https://logo.clearbit.com/allegiantair.com"},
    "HA": {"color": "#6B2FA0", "accent": "rgba(107,47,160,0.12)",
            "logo": "https://logo.clearbit.com/hawaiianairlines.com"},
    "SY": {"color": "#F7931E", "accent": "rgba(247,147,30,0.12)",
            "logo": "https://logo.clearbit.com/suncountry.com"},
}
_DEFAULT_BRAND: Dict[str, str] = {
    "color": "#38bdf8", "accent": "rgba(56,189,248,0.12)", "logo": ""
}

# ─── Pre-computed Route Distances (miles) ────────────────────
_ROUTE_DISTANCES: Dict[frozenset, float] = {
    frozenset({"JFK","LAX"}):2475, frozenset({"JFK","SFO"}):2586,
    frozenset({"JFK","ORD"}):740,  frozenset({"JFK","MIA"}):1089,
    frozenset({"JFK","BOS"}):187,  frozenset({"JFK","ATL"}):760,
    frozenset({"JFK","DFW"}):1391, frozenset({"JFK","SEA"}):2422,
    frozenset({"JFK","DEN"}):1626, frozenset({"JFK","LAS"}):2243,
    frozenset({"LAX","SFO"}):337,  frozenset({"LAX","ORD"}):1745,
    frozenset({"LAX","MIA"}):2342, frozenset({"LAX","ATL"}):1946,
    frozenset({"LAX","DFW"}):1235, frozenset({"LAX","SEA"}):954,
    frozenset({"LAX","DEN"}):862,  frozenset({"LAX","LAS"}):236,
    frozenset({"LAX","PHX"}):370,  frozenset({"ORD","ATL"}):606,
    frozenset({"ORD","DFW"}):802,  frozenset({"ORD","MIA"}):1197,
    frozenset({"ORD","SEA"}):1720, frozenset({"ORD","DEN"}):920,
    frozenset({"ORD","BOS"}):867,  frozenset({"ATL","DFW"}):732,
    frozenset({"ATL","MIA"}):661,  frozenset({"ATL","BOS"}):946,
    frozenset({"ATL","SEA"}):2182, frozenset({"DFW","MIA"}):1121,
    frozenset({"DFW","SEA"}):1660, frozenset({"DFW","DEN"}):641,
    frozenset({"DFW","LAS"}):1055, frozenset({"DFW","PHX"}):868,
    frozenset({"SFO","SEA"}):679,  frozenset({"SFO","DEN"}):967,
    frozenset({"SFO","LAS"}):414,  frozenset({"SFO","PHX"}):651,
    frozenset({"SEA","DEN"}):1024, frozenset({"SEA","LAS"}):867,
    frozenset({"DEN","LAS"}):598,  frozenset({"DEN","PHX"}):586,
    frozenset({"LAS","PHX"}):256,  frozenset({"MIA","BOS"}):1258,
    frozenset({"BOS","ORD"}):867,
}

# ─── Airport Coordinates ─────────────────────────────────────
_AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    "ATL":(33.64,-84.43),  "BOS":(42.37,-71.01),  "CLT":(35.21,-80.94),
    "DEN":(39.86,-104.67), "DFW":(32.90,-97.04),  "DTW":(42.21,-83.35),
    "EWR":(40.69,-74.17),  "IAH":(29.99,-95.34),  "JFK":(40.64,-73.78),
    "LAX":(33.94,-118.41), "LAS":(36.08,-115.15), "LGA":(40.78,-73.87),
    "MCO":(28.43,-81.31),  "MDW":(41.79,-87.75),  "MIA":(25.80,-80.29),
    "MSP":(44.88,-93.22),  "ORD":(41.97,-87.91),  "PDX":(45.59,-122.60),
    "PHL":(39.87,-75.24),  "PHX":(33.44,-112.01), "SAN":(32.73,-117.19),
    "SEA":(47.45,-122.31), "SFO":(37.62,-122.38), "SLC":(40.79,-111.98),
    "STL":(38.75,-90.37),  "TPA":(27.98,-82.53),  "AUS":(30.20,-97.67),
    "BNA":(36.12,-86.68),  "HNL":(21.32,-157.92), "MSY":(29.99,-90.26),
}
