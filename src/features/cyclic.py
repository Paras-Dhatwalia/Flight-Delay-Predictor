"""
cyclic.py
---------
Cyclic (sin/cos) encoding for periodic temporal variables.

    x_sin = sin(2π * t / T)
    x_cos = cos(2π * t / T)

This ensures that the model sees the natural wrap-around structure
(e.g. 23:59 is close to 00:00, Sunday is close to Monday).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def encode_cyclic(df: pd.DataFrame, col: str, period: float) -> pd.DataFrame:
    """Add sin/cos cyclic encodings for a periodic column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Modified in-place.
    col : str
        Name of the column to encode.
    period : float
        Full cycle length (e.g. 1440 for minutes-in-a-day, 7 for day-of-week).

    Returns
    -------
    pd.DataFrame
        Dataframe with ``{col}_sin`` and ``{col}_cos`` columns appended.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")
    if period <= 0:
        raise ValueError(f"period must be > 0, got {period}")

    angle = 2.0 * np.pi * df[col] / period
    df[f"{col}_sin"] = np.sin(angle)
    df[f"{col}_cos"] = np.cos(angle)
    logger.debug("Cyclic-encoded '%s' (period=%s) → '%s_sin', '%s_cos'", col, period, col, col)
    return df


def add_temporal_features(
    df: pd.DataFrame,
    dep_time_col: str = "CRS_DEP_TIME",
) -> pd.DataFrame:
    """Derive and cyclic-encode temporal features from BTS flight data.

    Expected input columns
    ----------------------
    CRS_DEP_TIME : int
        Scheduled departure time as HHMM integer (e.g. 1430 = 14:30).
    DayOfWeek : int
        Day of week (BTS convention: 1=Monday … 7=Sunday).
    Month : int
        Calendar month, 1–12.

    New columns added
    -----------------
    dep_minutes_since_midnight : int
    CRS_DEP_TIME_sin, CRS_DEP_TIME_cos : float  (period=1440)
    DayOfWeek_sin, DayOfWeek_cos : float          (period=7)
    Month_sin, Month_cos : float                  (period=12)
    departure_hour : int

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, modified in-place.
    dep_time_col : str
        Name of the scheduled departure time column.

    Returns
    -------
    pd.DataFrame
        Dataframe with all new temporal feature columns.
    """
    required = {dep_time_col, "DayOfWeek", "Month"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Convert HHMM integer to minutes since midnight
    hhmm = df[dep_time_col].astype(int)
    hours   = hhmm // 100
    minutes = hhmm % 100
    dep_minutes = hours * 60 + minutes

    # Clamp any out-of-range values (e.g. 2500 from bad data)
    invalid = (dep_minutes < 0) | (dep_minutes >= 1440)
    if invalid.any():
        logger.warning(
            "%d rows have out-of-range %s values; clamping to [0, 1439].",
            int(invalid.sum()), dep_time_col,
        )
        dep_minutes = dep_minutes.clip(0, 1439)

    df["dep_minutes_since_midnight"] = dep_minutes
    df["departure_hour"] = hours.clip(0, 23).astype(int)

    # Cyclic encode departure time (period = 1440 minutes)
    angle_dep = 2.0 * np.pi * dep_minutes / 1440.0
    df[f"{dep_time_col}_sin"] = np.sin(angle_dep)
    df[f"{dep_time_col}_cos"] = np.cos(angle_dep)

    # Cyclic encode day of week and month
    df = encode_cyclic(df, "DayOfWeek", period=7.0)
    df = encode_cyclic(df, "Month", period=12.0)

    logger.info(
        "Temporal features added: %s_sin/cos, DayOfWeek_sin/cos, "
        "Month_sin/cos, departure_hour, dep_minutes_since_midnight.",
        dep_time_col,
    )
    return df
