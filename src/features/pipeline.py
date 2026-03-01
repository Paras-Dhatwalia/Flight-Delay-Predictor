"""
pipeline.py
-----------
Full feature engineering pipeline for flight delay prediction.

Orchestrates all feature-building steps in consistent order for both
training (fit_transform) and inference (transform):

  1. Temporal cyclic features  (departure time, day-of-week, month)
  2. Haversine route distance
  3. Historical delay aggregates  (route and airline delay rates)
  4. K-Fold smoothed target encoding  (carrier, origin, dest, tail number)
  5. Holiday proximity flag
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd

from .cyclic import add_temporal_features
from .geospatial import add_route_distance
from .target_encoding import DEFAULT_ENCODE_COLS, KFoldTargetEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US Federal Holiday helpers
# ---------------------------------------------------------------------------
_FIXED_HOLIDAYS = [(1, 1), (7, 4), (11, 11), (12, 25)]  # (month, day)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Return the n-th occurrence (1-indexed, or -1 for last) of weekday in month/year."""
    if n > 0:
        first = pd.Timestamp(year, month, 1)
        delta = (weekday - first.weekday()) % 7
        return first + pd.Timedelta(days=int(delta)) + pd.Timedelta(weeks=n - 1)
    # n == -1: last occurrence
    last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    delta = (last.weekday() - weekday) % 7
    return last - pd.Timedelta(days=int(delta))


def _variable_holidays(year: int) -> List[pd.Timestamp]:
    return [
        _nth_weekday_of_month(year, 1,  0, 3),   # MLK Day: 3rd Mon Jan
        _nth_weekday_of_month(year, 2,  0, 3),   # Presidents Day: 3rd Mon Feb
        _nth_weekday_of_month(year, 5,  0, -1),  # Memorial Day: last Mon May
        _nth_weekday_of_month(year, 9,  0, 1),   # Labor Day: 1st Mon Sep
        _nth_weekday_of_month(year, 10, 0, 2),   # Columbus Day: 2nd Mon Oct
        _nth_weekday_of_month(year, 11, 3, 4),   # Thanksgiving: 4th Thu Nov
    ]


def _build_holiday_set(years: List[int]) -> Set[pd.Timestamp]:
    holidays: Set[pd.Timestamp] = set()
    for year in years:
        for month, day in _FIXED_HOLIDAYS:
            holidays.add(pd.Timestamp(year, month, day))
        holidays.update(_variable_holidays(year))
    return holidays


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """End-to-end feature engineering pipeline for flight delay prediction.

    Parameters
    ----------
    target_encode_cols : list[str], optional
        Categorical columns to target-encode. Defaults to
        ``["OP_CARRIER", "ORIGIN", "DEST", "TAIL_NUM"]``.
    smoothing : float
        Laplace smoothing factor for the target encoder (default 20.0).
    """

    FEATURE_COLS: List[str] = [
        "CRS_DEP_TIME_sin",
        "CRS_DEP_TIME_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "Month_sin",
        "Month_cos",
        "departure_hour",
        "route_distance_km",
        "DISTANCE",
        "CRS_ELAPSED_TIME",
        "OP_CARRIER_te",
        "ORIGIN_te",
        "DEST_te",
        "TAIL_NUM_te",
        "route_delay_rate",
        "airline_delay_rate",
        "is_near_holiday",
    ]

    def __init__(
        self,
        target_encode_cols: Optional[List[str]] = None,
        smoothing: float = 20.0,
    ) -> None:
        self.target_encode_cols = target_encode_cols or DEFAULT_ENCODE_COLS
        self.smoothing = smoothing

        self._target_encoder: Optional[KFoldTargetEncoder] = None
        self._route_delay_map: Dict[tuple, float] = {}
        self._airline_delay_map: Dict[str, float] = {}
        self._global_delay_rate: float = 0.5
        self._global_airline_rate: float = 0.5
        self._holiday_set: Set[pd.Timestamp] = set()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private feature builders
    # ------------------------------------------------------------------

    def _compute_route_delay_rate(self, df: pd.DataFrame) -> None:
        """Fit smoothed per-route delay rates from training data."""
        _MIN = 30
        global_rate = float(df["y"].mean())
        self._global_delay_rate = global_rate

        grouped = (
            df.groupby(["ORIGIN", "DEST"])["y"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grouped.columns = ["ORIGIN", "DEST", "n_delayed", "n_total"]
        weight = grouped["n_total"].clip(upper=_MIN) / _MIN
        grouped["rate"] = (
            weight * (grouped["n_delayed"] + 1) / (grouped["n_total"] + 2)
            + (1 - weight) * global_rate
        )
        self._route_delay_map = {
            (r.ORIGIN, r.DEST): r.rate for r in grouped.itertuples(index=False)
        }
        logger.info(
            "Route delay rates computed for %d O-D pairs. Global: %.4f.",
            len(self._route_delay_map), global_rate,
        )

    def _apply_route_delay_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        route_keys = list(zip(df["ORIGIN"], df["DEST"]))
        df["route_delay_rate"] = [
            self._route_delay_map.get(k, self._global_delay_rate)
            for k in route_keys
        ]
        return df

    def _compute_airline_delay_rate(self, df: pd.DataFrame) -> None:
        """Fit smoothed per-airline delay rates from training data."""
        _MIN = 100
        global_rate = float(df["y"].mean())
        self._global_airline_rate = global_rate

        grouped = (
            df.groupby("OP_CARRIER")["y"]
            .agg(["sum", "count"])
            .reset_index()
        )
        grouped.columns = ["OP_CARRIER", "n_delayed", "n_total"]
        weight = grouped["n_total"].clip(upper=_MIN) / _MIN
        grouped["rate"] = (
            weight * (grouped["n_delayed"] + 1) / (grouped["n_total"] + 2)
            + (1 - weight) * global_rate
        )
        self._airline_delay_map = dict(zip(grouped["OP_CARRIER"], grouped["rate"]))
        logger.info(
            "Airline delay rates computed for %d carriers. Global: %.4f.",
            len(self._airline_delay_map), global_rate,
        )

    def _apply_airline_delay_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        df["airline_delay_rate"] = (
            df["OP_CARRIER"].map(self._airline_delay_map).fillna(self._global_airline_rate)
        )
        return df

    def _add_holiday_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary ``is_near_holiday`` = 1 if within 2 days of a US federal holiday."""
        date_col = "FL_DATE" if "FL_DATE" in df.columns else None
        if date_col is None:
            logger.warning("'FL_DATE' not found; is_near_holiday will be 0.")
            df["is_near_holiday"] = 0
            return df

        flight_dates = pd.to_datetime(df[date_col])
        years = sorted(flight_dates.dt.year.unique().tolist())

        if not self._holiday_set:
            all_years = list(range(min(years) - 1, max(years) + 2))
            self._holiday_set = _build_holiday_set(all_years)
            logger.debug(
                "Holiday set built for years %d-%d: %d holidays.",
                min(all_years), max(all_years), len(self._holiday_set),
            )

        # Convert holidays to same resolution as the datetime64 array
        # pandas >= 2.0 uses datetime64[us] (microseconds), not ns
        flight_arr = flight_dates.values
        dt_unit = np.datetime_data(flight_arr.dtype)[0]  # 'us', 'ns', etc.
        holiday_dt64 = np.array(
            [np.datetime64(h.isoformat(), dt_unit) for h in self._holiday_set]
        )
        _CHUNK = 50_000
        _TIMEDELTA_2D = np.timedelta64(2, "D")
        min_days = np.empty(len(df), dtype=float)

        for start in range(0, len(df), _CHUNK):
            end = min(start + _CHUNK, len(df))
            chunk = flight_arr[start:end]
            # Compute absolute day differences (broadcast: flights × holidays)
            diff = np.abs(chunk[:, None] - holiday_dt64[None, :])
            min_diff = diff.min(axis=1)
            # Convert timedelta64 to float days
            min_days[start:end] = min_diff / np.timedelta64(1, "D")

        df["is_near_holiday"] = (min_days <= 2).astype(int)
        logger.info(
            "is_near_holiday: %d / %d rows within 2 days of a US holiday.",
            int(df["is_near_holiday"].sum()), len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, train_df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
        """Fit all encoders on training data and return the transformed frame.

        Parameters
        ----------
        train_df : pd.DataFrame
            Raw training dataframe.
        target_col : str
            Binary delay target column name.

        Returns
        -------
        pd.DataFrame
            Transformed training dataframe with all FEATURE_COLS present.
        """
        logger.info("FeaturePipeline.fit_transform() on %d rows.", len(train_df))
        df = train_df.copy()

        logger.info("Step 1: Temporal cyclic features.")
        df = add_temporal_features(df)

        logger.info("Step 2: Haversine route distance.")
        df = add_route_distance(df)

        logger.info("Step 3: Historical delay rates.")
        self._compute_route_delay_rate(df)
        self._compute_airline_delay_rate(df)
        df = self._apply_route_delay_rate(df)
        df = self._apply_airline_delay_rate(df)

        logger.info("Step 4: K-Fold target encoding.")
        self._target_encoder = KFoldTargetEncoder(
            cols=self.target_encode_cols, smoothing=self.smoothing,
        )
        df = self._target_encoder.fit_transform(df, target_col=target_col)

        logger.info("Step 5: Holiday proximity.")
        df = self._add_holiday_proximity(df)

        self._is_fitted = True
        logger.info("FeaturePipeline.fit_transform() complete. Shape: %s.", df.shape)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transformations to val/test/production data.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe with all FEATURE_COLS present.
        """
        if not self._is_fitted:
            raise RuntimeError("FeaturePipeline.transform() called before fit_transform().")

        logger.info("FeaturePipeline.transform() on %d rows.", len(df))
        df = df.copy()
        df = add_temporal_features(df)
        df = add_route_distance(df)
        df = self._apply_route_delay_rate(df)
        df = self._apply_airline_delay_rate(df)
        df = self._target_encoder.transform(df)
        df = self._add_holiday_proximity(df)
        logger.info("FeaturePipeline.transform() complete. Shape: %s.", df.shape)
        return df

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of final model feature column names."""
        return list(self.FEATURE_COLS)

    def save(self, dir_path: str | Path) -> None:
        """Save all pipeline artifacts to a directory."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dir_path / "pipeline.pkl")
        logger.info("FeaturePipeline saved to '%s/pipeline.pkl'.", dir_path)
        if self._target_encoder is not None:
            self._target_encoder.save(dir_path / "target_encoder.pkl")

    @classmethod
    def load(cls, dir_path: str | Path) -> "FeaturePipeline":
        """Load a saved FeaturePipeline from a directory."""
        pipeline: "FeaturePipeline" = joblib.load(Path(dir_path) / "pipeline.pkl")
        logger.info("FeaturePipeline loaded from '%s'.", dir_path)
        return pipeline
