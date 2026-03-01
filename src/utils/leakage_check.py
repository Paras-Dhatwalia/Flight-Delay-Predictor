"""
leakage_check.py
----------------
Feature leakage detection utilities for the flight delay model.

Run after feature engineering to confirm that no blacklisted columns
survived, and that no single feature has suspiciously high correlation
with the binary target (which would indicate proxied leakage).
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

BLACKLISTED_COLUMNS: List[str] = [
    "TaxiOut",
    "TaxiIn",
    "WheelsOff",
    "WheelsOn",
    "ActualElapsedTime",
    "AirTime",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    "DiversionDistance",
    "ActualDeparture",
    "ActualArrival",
]


def check_no_blacklisted_features(df: pd.DataFrame) -> bool:
    """Assert that no blacklisted columns are present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe to audit.

    Returns
    -------
    bool
        ``True`` if no blacklisted columns are found.

    Raises
    ------
    ValueError
        If any blacklisted column is found.
    """
    found = [c for c in BLACKLISTED_COLUMNS if c in df.columns]
    if found:
        raise ValueError(
            f"Leakage detected! Found {len(found)} blacklisted column(s) in the "
            f"dataframe: {found}. Remove these before training."
        )
    logger.info("Blacklist check passed — no blacklisted columns found.")
    return True


def check_feature_target_correlation(
    df: pd.DataFrame,
    target_col: str = "y",
    threshold: float = 0.95,
) -> List[str]:
    """Identify features with suspiciously high correlation to the target.

    A Pearson correlation > ``threshold`` with the binary target almost
    certainly indicates direct or proxied data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing both features and the target.
    target_col : str
        Binary target column name (default ``"y"``).
    threshold : float
        Absolute correlation above which a feature is flagged (default 0.95).

    Returns
    -------
    list[str]
        Names of features with ``|corr| > threshold``.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    numeric_df = df.select_dtypes(include="number")
    suspicious: List[str] = []

    for col in numeric_df.columns:
        if col == target_col:
            continue
        corr = float(numeric_df[col].corr(numeric_df[target_col]))
        if abs(corr) > threshold:
            logger.warning(
                "POTENTIAL LEAKAGE: feature '%s' has |correlation| = %.4f > %.2f "
                "with target '%s'. Investigate immediately.",
                col, abs(corr), threshold, target_col,
            )
            suspicious.append(col)

    if not suspicious:
        logger.info(
            "Correlation check passed — no feature exceeds |corr| = %.2f with target.",
            threshold,
        )
    return suspicious


def run_leakage_audit(
    df: pd.DataFrame,
    target_col: str = "y",
    corr_threshold: float = 0.95,
) -> dict:
    """Run all leakage checks and return a summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe (post feature-engineering, pre-model training).
    target_col : str
        Binary target column name.
    corr_threshold : float
        Correlation threshold for flagging features.

    Returns
    -------
    dict
        ``{"blacklist_ok": bool, "high_corr_features": list, "all_passed": bool}``
    """
    print("\n" + "=" * 60)
    print("  LEAKAGE AUDIT REPORT")
    print("=" * 60)

    # --- Blacklist check ---
    blacklist_ok = True
    try:
        check_no_blacklisted_features(df)
        print("[PASS] Blacklist check: no leakage columns found.")
    except ValueError as exc:
        blacklist_ok = False
        print(f"[FAIL] Blacklist check: {exc}")

    # --- Correlation check ---
    high_corr: List[str] = []
    try:
        high_corr = check_feature_target_correlation(df, target_col, corr_threshold)
        if high_corr:
            print(f"[WARN] High-correlation features (|r| > {corr_threshold}): {high_corr}")
        else:
            print(f"[PASS] Correlation check: no feature exceeds |r| = {corr_threshold}.")
    except Exception as exc:
        print(f"[ERROR] Correlation check failed: {exc}")

    all_passed = blacklist_ok and len(high_corr) == 0

    print("-" * 60)
    print(f"  Overall: {'ALL CHECKS PASSED' if all_passed else 'ISSUES DETECTED'}")
    print("=" * 60 + "\n")

    return {
        "blacklist_ok": blacklist_ok,
        "high_corr_features": high_corr,
        "all_passed": all_passed,
    }
