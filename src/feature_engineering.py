"""
Feature Engineering: RFM Segmentation, Recency, Dynamic Behavioral Trends, and Seasonality.
"""
import pandas as pd
import numpy as np
from typing import List


def compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) quintile/tertile scores.
    """
    df = df.copy()

    # Invert recency ranking: lower recency = higher engagement
    df["R_score"] = pd.qcut(df["Recency_Months"].rank(method="first"), q=3, labels=[3, 2, 1]).astype(int)
    df["F_score"] = pd.qcut(df["Total_Bookings"].rank(method="first"), q=3, labels=[1, 2, 3]).astype(int)
    df["M_score"] = pd.qcut(df["CLV"].rank(method="first"), q=3, labels=[1, 2, 3]).astype(int)

    return df


def compute_trend_metric(recent_series: pd.Series, prior_series: pd.Series, epsilon: float = 1e-5) -> pd.Series:
    """
    Compute percentage trend between recent activity and prior activity to capture momentum:
    Trend = (Recent - Prior) / (Prior + epsilon)
    """
    return (recent_series - prior_series) / (prior_series + epsilon)


def clean_infinite_and_nulls(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Safely clip extreme ratios and replace infinite/NaN values generated during trend calculation.
    """
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
    return df
