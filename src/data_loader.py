"""
Data loading, cleaning, and aggregation pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from src.config import RAW_FILES, PROCESSED_FILES


def load_raw_datasets(custom_paths: Optional[Dict[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw loyalty history, flight activity, and calendar datasets.
    """
    paths = custom_paths or RAW_FILES

    loyalty_df = pd.read_csv(paths["loyalty"])
    activity_df = pd.read_csv(paths["activity"])
    calendar_df = pd.read_csv(paths["calendar"])

    return loyalty_df, activity_df, calendar_df


def aggregate_customer_activity(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly flight activity records into customer-level behavioral summaries.
    """
    # Clean nulls
    activity = activity_df.copy()
    activity.fillna({
        "Total Flights": 0,
        "Distance": 0,
        "Points Accumulated": 0,
        "Points Redeemed": 0,
        "Dollar Cost Points Redeemed": 0
    }, inplace=True)

    # Customer aggregates
    agg_df = activity.groupby("Loyalty Number").agg(
        Total_Bookings=("Total Flights", "sum"),
        Total_Distance=("Distance", "sum"),
        Average_distance=("Distance", "mean"),
        Total_Points_Accumulated=("Points Accumulated", "sum"),
        Total_Points_Redeemed=("Points Redeemed", "sum"),
        Active_Months=("Total Flights", lambda x: (x > 0).sum())
    ).reset_index()

    return agg_df


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load ready-to-use processed datasets.
    """
    data_main = pd.read_csv(PROCESSED_FILES["main"])
    retention_df = pd.read_csv(PROCESSED_FILES["retention"])
    feature_importance = pd.read_csv(PROCESSED_FILES["feature_importance"])
    relative_profile = pd.read_csv(PROCESSED_FILES["relative_profile"], index_col=0)

    return data_main, retention_df, feature_importance, relative_profile
