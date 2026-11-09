"""
Configuration parameters, file paths, and feature schemas.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw Data Filenames
RAW_FILES = {
    "loyalty": RAW_DATA_DIR / "Customer Loyalty History.csv",
    "activity": RAW_DATA_DIR / "Customer Flight Activity.csv",
    "calendar": RAW_DATA_DIR / "Calendar.csv",
    "dictionary": RAW_DATA_DIR / "Airline Loyalty Data Dictionary.csv"
}

# Processed Data Filenames
PROCESSED_FILES = {
    "main": PROCESSED_DATA_DIR / "data_main.csv",
    "retention": PROCESSED_DATA_DIR / "retention_dashboard.csv",
    "segment_behavior": PROCESSED_DATA_DIR / "segment_behavior.csv",
    "relative_profile": PROCESSED_DATA_DIR / "relative_segment_profile.csv",
    "feature_importance": MODELS_DIR / "feature_importance.csv",
    "model": MODELS_DIR / "xgb_model.pkl"
}

# Machine Learning & Segmentation Features
MODEL_FEATURES = [
    "CLV",
    "Total_Bookings",
    "Average_distance",
    "Recency_Months",
    "Booking_Trend",
    "Accumulation_Trend",
    "Redemption_Trend",
    "Seasonality_Index"
]

SEGMENT_FEATURES = [
    "CLV",
    "Total_Bookings",
    "Average_distance",
    "Recency_Months",
    "Booking_Trend",
    "Accumulation_Trend",
    "Redemption_Trend",
    "Seasonality_Index"
]

TARGET_COLUMN = "Hard_Churn"

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss"
}

# Risk Thresholds
HIGH_RISK_THRESHOLD = 0.80
MEDIUM_RISK_THRESHOLD = 0.40

# Segment Mapping
SEGMENT_LABELS = {
    0: "Active Regular Customers",
    1: "Champions / Premium Travelers",
    2: "Dormant Customers",
    3: "Occasional Flyers"
}
