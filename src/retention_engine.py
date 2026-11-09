"""
Prescriptive Retention Engine: Risk Tiering & Personalized Intervention Rules.
"""
import pandas as pd
import numpy as np
from src.config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD


def assign_risk_level(prob: float) -> str:
    """
    Categorize churn probability into business risk tiers.
    """
    if prob >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


def recommend_action(row: pd.Series) -> str:
    """
    Prescribe segment-specific business interventions based on risk level and behavioral persona.
    """
    risk = row["Risk_Level"]
    segment = row["Customer_Segment"]

    if risk == "High Risk":
        if segment == "Dormant Customers":
            return "30% bonus miles + reactivation email"
        elif segment == "Active Regular Customers":
            return "Personalized route discount"
        elif segment == "Occasional Flyers":
            return "Double miles on next booking"
        elif segment == "Champions / Premium Travelers":
            return "Dedicated concierge outreach + VIP upgrade"
        else:
            return "Targeted reactivation campaign"

    elif risk == "Medium Risk":
        if segment == "Dormant Customers":
            return "Seasonal discount campaign"
        elif segment == "Active Regular Customers":
            return "Loyalty milestone bonus"
        elif segment == "Occasional Flyers":
            return "Partner hotel/car rental coupon"
        elif segment == "Champions / Premium Travelers":
            return "Lounge pass renewal offer"
        else:
            return "Engagement newsletter"

    else:
        return "No immediate action"


def build_retention_dataset(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    customer_segments: pd.Series
) -> pd.DataFrame:
    """
    Compile customer metrics, churn probabilities, risk levels, and prescriptive retention actions.
    """
    dashboard = X_test.copy()
    dashboard["Actual_Churn"] = y_test.values
    dashboard["Churn_Probability"] = y_prob
    dashboard["Risk_Level"] = dashboard["Churn_Probability"].apply(assign_risk_level)
    dashboard["Customer_Segment"] = customer_segments.loc[X_test.index].values
    dashboard["Recommended_Action"] = dashboard.apply(recommend_action, axis=1)

    return dashboard
