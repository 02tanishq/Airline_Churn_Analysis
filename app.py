import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Airline Loyalty Analytics",
    page_icon="✈",
    layout="wide"
)

# ---------------------------------------------------------
# Path Configuration & Data Loading
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def load_data():
    # Robust path fallbacks
    def resolve_path(primary: Path, fallback: Path) -> Path:
        return primary if primary.exists() else fallback

    data_main_path = resolve_path(
        DATA_PROCESSED / "data_main.csv",
        BASE_DIR / "data_main.csv"
    )
    retention_path = resolve_path(
        DATA_PROCESSED / "retention_dashboard.csv",
        BASE_DIR / "retention_dashboard.csv"
    )
    feat_imp_path = resolve_path(
        MODELS_DIR / "feature_importance.csv",
        DATA_PROCESSED / "feature_importance.csv"
    )
    relative_prof_path = resolve_path(
        DATA_PROCESSED / "relative_segment_profile.csv",
        BASE_DIR / "relative_segment_profile.csv"
    )

    data_main = pd.read_csv(data_main_path)
    retention_df = pd.read_csv(retention_path)
    feature_importance = pd.read_csv(feat_imp_path)
    relative_profile = pd.read_csv(relative_prof_path, index_col=0)

    return data_main, retention_df, feature_importance, relative_profile


data_main, retention_df, feature_importance, relative_profile = load_data()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("✈ Airline Loyalty Analytics")
st.sidebar.caption("Customer Behavioral Intelligence & Churn Retention")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Churn Prediction",
        "Smart Retention",
        "Business Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Highlights:**
    - 🎯 **Best Model:** XGBoost (ROC-AUC: 0.963)
    - 👥 **Segments:** 4 Behavioral Clusters
    - 💡 **Action Engine:** Prescriptive Interventions
    """
)

# =========================================================
# HOME
# =========================================================
if page == "Home":

    st.title("✈ Airline Loyalty Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{len(data_main):,}"
    )

    col2.metric(
        "Champions",
        f"{len(data_main[data_main['Customer_Segment'] == 'Champions / Premium Travelers']):,}"
    )

    col3.metric(
        "Dormant Customers",
        f"{len(data_main[data_main['Customer_Segment'] == 'Dormant Customers']):,}"
    )

    col4.metric(
        "Churn Rate %",
        f"{round(data_main['Hard_Churn'].mean() * 100, 2)}%"
    )

    st.markdown("---")

    st.subheader("Customer Segment Distribution")

    fig = px.pie(
        data_main,
        names="Customer_Segment",
        title="Customer Segments Proportion",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================
elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    st.subheader("Segment Counts")

    fig = px.histogram(
        data_main,
        x="Customer_Segment",
        color="Customer_Segment",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(xaxis_title="Customer Segment", yaxis_title="Customer Count")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Normalized Segment Profile Heatmap")

    fig2, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        relative_profile,
        annot=True,
        cmap='RdYlBu_r',
        center=1,
        fmt='.2f',
        linewidths=0.5,
        ax=ax
    )

    plt.title("Relative Segment Profiles (1.0 = Average Customer Baseline)", fontsize=13, pad=12)
    plt.xticks(rotation=20, ha="right")

    st.pyplot(fig2)

# =========================================================
# CHURN PREDICTION
# =========================================================
elif page == "Churn Prediction":

    st.title("Customer Churn Risk")

    customer = st.selectbox(
        "Select Customer Index",
        retention_df.index
    )

    row = retention_df.loc[customer]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{row['Churn_Probability']*100:.1f}%"
    )

    col2.metric(
        "Risk Level",
        row["Risk_Level"]
    )

    col3.metric(
        "Segment",
        row["Customer_Segment"]
    )

    st.success(
        f"Recommended Action: {row['Recommended_Action']}"
    )

    # Detailed Customer Profile expander
    with st.expander("View Full Customer Metrics"):
        st.write(pd.DataFrame([row]).drop(columns=["Recommended_Action"], errors="ignore"))

# =========================================================
# SMART RETENTION
# =========================================================
elif page == "Smart Retention":

    st.title("Smart Retention Dashboard")

    risk_filter = st.multiselect(
        "Filter Risk Level",
        retention_df["Risk_Level"].unique(),
        default=retention_df["Risk_Level"].unique()
    )

    temp = retention_df[
        retention_df["Risk_Level"].isin(risk_filter)
    ]

    st.dataframe(
        temp.sort_values(
            by="Churn_Probability",
            ascending=False
        ),
        use_container_width=True
    )

    st.download_button(
        label="Download Retention Actions CSV",
        data=temp.to_csv(index=False),
        file_name="retention_dashboard.csv",
        mime="text/csv"
    )

# =========================================================
# BUSINESS INSIGHTS
# =========================================================
elif page == "Business Insights":

    st.title("Business Insights")

    st.subheader("Feature Importance (XGBoost)")

    fig3 = px.bar(
        feature_importance.sort_values(by="Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.subheader("Key Findings & Strategic Recommendations")

    st.info("""
### Key Findings

- **XGBoost achieved the highest performance** with **ROC-AUC = 0.963** (vs Random Forest and Logistic Regression).
- **Behavior Trumps Demographics:** Total Bookings (45.0%), Booking Trend (22.4%), and Recency Months (20.3%) explain nearly 88% of churn variance.
- **Demographics & Historical CLV** provide minimal forward-looking churn signal compared to dynamic behavioral momentum.
- **Four Distinct Behavioral Segments Identified:**
    - **Champions / Premium Travelers:** High travel frequency, highest seasonality and positive booking trend.
    - **Active Regular Customers:** Steady flight activity and balanced accumulation/redemption.
    - **Occasional Flyers:** Low frequency, short distances, responsive to bonus incentives.
    - **Dormant Customers:** High recency (>15 months), sharp negative trend, severe attrition risk.

---

### Prescriptive Strategic Recommendations

✅ **Champions / Premium Travelers:** Exclusive VIP perks, proactive concierge, priority tier renewal incentives.

✅ **Active Regular Customers:** Personalized route promotions, loyalty multiplier events to sustain consistency.

✅ **Occasional Flyers:** Targeted seasonal discounts, low-barrier bonus mile promotions to build travel habits.

✅ **Dormant Customers:** Automated win-back campaigns with 30% bonus redemption value before total attrition.
""")
