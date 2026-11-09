# Resume Bullet Points & Interview Talking Points Guide
**Project:** Airline Loyalty Behavioral Intelligence & Smart Churn Retention  
**Author:** Tanishq Gupta  
**Target Roles:** Data Scientist, Machine Learning Engineer, Customer/Growth Analytics

---

## 1. Resume Bullet Points (Ready to Copy-Paste)

### Option A: Standard Data Scientist Role (Balanced Impact, ML & Business)
> - **Engineered an end-to-end customer churn prediction and retention platform** analyzing **400K+ flight activity records** across 16.7K loyalty members, driving proactive customer engagement.
> - **Formulated dynamic behavioral momentum features** (Booking, Accumulation, and Redemption Trends, Recency, Seasonality Index), uncovering that behavioral dynamics drive **87.6%** of churn variance over static demographics.
> - **Developed and tuned an XGBoost classifier** achieving **0.963 ROC-AUC**, **93.1% accuracy**, and **81.6% recall**, outperforming Random Forest and Logistic Regression baselines.
> - **Segmented customer base into 4 behavioral personas** via **K-Means clustering and RFM analysis** (evaluated using Elbow and Silhouette methods), identifying that the top customer tier generated disproportionate program value.
> - **Architected a prescriptive smart retention engine & Streamlit dashboard** mapping churn probabilities to automated retention interventions (e.g., targeted bonus miles, concierge outreach), enabling marketing teams to target high-risk accounts.

---

### Option B: Machine Learning Engineer Role (Pipeline, Architecture & Deployment Focus)
> - **Architected a production-grade ML pipeline** in Python for airline churn prediction, incorporating modular data preprocessing, feature engineering, and automated evaluation modules.
> - **Trained and validated XGBoost, Random Forest, and regularized Logistic Regression** models on 16.7K+ user profiles, boosting predictive recall to **81.6%** with an **ROC-AUC of 0.963**.
> - **Engineered a scalable feature transformation pipeline** handling time-series flight aggregations, ratio-based trend derivatives, and outlier capping without data leakage.
> - **Deployed an interactive full-stack Streamlit dashboard** featuring real-time churn risk scoring, cohort drill-downs, segment heatmaps, and downloadable retention action lists.

---

### Option C: Customer / Product / Growth Analytics Role (Business Impact & ROI Focus)
> - **Identified critical attrition risk factors** across 16,700+ airline loyalty members, showing that booking trend decline and flight recency accounted for over **67%** of customer churn propensity.
> - **Implemented an RFM & K-Means customer segmentation framework** categorizing members into 4 actionable cohorts (Champions, Active Regulars, Occasional, Dormant) with normalized behavioral heatmaps.
> - **Designed a prescriptive decision matrix** linking individual churn risk scores to personalized retention campaigns, projected to reduce high-value member attrition.
> - **Delivered an executive technical report and interactive dashboard** empowering commercial and marketing teams to prioritize retention spend efficiently.

---

## 2. One-Paragraph Project Overview (For Portfolio, LinkedIn, or GitHub About)

> **Airline Loyalty Behavioral Analytics & Prescriptive Retention Platform**  
> Built an end-to-end predictive and prescriptive analytics system on 400K+ flight records to mitigate loyalty member attrition. Engineered momentum trend features and RFM scores that demonstrated behavioral dynamics dominate static demographics in predicting churn. Built and evaluated multiple models, selecting an XGBoost classifier that achieved an ROC-AUC of 0.963 and 81.6% recall. Clustered members into 4 distinct behavioral personas using K-Means and developed an automated decision engine and interactive Streamlit web dashboard prescribing targeted retention actions for commercial marketing teams.

---

## 3. Keywords & Skills for Resume ATS (Applicant Tracking Systems)

- **Languages & Frameworks:** Python, SQL, Pandas, NumPy, Scikit-Learn, XGBoost, Streamlit, Plotly, Seaborn, Matplotlib.
- **Machine Learning & Modeling:** Supervised Classification, XGBoost, Random Forest, Logistic Regression, Hyperparameter Tuning, Cross-Validation, ROC-AUC, Precision-Recall, Class Imbalance Handling.
- **Unsupervised Learning & Analytics:** K-Means Clustering, Elbow Method, Silhouette Analysis, RFM Analysis (Recency, Frequency, Monetary), Pareto Analysis, Feature Importance, Model Explainability.
- **Data Engineering & Workflow:** Modular OOP Architecture, Time-Series Feature Engineering, Data Aggregation, Pipeline Design, Git.
- **Business & Domain:** Customer Churn Prevention, Loyalty Programs, Prescriptive Retention Strategies, Customer Lifetime Value (CLV), Growth & Product Analytics.

---

## 4. Key Interview Questions & Winning Answers

### Q1: "Walk me through this Airline Loyalty project."
**Answer Framework:**
1. **Business Context:** Customer acquisition in aviation is 5–7x more costly than retention. The goal was to detect churn signals early and prescribe targeted retention actions.
2. **Data & Feature Engineering:** Integrated 4 datasets (16.7K members, 400K monthly flight records). Created momentum trend features (recent vs. past booking ratios, points accumulation/redemption trends, seasonality).
3. **Segmentation:** Grouped users into 4 behavioral archetypes (Champions, Active Regulars, Occasional, Dormant) using K-Means and normalized relative profiling.
4. **Modeling & Results:** Trained Logistic Regression, Random Forest, and XGBoost. XGBoost achieved top performance with an ROC-AUC of 0.963 and 81.6% recall.
5. **Business Execution:** Built a rule-based decision matrix matching segment + churn probability to retention actions, served via an interactive Streamlit dashboard.

---

### Q2: "Why did customer behavior outperform demographics and CLV in predicting churn?"
**Answer:**  
In our feature importance analysis, behavioral momentum variables accounted for over **87%** of the model's split decisions (Total Bookings: 45.0%, Booking Trend: 22.4%, Recency: 20.3%), whereas demographic features and static historical CLV contributed under 3%. This makes intuitive business sense: an affluent executive with high historical CLV who suddenly stops booking or reduces flight frequency represents an active churn risk, whereas demographics only tell you *who* the customer is, not *how their relationship with the brand is changing*.

---

### Q3: "How did you prevent data leakage during feature engineering?"
**Answer:**  
We established a strict temporal boundary:
- Behavioral features (Recency, Booking Trends, Seasonal flight activity, Point redemptions) were computed solely from activity logged **prior** to the target observation window.
- All transformations (standard scaling, imputation) were fitted strictly on the training set and applied downstream to the test set.
- Target definitions (`Hard_Churn` / cancellation status) were derived independently and excluded from the feature matrix.

---

### Q4: "How did you decide on 4 clusters for K-Means?"
**Answer:**  
We combined analytical rigor with business interpretability:
- **Elbow Method:** Plotted within-cluster sum of squares (inertia) across $k \in [2, 10]$ and observed the primary bend at $k=4$.
- **Silhouette Analysis:** Verified cluster separation and minimized negative silhouette coefficients.
- **Actionability:** $k=4$ yielded distinct, operationally viable personas:
  - *Champions / Premium Travelers* (High frequency, long distance, high seasonality)
  - *Active Regular Customers* (Stable, frequent, moderate distance)
  - *Occasional Flyers* (Low frequency, short distance, price-sensitive)
  - *Dormant Customers* (High recency >15 months, negative booking momentum)

---

### Q5: "If deployed to production, how would you measure business impact?"
**Answer:**  
I would run a randomized control trial (A/B Test) on customers flagged as High Risk ($\ge 80\%$ churn probability):
- **Treatment Group:** Receives prescribed retention interventions (e.g., targeted bonus miles, concierge outreach, route discounts).
- **Control Group:** Receives standard BAU (Business-As-Usual) marketing communications.
- **Primary Metric:** Churn rate reduction at 60 and 90 days.
- **Secondary Metric / Guardrail:** Incremental Net Revenue (CLV preserved minus retention campaign cost).
