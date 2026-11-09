# UNLOCKING BEHAVIORAL INTELLIGENCE IN AIRLINE LOYALTY PROGRAMS

## Customer Churn Prediction, Segmentation and Smart Retention

**Prepared By:** Tanishq Gupta

---

# 1. Executive Summary

Customer retention is one of the most important drivers of profitability in the airline industry. Acquiring new customers is significantly more expensive than retaining existing ones, making early identification of disengaged customers a strategic priority.

The objective of this project was to develop a framework capable of:

1. Predicting customer churn.
2. Identifying meaningful customer segments.
3. Recommending targeted retention strategies.

The analysis combined customer demographics, loyalty information and monthly activity records to create customer-level behavioral profiles. Multiple machine learning models were evaluated, and XGBoost achieved the strongest performance with an ROC-AUC score of 0.963.

Four customer segments were identified:

- Champions / Premium Travelers
- Active Regular Customers
- Occasional Flyers
- Dormant Customers

The analysis demonstrated that customer behavior is a much stronger indicator of future churn than demographic characteristics or even Customer Lifetime Value (CLV).

---

# 2. Problem Formulation

Customer retention is one of the most important drivers of profitability in the airline industry. Acquiring new customers is significantly more expensive than retaining existing ones, making proactive churn management essential.

The project addressed three business questions:

## Churn Prediction

Which customers are likely to stop engaging with the airline?

## Customer Segmentation

Are there naturally occurring groups of customers with different behavioral patterns?

## Smart Retention

How can marketing resources be allocated effectively to reduce customer attrition?

The challenge was to create a solution that translates analytical outputs into actionable business decisions.

---

# 3. Data Description and Preparation

The analysis was based on four integrated datasets:

## Customer Loyalty History

Provided demographic information, loyalty tier, customer lifetime value and cancellation records.

## Customer Flight Activity

Contained monthly records of flights booked, distance travelled and points accumulated or redeemed.

## Calendar Dataset

Mapped months to seasons and quarters.

## Data Dictionary

Provided definitions and descriptions of variables.

---

## Data Cleaning

Several preprocessing decisions were required.

### Missing Values

Missing values representing absence of activity were replaced with zero.

### Infinite Values

Ratio-based features occasionally generated infinite values. These were removed or replaced appropriately.

### Aggregation

Monthly activity records were aggregated to customer level to obtain a single profile per customer.

### Avoiding Data Leakage

Only information that would have been available at the time of prediction was used when constructing features.

This ensured that model performance reflected realistic business scenarios.

---

# 4. Feature Engineering

Rather than relying only on demographics, the analysis emphasized customer behavior.

## Customer Value Variables

- Customer Lifetime Value (CLV)
- Total Bookings
- Average Distance Travelled
- Total Points Accumulated
- Total Points Redeemed

## Recency

Recency was defined as the number of months between the customer's last flight and the end of 2018.

Customers with larger recency values were considered less engaged.

## Trend Variables

Behavioral changes were captured through:

### Booking Trend

Measures whether recent booking activity has increased or decreased relative to historical activity.

### Accumulation Trend

Measures changes in point accumulation behavior.

### Redemption Trend

Measures changes in point redemption behavior.

Trend variables provide insight into customer momentum rather than relying only on absolute values.

## Seasonality Index

Seasonal booking patterns were analyzed across:

- Spring
- Summer
- Autumn
- Winter

This variable helps distinguish consistent travelers from customers whose activity is concentrated in particular seasons.

---

# 5. Exploratory Analysis and Customer Value

Correlation analysis revealed that demographic variables such as salary exhibited very weak relationships with churn and customer lifetime value.

Behavioral variables showed considerably stronger associations with customer engagement.

## Pareto Analysis

Customers were ranked by Customer Lifetime Value to evaluate the concentration of value across the customer base.

The analysis confirmed the Pareto principle, showing that a relatively small group of customers contributes a disproportionately large share of total value.

## RFM Analysis

Customers were evaluated using:

- Recency
- Frequency
- Monetary Value

RFM analysis helped identify VIP customers and provided additional insight into customer value distribution.

---

# 6. Customer Segmentation

K-Means clustering was used to group customers according to behavioral characteristics.

## Selection of Number of Clusters

The elbow method and silhouette scores were used to evaluate different values of K.

A four-cluster solution was selected because it provided meaningful and interpretable customer groups while avoiding excessive complexity.

## Segment Profiles

### Champions / Premium Travelers

- Highest booking frequency.
- Long travel distances.
- Strong positive booking trends.
- High seasonality.

### Active Regular Customers

- Stable engagement.
- Consistent travel behavior.
- Moderate activity levels.

### Occasional Flyers

- Low booking frequency.
- Short travel distances.
- Moderate engagement.

### Dormant Customers

- High recency values.
- Declining activity.
- Negative behavioral trends.

## Demographic Insights

Cross-tabulation analysis revealed minimal differences across:

- Gender
- Education
- Marital Status
- Loyalty Card Type

This suggests that behavior, rather than demographics, drives customer segmentation.

---

# 7. Churn Prediction

Three machine learning models were evaluated.

## Logistic Regression

Accuracy: 91.6%

## Random Forest

Accuracy: 92.8%

## XGBoost

Accuracy: 93.1%

Recall: 81.6%

F1 Score: 0.746

ROC-AUC Score: 0.963

XGBoost achieved the strongest performance and was selected as the final model.

---

# 8. Key Drivers of Churn

Feature importance analysis showed that customer behavior dominates demographic information.

| Feature | Relative Importance |
|-----------|----------------|
| Total Bookings | 44.9% |
| Booking Trend | 22.4% |
| Recency Months | 20.3% |
| Accumulation Trend | 5.0% |
| Average Distance | 2.8% |
| Seasonality Index | 1.8% |
| Redemption Trend | 1.4% |
| CLV | 1.3% |

## Interesting Findings

Customer Lifetime Value had surprisingly low predictive importance.

This suggests that declining behavior provides stronger signals of future churn than historical customer value.

Demographic variables contributed very little to customer differentiation.

---

# 9. Smart Retention Framework

## Champions / Premium Travelers

- Priority services.
- Exclusive promotions.
- Tier upgrades.
- Premium reward programs.

## Active Regular Customers

- Maintain engagement.
- Personalized travel offers.
- Seasonal campaigns.

## Occasional Flyers

- Bonus miles.
- Promotional discounts.
- Targeted destination recommendations.

## Dormant Customers

- Reactivation campaigns.
- Personalized emails.
- Limited-time incentives.
- Win-back promotions.

---

# 10. Prototype Dashboard

A Streamlit dashboard was developed to provide a user-friendly interface for non-technical users.

The dashboard includes:

- Customer segmentation.
- Churn prediction.
- Smart retention recommendations.
- Feature importance analysis.
- Business insights.

---

# 11. Business Recommendations

## Recommendation 1: Prioritize Behavioral Monitoring

Customer behavior is a stronger predictor of churn than demographic variables.

Organizations should continuously monitor booking trends and recency indicators.

## Recommendation 2: Protect High-Value Customers

Champions and Premium Travelers generate disproportionate value.

Retention efforts should focus on preserving these relationships through exclusive rewards and personalized experiences.

## Recommendation 3: Intervene Early

Customers exhibiting declining booking trends should receive targeted incentives before complete disengagement occurs.

Proactive intervention is significantly more cost-effective than reacquiring lost customers.

---

# Conclusion

This project demonstrates that customer behavior contains valuable signals that can be leveraged to predict churn and understand customer value.

By combining segmentation with machine learning, the airline can move from reactive customer management to proactive retention strategies.

The resulting framework provides both predictive capability and actionable business insights, enabling the organization to strengthen customer relationships and maximize the long-term value of its loyalty program.