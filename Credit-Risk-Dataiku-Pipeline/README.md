# 🏦 Credit Risk Analysis & Prediction Pipeline

## 📌 Project Overview
An end-to-end Machine Learning pipeline built to automate credit risk assessment and loan default prediction. This project demonstrates a complete data science lifecycle—from raw data ingestion and custom feature engineering to model deployment (Batch Scoring) and business intelligence dashboarding. 

The pipeline is designed with a strong emphasis on **Explainable AI (XAI)** and business logic, ensuring that model predictions can be clearly interpreted by risk management teams.

## 🛠️ Tech Stack & Tools
* **Languages:** Python (Pandas, NumPy)
* **Platform:** Dataiku DSS (Data Science Studio)
* **Machine Learning:** Scikit-Learn (via Dataiku AutoML)
* **Visualization:** Dataiku Dashboards

## 🏗️ Pipeline Architecture
The project follows a structured Flow in Dataiku DSS:
1. **Data Ingestion:** Imported raw financial datasets (historical loan data & applicant profiles).
2. **Feature Engineering (Python):** Developed custom Python recipes to generate financially sound metrics (e.g., Financial Stress Index, Disposable Income).
3. **AutoML Classification:** Trained interpretable models (e.g., Random Forest) to predict loan default probability (`default_flag`).
4. **Production / Batch Scoring:** Deployed the winning model to score a continuous stream of new, unseen loan applications.
5. **Business Dashboard:** Created interactive visualizations for risk tier distributions and portfolio exposure.

![Dataiku Flow Screenshot](CREDIT_RISK_ANALYSIS-flow.pdf)  
*Caption: The Dataiku DSS Flow showing the end-to-end pipeline.*

## 🧠 Business-Driven Feature Engineering
Instead of relying solely on raw data, I engineered custom features reflecting real-world underwriting methodologies using Python Pandas:
* **`financial_stress_index`:** A weighted composite score evaluating previous defaults, recent delinquencies, and loan-to-income ratios.
* **`monthly_disposable_income`:** Calculated the real disposable income by subtracting the estimated monthly loan payment (using standard PMT formulas) from the monthly salary.
* **`credit_score_tier`:** Grouped raw credit scores into standardized industry tiers (Excellent, Good, Fair, Poor) to reduce noise.

## 📊 Model Evaluation & Business Impact
In the financial domain, a "black-box" model is not viable. I prioritized interpretable models and evaluated performance based on business impact rather than pure accuracy:
* Optimized for **ROC AUC** to handle imbalanced default classes.
* Analyzed the **Confusion Matrix** to balance the cost of *False Negatives* (loss of loan principal) vs. *False Positives* (loss of potential interest revenue).
* **Feature Importance:** Validated that the custom-engineered `financial_stress_index` was one of the strongest predictors of default.

## 📈 Executive Dashboard
The final output is a business-facing dashboard designed for credit risk managers, featuring:
* **Risk Tier Distribution:** Quick visibility into pipeline approval rates.
* **Disposable Income vs. Risk:** Validating the correlation between calculated disposable income and default probability.
* **Portfolio Exposure:** Average loan amounts requested by different credit tiers.

![Dashboard Screenshot](Credit_Risk_Analysis_s-default-dashboard.pdf)  
*Caption: Real-time risk assessment dashboard for new loan applications.*

## 🚀 How to Run / Reproduce
1. Clone this repository to view the Python scripts used for Feature Engineering (`feature_engineering.py`) and new data generation (`generate_new_customers.py`).
2. The datasets can be imported into any local or cloud instance of Dataiku DSS.
3. Apply the Python recipes in the Dataiku Flow and run the AutoML Lab to reproduce the model.

---
*Developed as a demonstration of translating complex data into actionable financial impact.*
