# Loan Amount Prediction

## Project Overview
This project builds a regression model to predict how much loan a bank or lender would actually sanction for an applicant, not simply whether they're approved, but the specific amount granted based on their financial and personal profile.

## Business Question
Given an applicant's income, credit history, property details, and requested loan amount, how much will the lender actually sanction, and which factors most influence that decision?

## Context
Loan applicants request a specific amount, but lenders don't always grant the full request, the sanctioned amount depends on risk factors like income stability, credit score, existing defaults, and co-applicant status. Modeling this gap between requested and sanctioned amount reflects a real underwriting decision lenders make, and helps quantify which applicant characteristics drive that decision.

## Dataset
Applicant-level data including:
- Demographics: Age, Gender, Profession, Type of Employment, Location
- Financials: Income (USD), Income Stability, Credit Score, No. of Defaults, Has Active Credit Card
- Loan details: Loan Amount Request (USD), Current Loan Expenses (USD), Dependents, Co-Applicant
- Property details: Property Age, Property Type, Property Location, Property Price
- **Target variable:** Loan Sanction Amount (USD)

## Approach
1. **Data Cleaning** : handled missing values (visualized with `missingno`) and inconsistent entries across applicant and property fields.
2. **Feature Engineering** : encoded categorical variables (One-Hot Encoding), scaled numerical features (MinMaxScaler), and built preprocessing pipelines for consistent train/test transformation.
3. **Model Comparison** : trained and compared over a dozen regression algorithms, including Linear, Ridge, Lasso, ElasticNet, Bayesian Ridge, KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Bagging, Extra Trees, Kernel Ridge, SVR, Gaussian Process, and MLP Regressor.
4. **Hyperparameter Tuning** : used `RandomizedSearchCV` to tune the best-performing model.
5. **Evaluation** : assessed final performance using RMSE on the test set, with a 95% confidence interval around prediction error.
6. **Feature Importance** : analyzed which applicant characteristics most influenced predicted loan amounts.

## Tools & Technologies
- **Python** (Pandas, NumPy)
- **Scikit-learn** : regression models, pipelines, `RandomizedSearchCV`, evaluation metrics
- **SHAP & Yellowbrick** :  model interpretation and feature importance
- **Matplotlib & Seaborn** : visualization
- **Missingno** : missing data visualization

## Key Findings
- The final tuned model achieved a test RMSE of approximately **$20,936**, meaning predictions typically fall within about $21K of the actual sanctioned amount.
- At 95% confidence, the model's prediction error falls between **$19,752 and $22,058**.
- Random Forest Regressor was the model that performed best and yielded an improved score after hyperparameter tuning.
- Loan amount requests, credit score, and co-applicant are top 3 fearures driving predictions from feature importance analysis

## Limitations & Next Steps
- RMSE in the ~$20K range indicates room for improvement, further feature engineering or additional data (e.g. employment history length, existing debt-to-income ratio) could tighten predictions.
- Future iterations could explore ensemble stacking of top-performing models or gradient boosting frameworks (XGBoost, LightGBM) not covered in this comparison.
