Fraud Detection for E-Commerce and Credit Card Transactions

Project Overview

This project develops machine learning models to detect fraudulent transactions using two datasets:

Fraud_Data.csv – E-commerce transaction fraud data

creditcard.csv – Credit card transaction fraud data

The project covers the complete machine learning lifecycle including data preprocessing, exploratory data analysis, feature engineering, class imbalance handling, model training, model evaluation, and model explainability using SHAP.

Business Problem

Fraudulent transactions cause significant financial losses and negatively impact customer trust. The objective of this project is to build predictive models capable of identifying fraudulent transactions accurately while minimizing false alarms.

Datasets
Fraud_Data.csv

Contains customer transaction information including:

User ID,
Signup Time,
Purchase Time,
Purchase Value,
Device ID,
Traffic Source,
Browser,
Gender,
Age,
IP Address,
Fraud Label (class),
IpAddress_to_Country.csv

Maps IP address ranges to countries and is used for geolocation analysis.

creditcard.csv

Contains anonymized credit card transaction features:

Time,
Amount,
V1–V28 PCA-transformed variables,
Fraud Label (Class),
Data Preprocessing

The following preprocessing steps were performed:

Missing value treatment,
Duplicate removal,
Data type correction,
IP address conversion,
Country mapping,
Feature engineering,
One-Hot Encoding,
Standard Scaling,
Train-Test Splitting,
Class Imbalance Handling using SMOTE,
Feature Engineering,
Fraud Dataset

Additional features created include:

Hour of Day,
Day of Week,
Time Since Signup,
Transaction Frequency,
Transaction Velocity,
Country-based Features,
Credit Card Dataset,
Standard Scaling of Amount and Time,
Feature selection and preprocessing,
Exploratory Data Analysis

EDA was performed to understand:

Feature distributions,
Fraud patterns,
Class imbalance,
Correlations,
Country-level fraud trends,
Relationships between features and fraud labels,
Machine Learning Models,
Baseline Model,
Logistic Regression,
Ensemble Models,
Random Forest,
XGBoost,
Model Evaluation

Models were evaluated using:

F1-Score,
AUC-PR (Area Under Precision-Recall Curve),
Confusion Matrix,
Stratified 5-Fold Cross Validation,
Best Model,
Fraud Dataset

XGBoost achieved the best overall performance by providing:

Highest F1-Score,
Highest AUC-PR,
Better detection of fraudulent transactions,
Credit Card Dataset

XGBoost also outperformed other models and demonstrated superior fraud detection capability.

Model Explainability

SHAP (SHapley Additive exPlanations) was used to:

Identify important features,
Explain model predictions,
Analyze True Positives,
Analyze False Positives,
Analyze False Negatives

Generated outputs include:

Feature Importance Plots

SHAP Summary Plots,
SHAP Force Plots
Key Findings

Fraudulent behavior is strongly influenced by transaction patterns rather than transaction amount alone.
Certain customer attributes and browser characteristics increase fraud risk.
PCA-derived variables in the credit card dataset are highly predictive of fraud.
XGBoost consistently provided the best fraud detection performance.

Business Recommendations

Apply additional verification for high-risk transactions.
Implement real-time fraud scoring using the XGBoost model.
Monitor suspicious behavioral patterns rather than relying only on transaction amounts.
Use explainable AI techniques to support fraud investigation teams.

Technologies Used

Python,
Pandas,
NumPy,
Scikit-Learn,
XGBoost,
SHAP,
Matplotlib,
Seaborn,
Jupyter Notebook
