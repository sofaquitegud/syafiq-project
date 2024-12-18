import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt

# Load the trained model
with open(
    "C:/Users/syafi/Desktop/syafiq-project/classification-task/model/saved_data/xgboost_model.pkl",
    "rb",
) as f:
    xgb_model = pickle.load(f)

# Load class labels
class_labels = [
    "Anaemia",
    "Arrhythmia",
    "Atherosclerosis",
    "Autonomic Dysfunction",
    "Cardiovascular Disease (CVD)",
    "Chronic Fatigue Syndrome (CFS)",
    "Diabetes",
    "Healthy",
    "Hypertension",
    "Respiratory Disease (COPD or Asthma)",
    "Stress-related Disorders",
]

# Define the rule-based logic
disease_rules = {
    "Hypertension": lambda row: (
        (
            row["Blood Pressure (systolic)"] >= 140
            or row["Blood Pressure (diastolic)"] >= 90
        )
        and 60 <= row["Heart Rate (bpm)"] <= 100
        and row["Hemoglobin A1c (%)"] <= 5.7
    ),
    "Cardiovascular Disease (CVD)": lambda row: (
        (row["Heart Rate (bpm)"] < 60 or row["Heart Rate (bpm)"] > 100)
        and row["HRV SDNN (ms)"] < 50
        and row["Oxygen Saturation (%)"] >= 95
    ),
    "Chronic Fatigue Syndrome (CFS)": lambda row: (
        row["HRV SDNN (ms)"] < 50 and row["Recovery Ability"] > 1
    ),
    "Diabetes": lambda row: row["Hemoglobin A1c (%)"] > 6.4,
    "Anaemia": lambda row: row["Hemoglobin (g/dl)"] < 13.5
    and row["Oxygen Saturation (%)"] >= 95,
    "Atherosclerosis": lambda row: (
        5.7 < row["Hemoglobin A1c (%)"] <= 6.4
        and row["Blood Pressure (systolic)"] < 140
    ),
    "Arrhythmia": lambda row: (
        row["Mean RRi (ms)"] < 600
        or row["HRV SDNN (ms)"] > 100
        and row["Heart Rate (bpm)"] >= 60
    ),
    "Stress-related Disorders": lambda row: (
        row["Stress Index"] > 70 or row["SNS Index"] > 1.0
    ),
    "Respiratory Disease (COPD or Asthma)": lambda row: (
        row["Breathing Rate (brpm)"] > 20 or row["Oxygen Saturation (%)"] < 95
    ),
    "Autonomic Dysfunction": lambda row: (
        row["PNS Index"] < -1.0 or row["SNS Index"] > 1.0
    ),
}


# Rule-based classification function
def classify_disease(row):
    for disease, rule in disease_rules.items():
        if rule(row):
            return disease
    return "Healthy"


# Sample test cases
test_cases = pd.DataFrame(
    [
        {  # Hypertension
            "Heart Rate (bpm)": 85,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 145,
            "Blood Pressure (diastolic)": 95,
            "Stress Index": 60,
            "Recovery Ability": 1,
            "PNS Index": -1.0,
            "SNS Index": 0.9,
            "Mean RRi (ms)": 800,
            "HRV SDNN (ms)": 50,
            "Hemoglobin (g/dl)": 14.0,
            "Hemoglobin A1c (%)": 5.5,
        },
        {  # CVD
            "Heart Rate (bpm)": 110,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 96,
            "Blood Pressure (systolic)": 130,
            "Blood Pressure (diastolic)": 80,
            "Stress Index": 60,
            "Recovery Ability": 1,
            "PNS Index": -0.5,
            "SNS Index": 1.2,
            "Mean RRi (ms)": 700,
            "HRV SDNN (ms)": 45,
            "Hemoglobin (g/dl)": 14.0,
            "Hemoglobin A1c (%)": 5.6,
        },
        {  # CFS
            "Heart Rate (bpm)": 75,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 120,
            "Blood Pressure (diastolic)": 80,
            "Stress Index": 65,
            "Recovery Ability": 2,
            "PNS Index": -1.2,
            "SNS Index": 0.8,
            "Mean RRi (ms)": 600,
            "HRV SDNN (ms)": 40,
            "Hemoglobin (g/dl)": 13.5,
            "Hemoglobin A1c (%)": 5.4,
        },
        {  # Diabetes
            "Heart Rate (bpm)": 85,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 130,
            "Blood Pressure (diastolic)": 85,
            "Stress Index": 60,
            "Recovery Ability": 1,
            "PNS Index": -0.5,
            "SNS Index": 1.0,
            "Mean RRi (ms)": 750,
            "HRV SDNN (ms)": 50,
            "Hemoglobin (g/dl)": 14.0,
            "Hemoglobin A1c (%)": 6.5,
        },
        {  # Anaemia
            "Heart Rate (bpm)": 75,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 97,
            "Blood Pressure (systolic)": 120,
            "Blood Pressure (diastolic)": 80,
            "Stress Index": 50,
            "Recovery Ability": 1,
            "PNS Index": -1.0,
            "SNS Index": 0.8,
            "Mean RRi (ms)": 750,
            "HRV SDNN (ms)": 55,
            "Hemoglobin (g/dl)": 12.0,
            "Hemoglobin A1c (%)": 5.5,
        },
        {  # Atherosclerosis
            "Heart Rate (bpm)": 80,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 130,
            "Blood Pressure (diastolic)": 85,
            "Stress Index": 60,
            "Recovery Ability": 1,
            "PNS Index": -1.0,
            "SNS Index": 0.9,
            "Mean RRi (ms)": 750,
            "HRV SDNN (ms)": 55,
            "Hemoglobin (g/dl)": 13.5,
            "Hemoglobin A1c (%)": 6.0,
        },
        {  # Arrhythmia
            "Heart Rate (bpm)": 110,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 97,
            "Blood Pressure (systolic)": 130,
            "Blood Pressure (diastolic)": 85,
            "Stress Index": 55,
            "Recovery Ability": 1,
            "PNS Index": -2.0,
            "SNS Index": 1.0,
            "Mean RRi (ms)": 550,
            "HRV SDNN (ms)": 120,
            "Hemoglobin (g/dl)": 14.0,
            "Hemoglobin A1c (%)": 5.5,
        },
        {  # Stress-related Disorders
            "Heart Rate (bpm)": 75,
            "Breathing Rate (brpm)": 18,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 120,
            "Blood Pressure (diastolic)": 80,
            "Stress Index": 80,
            "Recovery Ability": 1,
            "PNS Index": -1.0,
            "SNS Index": 1.0,
            "Mean RRi (ms)": 750,
            "HRV SDNN (ms)": 50,
            "Hemoglobin (g/dl)": 13.5,
            "Hemoglobin A1c (%)": 5.5,
        },
        {  # COPD or Asthma
            "Heart Rate (bpm)": 85,
            "Breathing Rate (brpm)": 22,
            "Oxygen Saturation (%)": 92,
            "Blood Pressure (systolic)": 130,
            "Blood Pressure (diastolic)": 85,
            "Stress Index": 60,
            "Recovery Ability": 1,
            "PNS Index": -1.0,
            "SNS Index": 0.9,
            "Mean RRi (ms)": 750,
            "HRV SDNN (ms)": 55,
            "Hemoglobin (g/dl)": 14.0,
            "Hemoglobin A1c (%)": 5.5,
        },
        {  # Autonomic Dysfunction
            "Heart Rate (bpm)": 75,
            "Breathing Rate (brpm)": 15,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 125,
            "Blood Pressure (diastolic)": 78,
            "Stress Index": 50,
            "Recovery Ability": 1,
            "PNS Index": -1.5,
            "SNS Index": 1.5,
            "Mean RRi (ms)": 800,
            "HRV SDNN (ms)": 60,
            "Hemoglobin (g/dl)": 13.8,
            "Hemoglobin A1c (%)": 5.6,
            "Disease": "Autonomic Dysfunction",
        },
        {  # Healthy
            "Heart Rate (bpm)": 72,
            "Breathing Rate (brpm)": 16,
            "Oxygen Saturation (%)": 98,
            "Blood Pressure (systolic)": 120,
            "Blood Pressure (diastolic)": 80,
            "Stress Index": 45,
            "Recovery Ability": 1,
            "PNS Index": 0.0,
            "SNS Index": 0.5,
            "Mean RRi (ms)": 900,
            "HRV SDNN (ms)": 50,
            "Hemoglobin (g/dl)": 14.0,
            "Hemoglobin A1c (%)": 5.4,
        },
    ]
)

# Get feature names from the model
feature_names = xgb_model.get_booster().feature_names

# Ensure all required features are in test_cases
for feature in feature_names:
    if feature not in test_cases.columns:
        test_cases[feature] = 0

# Align test cases to feature names
test_cases = test_cases[feature_names]

# Check alignment
assert len(test_cases.columns) == len(feature_names), "Feature count mismatch!"

# Compare predictions
rule_based_predictions = []
ml_predictions = []

for _, case in test_cases.iterrows():
    # Rule-based prediction
    rule_pred = classify_disease(case)
    rule_based_predictions.append(rule_pred)

    # ML-based prediction
    ml_pred = class_labels[int(xgb_model.predict(case.values.reshape(1, -1))[0])]
    ml_predictions.append(ml_pred)

# Display results
comparison_df = pd.DataFrame(
    {
        "Test Case": range(1, len(test_cases) + 1),
        "Rule-Based Prediction": rule_based_predictions,
        "Machine Learning Prediction": ml_predictions,
    }
)
print(comparison_df)

# Mismatch Analysis
mismatches = comparison_df[
    comparison_df["Rule-Based Prediction"]
    != comparison_df["Machine Learning Prediction"]
]
print("\nMismatches:\n", mismatches)

# SHAP Explanation for mismatches
if not mismatches.empty:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(test_cases.iloc[mismatches.index])

    # SHAP summary plot for mismatched cases
    print("\nSHAP Summary Plot for Mismatched Cases:")
    shap.summary_plot(shap_values, test_cases.iloc[mismatches.index])
