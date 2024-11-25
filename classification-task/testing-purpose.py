import fitz
import pandas as pd
import xgboost as xgb
import re
import pickle

# Load pre-trained model and scaler
model_path = "C:/Users/syafi/Desktop/syafiq-project/new classification task/model/saved_data/best_xgb_model_sample_size_30000.pkl"
scaler_path = "C:/Users/syafi/Desktop/syafiq-project/new classification task/model/saved_data/scaler.pkl"

with open(model_path, "rb") as model_file:
    xgb_model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Map predictions back to disease labels
disease_labels = [
    "Healthy",
    "Hypertension",
    "Diabetes",
    "Atherosclerosis",
    "Cardiovascular Disease (CVD)",
    "Respiratory Disease (COPD or Asthma)",
    "Chronic Fatigue Syndrome (CFS)",
    "Arrhythmias",
    "Stress-related Disorders",
    "Autonomic Dysfunction",
    "Anaemia",
]


# Function to extract text from a pdf
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            return "\n".join(page.get_text("text") for page in pdf_file)
    except Exception as e:
        print(f"Error extracting text from pdf: {e}")
        return ""


# Function to extract numeric features
def extract_numeric_feature(text, keyword):
    match = re.search(rf"{keyword}:?\s*(\d+\.?\d*)", text)
    return float(match.group(1)) if match else None


# Function to preprocess text for model (XGBoost) input
def preprocess_pdf_text(text):
    # Define the features to extract
    feature_keywords = {
        "Gender (0-M;1-F)": "Gender",
        "Blood Pressure (systolic)": "Blood Pressure (systolic)",
        "Blood Pressure (diastolic)": "Blood Pressure (diastolic)",
        "Heart Rate (bpm)": "Heart Rate",
        "Breathing Rate (brpm)": "Breathing Rate",
        "Oxygen Saturation (%)": "Oxygen Saturation",
        "Hemoglobin A1c (%)": "Hemoglobin A1c",
        "HRV SDNN (ms)": "HRV SDNN",
        "RMSSD (ms)": "RMSSD",
        "Recovery Ability": "Recovery Ability",
        "Mean RRi (ms)": "Mean RRi",
        "Stress Index": "Stress Index",
        "SNS Index": "SNS Index",
        "PNS Index": "PNS Index",
        "Hemoglobin (g/dl)": "Hemoglobin",
    }

    # Extract features
    data_dict = {
        feature: extract_numeric_feature(text, keyword)
        for feature, keyword in feature_keywords.items()
    }

    # Handle missing features with default values
    df = pd.DataFrame([data_dict]).fillna(0)

    # Align with scaler's expected feature order
    df = df[scaler.feature_names_in_]

    # Scale data
    scaled_data = scaler.tranform(df)

    return scaled_data


# Function to predict disease using the XGBoost model
def predict_disease(file_path):
    # Extract text
    text = extract_text_from_pdf(file_path)
    if not text:
        return "Error: Unable to extract text from PDF."

    # Preprocess text
    processed_data = preprocess_pdf_text(text)

    # Convert processed data into DMatrix format
    dmatrix_data = xgb.DMatrix(processed_data)

    # Make prediction
    prediction = xgb_model.predict(dmatrix_data)
    predicted_class = prediction.argmax(axis=-1)

    # Return the mapped disease label
    return disease_labels[predicted_class]


# Main script to run the prediction
if __name__ == "__main__":
    # Prompt user to input the pdf file path
    pdf_file_path = input("Enter the path to the pdf file: ").strip()

    # Predict disease
    predicted_disease = predict_disease(pdf_file_path)
    print(f"Predicted Disease: {predicted_disease}")
