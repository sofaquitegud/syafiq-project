import fitz
import numpy as np
import pandas as pd
import xgboost as xgb
import re
import streamlit as st
import tempfile
import logging
from PIL import Image
from joblib import load
from pytesseract import image_to_string
from pdf2image import convert_from_path

logging.basicConfig(
    filename="predictions.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

# Disease labels mapping
disease_labels = {
    0: "Anaemia",
    1: "Arrhythmia",
    2: "Atherosclerosis",
    3: "Autonomic Dysfunction",
    4: "Cardiovascular Disease (CVD)",
    5: "Chronic Fatigue Syndrome (CFS)",
    6: "Diabetes",
    7: "Healthy",
    8: "Hypertension",
    9: "Respiratory Disease (COPD or Asthma)",
    10: "Stress-related Disorders",
}

model_features = [
    "Heart Rate (bpm)",
    "Breathing Rate (brpm)",
    "Oxygen Saturation (%)",
    "Blood Pressure (systolic)",
    "Blood Pressure (diastolic)",
    "Stress Index",
    "Recovery Ability",
    "PNS Index",
    "SNS Index",
    "RMSSD (ms)",
    "SD2 (ms)",
    "Hemoglobin A1c (%)",
    "Mean RRi (ms)",
    "SD1 (ms)",
    "HRV SDNN (ms)",
    "Hemoglobin (g/dl)",
]

# Load pre-trained model
model_path = "C:/Users/syafi/Desktop/syafiq-project/classification-task/model/saved_data/xgboost_model.pkl"
xgb_model = load(model_path)


# Function to extract text from PDF using fitz and OCR fallback
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            raw_text = ""
            for page in pdf_file:
                raw_text += page.get_text("text")
            if not raw_text.strip():  # Fallback to OCR if no text
                pages = convert_from_path(file_path)
                raw_text = " ".join(image_to_string(page) for page in pages)
        return re.sub(r"\s+", " ", raw_text.strip())
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract features based on regex patterns
def extract_features_from_text(text, rules):
    features = {}
    for feature, pattern in rules.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = (
                float(match.group(1))
                if feature != "Recovery Ability"
                else {"Normal": 0, "Medium": 1, "Low": 2}.get(match.group(1), np.nan)
            )
        else:
            value = np.nan
        features[feature] = value
    return features


# Preprocess the extracted features
def preprocess_pdf_text(text):
    # Feature extraction
    features = extract_features_from_text(
        text,
        {
            "Heart Rate (bpm)": r"(?i)Heart\s*Rate\s*[:\s]*([\d.]+)\s*bpm",
            "Breathing Rate (brpm)": r"(?i)Breathing\s*Rate\s*[:\s]*([\d.]+)\s*brpm",
            "Oxygen Saturation (%)": r"(?i)Oxygen\s*Saturation\s*[:\s]*([\d.]+)\s*%",
            "Blood Pressure (systolic)": r"(?i)Blood\s*Pressure\s*[:\s]*([\d]+)/[\d]+",
            "Blood Pressure (diastolic)": r"(?i)Blood\s*Pressure\s*[:\s]*[\d]+/([\d]+)",
            "Stress Index": r"(?i)Stress\s*Index\s*[:\s]*([\d.]+)",
            "Recovery Ability": r"(?i)Recovery\s*Ability\s*\(PNS\s*Zone\)\s*[:\s]*(Normal|Medium|Low)",
            "PNS Index": r"(?i)PNS\s*Index\s*[:\s]*(-?[\d.]+)",
            "Mean RRi (ms)": r"(?i)Mean\s*RRi\s*[:\s]*([\d.]+)\s*ms",
            "RMSSD (ms)": r"(?i)RMSSD\s*[:\s]*([\d.]+)\s*ms",
            "SD1 (ms)": r"(?i)SD1\s*[:\s]*([\d.]+)\s*ms",
            "SD2 (ms)": r"(?i)SD2\s*[:\s]*([\d.]+)\s*ms",
            "HRV SDNN (ms)": r"(?i)HRV\s*SDNN\s*[:\s]*([\d.]+)\s*ms",
            "Hemoglobin (g/dl)": r"(?i)Hemoglobin\s*[:\s]*([\d.]+)\s*g/dl",
            "Hemoglobin A1c (%)": r"(?i)Hemoglobin\s*A1c\s*[:\s]*([\d.]+)\s*%",
            "SNS Index": r"(?i)SNS\s*Index\s*[:\s]*(-?[\d.]+)",
        },
    )

    # Impute missing features with default values
    for key in model_features:
        if pd.isna(features.get(key)):
            features[key] = {
                "Heart Rate (bpm)": 70,
                "Breathing Rate (brpm)": 16,
                "Oxygen Saturation (%)": 98,
                "Blood Pressure (systolic)": 120,
                "Blood Pressure (diastolic)": 80,
                "Stress Index": 50,
                "Recovery Ability": 0,
                "PNS Index": 0.0,
                "SNS Index": 0.5,
                "RMSSD (ms)": 40,
                "SD2 (ms)": 40,
                "Hemoglobin A1c (%)": 5.4,
                "Mean RRi (ms)": 900,
                "SD1 (ms)": 30,
                "HRV SDNN (ms)": 50,
                "Hemoglobin (g/dl)": 14.0,
            }.get(key, np.nan)

    return pd.DataFrame([features]).reindex(columns=model_features), features


# Render PDF pages as images
def render_pdf(file_path):
    images = []
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


st.title("Disease Prediction from PDF")
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    # Extract PDF content
    text = extract_text_from_pdf(temp_path)
    pdf_images = render_pdf(temp_path)
    feature_df, extracted_features = preprocess_pdf_text(text)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("PDF Preview")
        for img in pdf_images:
            st.image(img, use_container_width=True)

    with col2:
        # Disease Prediction
        try:
            prediction = xgb_model.predict(feature_df)
            st.success(f"Prediction: {disease_labels[int(prediction[0])]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

        st.subheader("Extracted Features")
        features_df = pd.DataFrame(
            list(extracted_features.items()), columns=["Feature", "Value"]
        )
        st.dataframe(features_df)
