import fitz
import numpy as np
import pandas as pd
import re
import streamlit as st
import tempfile
import logging
from PIL import Image
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
xgb_model = pd.read_pickle(model_path)


# Function to extract text from PDF using fitz and OCR fallback
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            raw_text = ""
            for page in pdf_file:
                raw_text += page.get_text("text")
            if not raw_text.strip():  # Fallback to OCR if no text
                pages = convert_from_path(file_path, dpi=300)
                raw_text = " ".join(
                    image_to_string(page, config="--psm 6") for page in pages
                )
        # Debug: Log raw extracted text
        logging.info(f"Raw Text Extracted: {raw_text}")
        print("DEBUG: Raw Extracted Text:\n", raw_text)  # Print for debugging
        return re.sub(r"\s+", " ", raw_text.strip())
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract features based on regex patterns
def extract_features_from_text(text, rules):
    normalized_text = text.upper()  # Normalize text to uppercase
    features = {}
    for feature, pattern in rules.items():
        match = re.search(pattern, normalized_text, re.DOTALL)
        if match:
            # Replace incorrect characters (colons and commas) with periods
            raw_value = match.group(1).replace(":", ".").replace(",", ".")
            try:
                value = (
                    float(raw_value)
                    if feature != "Recovery Ability"
                    else {"NORMAL": 0, "MEDIUM": 1, "LOW": 2}.get(
                        match.group(1).upper(), np.nan
                    )
                )
            except ValueError:
                value = np.nan  # Handle invalid numeric values
        else:
            value = np.nan
        features[feature] = value

    # Log missing features
    for key, val in features.items():
        if pd.isna(val):
            logging.warning(f"Feature '{key}' not found or misread in the PDF.")

    return features


# Preprocess the extracted features
def preprocess_pdf_text(text):
    feature_patterns = {
        "Heart Rate (bpm)": r"(?i)HEART\s*RATE\s*[:\s]*([\d.]+)\s*BPM",
        "Breathing Rate (brpm)": r"(?i)BREATHING\s*RATE\s*[:\s]*([\d.]+)\s*BRPM",
        "Oxygen Saturation (%)": r"(?i)OXYGEN\s*SATURATION\s*[:\s]*([\d.]+)\s*%",
        "Blood Pressure (systolic)": r"(?i)BLOOD\s*PRESSURE\s*[:\s]*([\d]+)\/[\d]+",
        "Blood Pressure (diastolic)": r"(?i)BLOOD\s*PRESSURE\s*[:\s]*[\d]+\/([\d]+)",
        "Stress Index": r"(?i)STRESS\s*INDEX\s*[:\s]*([\d.]+)",
        "Recovery Ability": r"(?i)RECOVERY\s*ABILITY\s*\(PNS\s*ZONE\)\s*[:\s]*(NORMAL|MEDIUM|LOW)",
        "PNS Index": r"(?i)PNS\s*INDEX\s*[:\s]*(-?[\d.]+)",
        "SNS Index": r"(?i)SNS\s*INDEX\s*[:\s]*(-?[\d.]+)",
        "RMSSD (ms)": r"(?i)RMSSD\s*[:\s]*([\d.]+)\s*MS",
        "SD1 (ms)": r"(?i)SD1\s*[:\s]*([\d.]+)\s*MS",
        "SD2 (ms)": r"(?i)SD2\s*[:\s]*([\d.]+)\s*MS",
        "HRV SDNN (ms)": r"(?i)HRV\s*SDNN\s*[:\s]*([\d.]+)\s*MS",
        "Hemoglobin (g/dl)": r"(?i)HEMOGLOBIN\s*[:\s]*([\d.]+)\s*G/DL",
        "Hemoglobin A1c (%)": r"(?i)HEMOGLOBIN\s*A[1IL]C\s*[:\s]*([\d.:,]+)\s*%",
        "Mean RRi (ms)": r"(?i)MEAN\s*RRI\s*[:\s]*([\d.]+)\s*MS",
    }

    features = extract_features_from_text(text, feature_patterns)

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
