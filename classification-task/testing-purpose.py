import fitz
import numpy as np
import pandas as pd
import re
import streamlit as st
import tempfile
from PIL import Image
import xgboost as xgb
from xgboost import Booster
from pytesseract import image_to_string
from pdf2image import convert_from_path
import cv2
import logging

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
xgb_model = Booster()
xgb_model.load_model("xgboost_model.json")

# Function to preprocess images for OCR
def preprocess_image(image):
    image_cv = np.array(image.convert("L"))  # Convert to grayscale
    image_cv = cv2.adaptiveThreshold(
        image_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    image_cv = cv2.fastNlMeansDenoising(image_cv, h=10)
    return Image.fromarray(image_cv)

# Function to extract text from PDF with OCR fallback
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            raw_text = "".join(page.get_text("text") for page in pdf_file)
            if raw_text.strip():
                return clean_text(raw_text), None

        # Fallback to OCR
        pages = convert_from_path(file_path, dpi=600)
        processed_text = []
        preprocessed_images = []
        for page in pages:
            preprocessed_image = preprocess_image(page)
            ocr_text = image_to_string(preprocessed_image, config="--psm 6 --oem 3")
            processed_text.append(ocr_text)
            preprocessed_images.append(preprocessed_image)

        # Combine OCR text and clean
        ocr_text_combined = clean_text(" ".join(processed_text))

        return ocr_text_combined, preprocessed_images
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return "", None

# Clean extracted text
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"[:,;]", ".", text)  # Replace colons/semicolons with periods
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces to a single space
    return text.strip().upper()

# Function to extract features based on regex patterns
def extract_features_from_text(text, rules):
    features = {}
    for feature, pattern in rules.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw_value = match.group(1).replace(",", ".")
            try:
                if feature == "Recovery Ability":
                    value = {"NORMAL": 0, "MEDIUM": 1, "LOW": 2}.get(raw_value.upper(), np.nan)
                else:
                    value = float(raw_value)
            except ValueError:
                value = np.nan
        else:
            value = np.nan
        features[feature] = value

    default_values = {
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
        "Hemoglobin (g/dl)": 14.0
    }
    for key in model_features:
        if pd.isna(features.get(key)):
            features[key] = default_values[key]
    return features

# Preprocess text to features
def preprocess_text_to_features(text):
    feature_patterns = {
        "Heart Rate (bpm)": r"HEART\s*RATE\s*[:\s]*([\d.]+)",
        "Breathing Rate (brpm)": r"BREATHING\s*RATE\s*[:\s]*([\d.]+)",
        "Oxygen Saturation (%)": r"OXYGEN\s*SATURATION\s*[:\s]*([\d.]+)",
        "Blood Pressure (systolic)": r"BLOOD\s*PRESSURE\s*[:\s]*([\d]+)\/[\d]+",
        "Blood Pressure (diastolic)": r"BLOOD\s*PRESSURE\s*[:\s]*[\d]+\/([\d]+)",
        "Stress Index": r"STRESS\s*INDEX\s*[:\s]*([\d.]+)",
        "Recovery Ability": r"RECOVERY\s*ABILITY\s*\(PNS\s*ZONE\)\s*[:\s]*\b(NORMAL|MEDIUM|LOW)\b",
        "PNS Index": r"PNS\s*INDEX\s*[:\s]*(-?[\d.]+)",
        "SNS Index": r"SNS\s*INDEX\s*[:\s]*(-?[\d.]+)",
        "RMSSD (ms)": r"RMSSD\s*[:\s]*([\d.]+)",
        "SD1 (ms)": r"SD1\s*[:\s]*([\d.]+)",
        "SD2 (ms)": r"SD2\s*[:\s]*([\d.]+)",
        "HRV SDNN (ms)": r"HRV\s*SDNN\s*[:\s]*([\d.]+)",
        "Hemoglobin (g/dl)": r"HEMOGLOBIN\s*[:\s]*([\d.]+)",
        "Hemoglobin A1c (%)": r"HEMOGLOBIN\s*A[1IL]C\s*[:\s]*([\d.]+)",
        "Mean RRi (ms)": r"MEAN\s*RRI\s*[:\s]*([\d.]+)",
    }
    features = extract_features_from_text(text, feature_patterns)
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

# Streamlit interface
st.title("Disease Prediction from PDF/Image")
option = st.radio("Select Input Method:", ["PDF Document", "Camera Image"])

if option == "PDF Document":
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        text, _ = extract_text_from_pdf(temp_path)

        feature_df, extracted_features = preprocess_text_to_features(text)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("PDF Preview")
            for img in render_pdf(temp_path):
                st.image(img, use_container_width=True)

        with col2:
            # Convert DataFrame to DMatrix
            dmatrix_features = xgb.DMatrix(feature_df)

            # Make predictions
            prediction = xgb_model.predict(dmatrix_features)

            # Ensure the prediction is processed correctly
            try:
                if prediction.ndim == 2:
                    # Extract the index of the maximum probability
                    predicted_label = int(np.argmax(prediction, axis=1)[0])
                    st.success(f"Prediction: {disease_labels[predicted_label]}")
                else:
                    st.error(f"Unexpected prediction format: {prediction}. Please verify the model output.")
            except Exception as e:
                st.error(f"Error processing prediction: {e}")
                st.error(f"Prediction output: {prediction}")

            st.subheader("Extracted Features")
            st.dataframe(pd.DataFrame(list(extracted_features.items()), columns=["Feature", "Value"]))

elif option == "Camera Image":
    uploaded_image = st.file_uploader("Upload a camera image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        text = image_to_string(img, config="--psm 6")
        feature_df, extracted_features = preprocess_text_to_features(clean_text(text))
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Image Preview")
            st.image(img, use_container_width=True)

        with col2:
            dmatrix_features = xgb.DMatrix(feature_df)

            # Make predictions
            prediction = xgb_model.predict(dmatrix_features)

            # Ensure the prediction is processed correctly
            try:
                if prediction.ndim == 2:
                    # Extract the index of the maximum probability
                    predicted_label = int(np.argmax(prediction, axis=1)[0])
                    st.success(f"Prediction: {disease_labels[predicted_label]}")
                else:
                    st.error(f"Unexpected prediction format: {prediction}. Please verify the model output.")
            except Exception as e:
                st.error(f"Error processing prediction: {e}")
                st.error(f"Prediction output: {prediction}")

            st.subheader("Extracted Features")
            st.dataframe(pd.DataFrame(list(extracted_features.items()), columns=["Feature", "Value"]))
