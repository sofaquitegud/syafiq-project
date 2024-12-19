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

logging.basicConfig(level=logging.DEBUG)

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

# Improved image preprocessing
def preprocess_image(image):
    image_cv = np.array(image.convert("L"))  # Grayscale
    _, image_cv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(image_cv)

# Extract text from PDF with fallback to OCR
def extract_text_from_pdf(file_path, max_pages=3):
    """Extract text from PDF with optional OCR fallback."""
    try:
        with fitz.open(file_path) as pdf_file:
            raw_text = ""
            for page_num in range(min(len(pdf_file), max_pages)):
                page_text = pdf_file[page_num].get_text("text")
                raw_text += page_text
            if raw_text.strip():
                logging.info("Text successfully extracted from PDF.")
                return clean_text(raw_text), None

        # Fallback to OCR
        logging.warning("Direct text extraction failed. Switching to OCR.")
        pages = convert_from_path(file_path, dpi=300, first_page=1, last_page=max_pages)
        processed_text = []
        for idx, page in enumerate(pages):
            preprocessed_image = preprocess_image(page)
            ocr_text = image_to_string(preprocessed_image, config="--psm 4 --oem 3")
            logging.debug(f"OCR Text (Page {idx + 1}): {ocr_text}")
            processed_text.append(ocr_text)

        ocr_text_combined = clean_text(" ".join(processed_text))
        return ocr_text_combined, pages

    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return "", None

# Clean extracted text
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\$D2|S\$D2", "SD2", text)  # Fix SD2 misrecognitions
    text = re.sub(r"\s*\.\s*", ".", text)  # Remove unnecessary spaces around periods
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"\s*([0-9]+)\s*\.\s*([0-9]+)", r"\1.\2", text)  # Fix decimal formatting
    return text.strip().upper()


# Extract features from text using regex patterns
def extract_features_from_text(text, rules):
    # Define default values at the beginning
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
        "Hemoglobin (g/dl)": 14.0,
    }

    def parse_feature(pattern, default_value):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                return default_value
        return default_value

    def parse_categorical_feature(pattern, mapping, default_value):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return mapping.get(match.group(1).upper(), default_value)
        return default_value

    features = {}
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
        "SD2 (ms)": r"SD2|[^\w]SD2[^\w]*[:\s]*([\d.]+)",
        "HRV SDNN (ms)": r"HRV\s*SDNN\s*[:\s]*([\d.]+)",
        "Hemoglobin (g/dl)": r"HEMOGLOBIN\s*[:\s]*([\d.]+)",
        "Hemoglobin A1c (%)": r"HEMOGLOBIN\s*A[1IL]C\s*[:\s]*([\d.]+)",
        "Mean RRi (ms)": r"MEAN\s*RRI\s*[:\s]*([\d.]+)",
    }

    for feature, pattern in feature_patterns.items():
        if feature == "Recovery Ability":
            features[feature] = parse_categorical_feature(
                pattern, {"NORMAL": 0, "MEDIUM": 1, "LOW": 2}, default_values[feature]
            )
        else:
            features[feature] = parse_feature(pattern, default_values[feature])

    # Fill missing features with default values
    for key in model_features:
        features.setdefault(key, default_values[key])
    return features

# Process text to features
def preprocess_text_to_features(text):
    features = extract_features_from_text(text, model_features)
    return pd.DataFrame([features]).reindex(columns=model_features), features

# Render PDF as images for preview
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
            dmatrix_features = xgb.DMatrix(feature_df)
            prediction = xgb_model.predict(dmatrix_features)

            try:
                if prediction.ndim == 2:
                    predicted_label = int(np.argmax(prediction, axis=1)[0])
                    st.success(f"Prediction: {disease_labels[predicted_label]}")
                else:
                    st.error(f"Unexpected prediction format: {prediction}.")
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
            prediction = xgb_model.predict(dmatrix_features)

            try:
                if prediction.ndim == 2:
                    predicted_label = int(np.argmax(prediction, axis=1)[0])
                    st.success(f"Prediction: {disease_labels[predicted_label]}")
                else:
                    st.error(f"Unexpected prediction format: {prediction}.")
            except Exception as e:
                st.error(f"Error processing prediction: {e}")
                st.error(f"Prediction output: {prediction}")

            st.subheader("Extracted Features")
            st.dataframe(pd.DataFrame(list(extracted_features.items()), columns=["Feature", "Value"]))
