import numpy as np
import pandas as pd
import streamlit as st
import fitz
import re
import tempfile
import cv2
import logging
import os
import urllib.request
import pytesseract
import xgboost as xgb
from xgboost import Booster
from PIL import Image
from pytesseract import image_to_string
from pdf2image import convert_from_path


# Configure logging
logging.basicConfig(level=logging.DEBUG)

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Constants
MODEL_PATH_GITHUB = "https://raw.githubusercontent.com/sofaquitegud/syafiq-project/refs/heads/main/classification-task/xgboost_model.json"
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "xgboost_model.json")
MAX_PAGES = 3

# Function to download model from GitHub
def download_model(url, model_path):
    try:
        urllib.request.urlretrieve(url, model_path)
        logging.info(f"Model downloaded to {model_path}")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")

# Function to determine whether running on Streamlit Cloud or locally
def is_st_cloud():
    try:
        return os.environ.get("STREAMLIT_SERVER", "") != ""
    except KeyError:
        return False

# Temporary path where the model will be downloaded in the Streamlit Cloud environment
model_tmp_path = "/tmp/xgboost_model.json" if is_st_cloud() else LOCAL_MODEL_PATH
logging.debug(f"Model path set to : {model_tmp_path}")

# Download the model from GitHub or load locally if running local
if is_st_cloud():
    download_model(MODEL_PATH_GITHUB, model_tmp_path)

# Load pre-trained model
xgb_model = Booster()

# Check if the file exists
if os.path.exists(model_tmp_path):
    xgb_model.load_model(model_tmp_path)
    logging.info("Model loaded successfully.")
else:
    logging.error(f"Model file does not exist at {model_tmp_path}")
    st.error("Model file does not exist. Please ensure the model is downloaded correctly.")

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

# Model features
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

# Image preprocessing
def preprocess_image(image):
    image_cv = np.array(image.convert("L"))  # Grayscale
    image_cv = cv2.fastNlMeansDenoising(image_cv, None, 30, 7, 21)  # Noise removal
    _, image_cv = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(image_cv)

# Extract text from image
def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    ocr_text = image_to_string(preprocessed_image, config="--psm 6")
    logging.debug(f"OCR Text: {ocr_text[:200]}") # Log only the first 200 characters
    return clean_text(ocr_text)

# Check if image is clear
def is_image_clear(image):
    image_cv = np.array(image.convert("L"))  # Grayscale
    variance_of_laplacian = cv2.Laplacian(image_cv, cv2.CV_64F).var()
    return variance_of_laplacian >= 100

# Extract text from PDF with OCR fallback
def extract_text_from_pdf(file_path, max_pages=MAX_PAGES):
    try:
        with fitz.open(file_path) as pdf_file:
            raw_text = ""
            for page_num in range(min(len(pdf_file), max_pages)):
                page_text = pdf_file[page_num].get_text("text")
                raw_text += page_text
            if raw_text.strip():
                logging.info("Text successfully extracted from PDF.")
                return clean_text(raw_text), None

        # OCR Fallback
        logging.warning("Direct text extraction failed. Switching to OCR.")
        pages = convert_from_path(file_path, dpi=300, first_page=1, last_page=max_pages)
        processed_text = [image_to_string(preprocess_image(page), config="--psm 4 --oem 3") for page in pages]
        ocr_text_combined = clean_text(" ".join(processed_text))
        return ocr_text_combined, pages

    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return "", None

# Clean extracted text
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\$D2|S\$D2", "SD2", text)  # Fix SD2 misrecognitions
    text = re.sub(r"\$D1|S\$D1", "SD1", text)  # Fix SD1 misrecognitions
    text = re.sub(r"\s*\.\s*", ".", text)  # Remove unnecessary spaces around periods
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"\s*([0-9]+)\s*\.\s*([0-9]+)", r"\1.\2", text)  # Fix decimal formatting
    return text.strip().upper()

# Extract features from text using regex patterns
def extract_features_from_text(text, rules):
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
                value = float(match.group(1).replace(",", "."))
                logging.debug(f"Extracted {pattern}: {value}")
                return value
            except ValueError:
                logging.warning(f"ValueError for pattern {pattern}, using default {default_value}")
                return default_value
        logging.warning(f"Pattern {pattern} not found, using default {default_value}")
        return default_value

    def parse_categorical_feature(pattern, mapping, default_value):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = mapping.get(match.group(1).upper(), default_value)
            logging.debug(f"Extracted {pattern}: {value}")
            return value
        logging.warning(f"Pattern {pattern} not found, using default {default_value}")
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
        "SD1 (ms)": r"SD1|[^\w]SD1[^\w]*[:\s]*([\d.]+)",
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
    
    logging.debug(f"Extracted features: {features}")
    return features

# Process text to features
def preprocess_text_to_features(text):
    features = extract_features_from_text(text, model_features)
    return pd.DataFrame([features]).reindex(columns=model_features), features

# Render PDF as images for preview
def render_pdf(file_path, max_pages=MAX_PAGES):
    images = []
    with fitz.open(file_path) as pdf:
        for page_num in range(min(len(pdf), max_pages)):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

# Streamlit interface
st.title("Disease Prediction from PDF/Image")
option = st.radio("Select Input Method:", ["PDF Document", "Camera Image"])

def handle_pdf_upload(uploaded_file):
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

def handle_image_upload(uploaded_image):
    img = Image.open(uploaded_image)
    if not is_image_clear(img):
        st.warning("The uploaded image appears to be blurry. Please try again with a clearer image.")
    text = extract_text_from_image(img)
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

if option == "PDF Document":
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        handle_pdf_upload(uploaded_file)

elif option == "Camera Image":
    uploaded_image = st.file_uploader("Upload a camera image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        handle_image_upload(uploaded_image)
