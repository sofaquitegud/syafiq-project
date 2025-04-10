import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import fitz
import re
import urllib.request
import easyocr
import xgboost as xgb
import cv2
import logging
from pathlib import Path
from PIL import Image, ExifTags
from xgboost import Booster

logging.basicConfig(level=logging.DEBUG)

# Initialize OCR reader
ocr_reader = easyocr.Reader(["en"])

# Model paths
MODEL_PATH_GITHUB = "https://raw.githubusercontent.com/sofaquitegud/syafiq-project/refs/heads/main/xgboost_model.json"
LOCAL_MODEL_PATH = "./xgboost_model.json"
MAX_PAGES = 3

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


def download_model(url, model_path):
    try:
        urllib.request.urlretrieve(url, model_path)
        logging.info(f"Model downloaded to {model_path}")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")


def is_st_cloud():
    return os.environ.get("STREAMLIT_SERVER", "") != ""


# Determine model path
model_tmp_path = "/tmp/xgboost_model.json" if is_st_cloud() else LOCAL_MODEL_PATH

if is_st_cloud() and not os.path.exists(model_tmp_path):
    download_model(MODEL_PATH_GITHUB, model_tmp_path)

# Load the model
xgb_model = Booster()

if os.path.exists(model_tmp_path):
    xgb_model.load_model(model_tmp_path)
    logging.info("Model loaded successfully.")
else:
    logging.error(f"Model file does not exist at {model_tmp_path}")


# === Image & PDF Processing Functions ===
def preprocess_image(image):
    image_cv = np.array(image.convert("L"))  # Convert to grayscale
    denoised = cv2.fastNlMeansDenoising(
        image_cv, h=30, templateWindowSize=7, searchWindowSize=21
    )
    thresholded = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    return Image.fromarray(thresholded)


def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except AttributeError:
        pass
    return image


def extract_text_from_image(image):
    try:
        preprocessed_image = preprocess_image(image)
        ocr_result = ocr_reader.readtext(np.array(preprocessed_image), detail=0)
        return clean_text(" ".join(ocr_result))
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return ""


def extract_text_from_pdf(file_path, max_pages=MAX_PAGES):
    with fitz.open(file_path) as pdf_file:
        raw_text = ""
        for page_num in range(min(len(pdf_file), max_pages)):
            page_text = pdf_file[page_num].get_text("text")
            raw_text += page_text
        return clean_text(raw_text)


def clean_text(text):
    text = re.sub(r"PNS\s*Index(?![\s:])", "PNS Index:", text, flags=re.IGNORECASE)
    text = re.sub(
        r"PNS\s*Index\s*Mean", "PNS Index: -1 Mean", text, flags=re.IGNORECASE
    )
    text = text.replace("Oxvgen", "Oxygen")
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s{2,}", " ", text).strip()

    logging.debug(f"Cleaned Text: {text}")
    return text.upper()


def extract_features_from_text(text, rules):
    def parse_feature(patterns):
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and match.group(1):
                try:
                    value = float(match.group(1).replace(",", "."))
                    logging.debug(f"Extracted {pattern}: {value}")
                    return value
                except ValueError:
                    logging.warning(
                        f"ValueError for pattern {pattern}, skipping this pattern"
                    )
        logging.warning(
            f"None of the patterns for this feature matched, setting value to null"
        )
        return 0

    def parse_categorical_feature(pattern, mapping):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = mapping.get(match.group(1).upper(), None)
            logging.debug(f"Extracted {pattern}: {value}")
            return value
        logging.warning(f"Pattern {pattern} not found, setting value to null")
        return 0

    features = {}
    feature_patterns = {
        "Heart Rate (bpm)": [r"HEART\s*RATE\s*[:\s]*([\d.]+)"],
        "Breathing Rate (brpm)": [r"BREATHING\s*RATE\s*[:\s]*([\d.]+)"],
        "Oxygen Saturation (%)": [r"OXYGEN\s*SATURATION\s*[:\s]*([\d.]+)"],
        "Blood Pressure (systolic)": [r"BLOOD\s*PRESSURE\s*[:\s]*([\d]+)\/[\d]+"],
        "Blood Pressure (diastolic)": [r"BLOOD\s*PRESSURE\s*[:\s]*[\d]+\/([\d]+)"],
        "Stress Index": [r"STRESS\s*INDEX\s*[:\s]*([\d.]+)"],
        "Recovery Ability": [
            r"RECOVERY\s*ABILITY\s*\(PNS\s*ZONE\)\s*[:\s]*\b(NORMAL|MEDIUM|LOW)\b"
        ],
        "PNS Index": [r"PNS\s*INDEX\s*[:\s]*(-?\d+(?:\.\d+)?)(?=\s*MEAN|$)"],
        "SNS Index": [r"SNS\s*INDEX\s*[:\s]*(-?[\d.]+)"],
        "RMSSD (ms)": [r"RMSSD\s*[:\s]*([\d.]+)", r"RMSSD[^\w]*[:\s]*([\d.]+)"],
        "SD1 (ms)": [r"SD1\s*[:\s]*([\d.]+)", r"SD1[^\w]*[:\s]*([\d.]+)"],
        "SD2 (ms)": [r"SD2\s*[:\s]*([\d.]+)", r"SD2[^\w]*[:\s]*([\d.]+)"],
        "HRV SDNN (ms)": [r"HRV\s*SDNN\s*[:\s]*([\d.]+)"],
        "Hemoglobin (g/dl)": [r"HEMOGLOBIN\s*[:\s]*([\d.]+)"],
        "Hemoglobin A1c (%)": [r"HEMOGLOBIN\s*A[1IL]C\s*[:\s]*([\d.]+)"],
        "Mean RRi (ms)": [
            r"MEAN\s*RRI\s*[:\s]*([\d.]+)",
            r"MEAN\s*RRi\s*[:\s]*([\d.]+)",
        ],
    }

    for feature, patterns in feature_patterns.items():
        if feature == "Recovery Ability":
            features[feature] = parse_categorical_feature(
                patterns[0], {"NORMAL": 0, "MEDIUM": 1, "LOW": 2}
            )
        else:
            features[feature] = parse_feature(patterns)

    for key in model_features:
        features.setdefault(key, None)

    return features


def preprocess_text_to_features(text):
    features = extract_features_from_text(text, model_features)
    return pd.DataFrame([features]).reindex(columns=model_features), features


def render_pdf(file_path, max_pages=MAX_PAGES):
    images = []
    with fitz.open(file_path) as pdf:
        for page_num in range(min(len(pdf), max_pages)):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


def process_extracted_features(feature_df, extracted_features, col2):
    try:
        dmatrix_features = xgb.DMatrix(feature_df)
        prediction = xgb_model.predict(dmatrix_features)

        if prediction.ndim == 2:
            predicted_label = int(np.argmax(prediction, axis=1)[0])
            col2.success(f"Prediction: {disease_labels[predicted_label]}")
        else:
            col2.error("Unexpected prediction format.")
    except Exception as e:
        col2.error(f"Error processing prediction: {e}")

    col2.subheader("Extracted Features")
    col2.dataframe(
        pd.DataFrame(list(extracted_features.items()), columns=["Feature", "Value"])
    )
