import fitz
import pandas as pd
import xgboost as xgb
import re
import pickle
import streamlit as st
from PIL import Image
import tempfile
import logging

# Configure logging
logging.basicConfig(
    filename="predictions.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

# Define disease labels mapping
disease_labels = {
    0: "Hypertension",
    1: "Cardiovascular Disease (CVD)",
    2: "Chronic Fatigue Syndrome (CFS)",
    3: "Stress-related Disorders",
    4: "Healthy",
    5: "Diabetes",
    6: "Anaemia",
    7: "Atherosclerosis",
    8: "Arrhythmia",
    9: "Respiratory Disease (COPD or Asthma)",
    10: "Autonomic Dysfunction",
}

# Load pre-trained model
model_path = "./model/saved_data/xgb_model.pkl"
with open(model_path, "rb") as model_file:
    xgb_model = pickle.load(model_file)

model_features = xgb_model.feature_names

# Feature extraction rules
feature_extraction_rules = {
    "Heart Rate (bpm)": r"Heart Rate[:\s]*(-?[0-9.]+)",
    "Breathing Rate (brpm)": r"Breathing Rate[:\s]*(-?[0-9.]+)",
    "Oxygen Saturation (%)": r"Oxygen Saturation[:\s]*(-?[0-9.]+)",
    "Blood Pressure (systolic)": r"Blood Pressure[:\s]*([\d]+)/[\d]+",
    "Blood Pressure (diastolic)": r"Blood Pressure[:\s]*[\d]+/([\d]+)",
    "Stress Index": r"Stress Index[:\s]*(-?[0-9.]+)",
    "Recovery Ability": r"Recovery Ability[:\s]*(Normal|Medium|Low)",
    "PNS Index": r"PNS Index[:\s]*(-?[0-9.]+)",
    "Mean RRi (ms)": r"Mean RRi[:\s]*(-?[0-9.]+)",
    "RMSSD (ms)": r"RMSSD[:\s]*(-?[0-9.]+)",
    "SD1 (ms)": r"SD1[:\s]*(-?[0-9.]+)",
    "SD2 (ms)": r"SD2[:\s]*(-?[0-9.]+)",
    "HRV SDNN (ms)": r"HRV SDNN[:\s]*(-?[0-9.]+)",
    "Hemoglobin (g/dl)": r"Hemoglobin[:\s]*(-?[0-9.]+)",
    "Hemoglobin A1c (%)": r"Hemoglobin A1c[:\s]*(-?[0-9.]+)",
    "SNS Index": r"SNS Index[:\s]*(-?[0-9.]+)",
    "Gender (0-M;1-F)": r"Gender[:\s]*(0|1)",
}


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            return "\n".join(page.get_text("text") for page in pdf_file)
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract features based on rules
def extract_features_from_text(text, rules):
    features = {}
    for feature, pattern in rules.items():
        match = re.search(pattern, text)
        if feature == "Recovery Ability":
            value = (
                {"Normal": 0, "Medium": 1, "Low": 2}.get(match.group(1), 0)
                if match
                else None
            )
        else:
            value = float(match.group(1)) if match else None
        features[feature] = value if value is not None else 0
    return features


# Preprocess text and create input for the model
def preprocess_pdf_text(text):
    features = extract_features_from_text(text, feature_extraction_rules)
    df = pd.DataFrame([features]).reindex(columns=model_features, fill_value=0)
    return df, features


# Prediction function
def predict_disease(file_path):
    text = extract_text_from_pdf(file_path)
    if not text:
        return None, None, None

    feature_data, extracted_features = preprocess_pdf_text(text)

    # Validate features
    if feature_data.isnull().values.any() or feature_data.eq(0).all(axis=1).values[0]:
        st.error("Insufficient or invalid data extracted from the PDF.")
        return None, None, None

    dmatrix_data = xgb.DMatrix(feature_data)
    prediction = xgb_model.predict(dmatrix_data)

    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)

    predicted_class = int(prediction[0].argmax())
    confidence = prediction[0][predicted_class]
    predicted_disease = disease_labels.get(predicted_class, "Unknown Disease")

    # Log prediction and features
    logging.info(f"Extracted Features: {extracted_features}")
    logging.info(
        f"Predicted Disease: {predicted_disease}, Confidence: {confidence:.2f}"
    )

    return predicted_disease, confidence, extracted_features


# Render PDF as images
def render_pdf(file_path):
    images = []
    with fitz.open(file_path) as pdf_document:
        for page in pdf_document:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


# Streamlit interface
st.title("Disease Prediction from PDF")

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    # Avoid duplicate processing
    if uploaded_file.name not in st.session_state.uploaded_files:
        try:
            with st.spinner("Processing..."):
                predicted_disease, confidence, extracted_features = predict_disease(
                    temp_file_path
                )
            if predicted_disease:
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.session_state.predictions[uploaded_file.name] = (
                    predicted_disease,
                    confidence,
                    extracted_features,
                )
            else:
                st.error("Prediction failed. Please check the input file.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Display PDF and results
    col1, col2 = st.columns([2, 3])
    with col1:
        st.write("### Uploaded PDF")
        pdf_images = render_pdf(temp_file_path)
        for img in pdf_images:
            st.image(img, use_container_width=True)

    with col2:
        st.write("### Predicted Disease")
        if uploaded_file.name in st.session_state.predictions:
            disease, confidence, extracted_features = st.session_state.predictions[
                uploaded_file.name
            ]
            st.success(f"{disease} (Confidence: {confidence:.2f})")

            st.write("### Extracted Features")
            st.table(
                pd.DataFrame.from_dict(
                    extracted_features, orient="index", columns=["Value"]
                )
            )

if st.session_state.uploaded_files:
    st.write("### Prediction History")
    for file_name in st.session_state.uploaded_files:
        disease, confidence, _ = st.session_state.predictions[file_name]
        st.write(f"- **{file_name}**: {disease} (Confidence: {confidence:.2f})")

# Add option to clear history
if st.button("Clear History"):
    st.session_state.uploaded_files = []
    st.session_state.predictions = {}
    st.success("Prediction history cleared.")
