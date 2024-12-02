import fitz
import pandas as pd
import xgboost as xgb
import re
import pickle
import streamlit as st
from PIL import Image
import os

# Load pre-trained model and scaler
model_path = "model/saved_data/best_xgb_model_sample_size_40000.pkl"
scaler_path = "model/saved_data/scaler.pkl"
label_mapping_path = "model/saved_data/label_mapping.pkl"

with open(model_path, "rb") as model_file:
    xgb_model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(label_mapping_path, "rb") as label_file:
    label_mapping = pickle.load(label_file)

# Map numeric labels back to disease labels
disease_labels = {v: k for k, v in label_mapping.items()}


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            return "\n".join(page.get_text("text") for page in pdf_file)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract numeric features from text
def extract_numeric_feature(text, keyword):
    if keyword == "Blood Pressure":
        # Regex to capture blood pressure in "systolic/diastolic" format
        match = re.search(r"Blood Pressure[:\s]*([\d]+)/([\d]+)", text)
        if match:
            systolic = float(match.group(1))
            diastolic = float(match.group(2))
            return systolic, diastolic
    else:
        match = re.search(rf"{keyword}[:\s]*([0-9.]+)", text)
        return float(match.group(1)) if match else None
    return None


# Preprocess PDF text into a feature vector
def preprocess_pdf_text(text):
    # Define feature keywords based on the content of the PDF
    feature_keywords = {
        "Heart Rate (bpm)": "Heart Rate",
        "Breathing Rate (brpm)": "Breathing Rate",
        "Oxygen Saturation (%)": "Oxygen Saturation",
        "Blood Pressure (systolic)": "Blood Pressure Systolic",
        "Blood Pressure (diastolic)": "Blood Pressure Diastolic",
        "Stress Index": "Stress Index",
        "Recovery Ability": "Recovery Ability (PNS Zone)",
        "PNS Index": "PNS Index",
        "Mean RRi (ms)": "Mean RRi",
        "RMSSD (ms)": "RMSSD",
        "SD1 (ms)": "SD1",
        "SD2 (ms)": "SD2",
        "HRV SDNN (ms)": "HRV SDNN",
        "Hemoglobin (g/dl)": "Hemoglobin",
        "Hemoglobin A1c (%)": "Hemoglobin A1c",
    }

    # Extract features from the text using regex
    data_dict = {}
    for feature, keyword in feature_keywords.items():
        extracted_value = extract_numeric_feature(text, keyword)
        if keyword == "Blood Pressure":
            if extracted_value:
                systolic, diastolic = extracted_value
                data_dict["Blood Pressure (systolic)"] = systolic
                data_dict["Blood Pressure (diastolic)"] = diastolic
        else:
            data_dict[feature] = extracted_value if not None else 0

    # Reindex to align with scaler features
    df = pd.DataFrame([data_dict]).reindex(
        columns=scaler.feature_names_in_, fill_value=0
    )

    # Ensure the data is scaled
    scaled_data = scaler.transform(df)

    return scaled_data


# Predict disease using the XGBoost model
def predict_disease(file_path):
    text = extract_text_from_pdf(file_path)
    if not text:
        return None, None

    processed_data = preprocess_pdf_text(text)
    dmatrix_data = xgb.DMatrix(processed_data)
    prediction = xgb_model.predict(dmatrix_data)
    predicted_class = int(prediction[0].argmax())  # Extract scalar value
    confidence = prediction[0][predicted_class]  # Get confidence score
    return disease_labels[predicted_class], confidence


# Function to render PDF as images
def render_pdf(file_path):
    images = []
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


# Streamlit app
st.title("Disease Prediction from PDF")

# Maintain state for uploaded files and predictions
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Upload PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Avoid duplicate processing
    if uploaded_file.name not in st.session_state.uploaded_files:
        # Process the PDF and store results
        try:
            predicted_disease, confidence = predict_disease(temp_file_path)
            if predicted_disease is not None:
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.session_state.predictions[uploaded_file.name] = (
                    predicted_disease,
                    confidence,
                )
            else:
                st.error("Prediction failed. Please check the input file.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Display the PDF
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Uploaded PDF")
        pdf_images = render_pdf(temp_file_path)
        for img in pdf_images:
            st.image(img, use_container_width=True)

    with col2:
        st.write("### Predicted Disease")
        if uploaded_file.name in st.session_state.predictions:
            disease, confidence = st.session_state.predictions[uploaded_file.name]
            st.success(f"{disease} (Confidence: {confidence:.2f})")

    # Optionally clean up temporary files after display
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# Show history of uploaded files and predictions
if st.session_state.uploaded_files:
    st.write("### Prediction History")
    for file_name in st.session_state.uploaded_files:
        disease, confidence = st.session_state.predictions[file_name]
        st.write(f"- **{file_name}**: {disease} (Confidence: {confidence:.2f})")
