import fitz
import pandas as pd
import xgboost as xgb
import re
import pickle
from tkinter import Tk, filedialog

# Load pre-trained model and scaler
model_path = "C:/Users/syafi/Desktop/syafiq-project/classification-task/model/saved_data/best_xgb_model_sample_size_30000.pkl"
scaler_path = "C:/Users/syafi/Desktop/syafiq-project/classification-task/model/saved_data/scaler.pkl"
label_mapping_path = "C:/Users/syafi/Desktop/syafiq-project/classification-task/model/saved_data/label_mapping.pkl"

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
    match = re.search(rf"{keyword}[:\s]*([0-9.]+)", text)
    return float(match.group(1)) if match else None


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
        data_dict[feature] = extracted_value if extracted_value is not None else 0

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
        return "Error: Unable to extract text from PDF."

    processed_data = preprocess_pdf_text(text)
    dmatrix_data = xgb.DMatrix(processed_data)
    prediction = xgb_model.predict(dmatrix_data)
    predicted_class = int(prediction[0].argmax())  # Extract scalar value
    return disease_labels[predicted_class]


# Debugging function to print the extracted data and model prediction
def debug_predictions(file_path):
    # Extract text
    text = extract_text_from_pdf(file_path)
    print("Extracted Text: ", text)

    # Preprocess text
    processed_data = preprocess_pdf_text(text)
    print("Processed Data: ", processed_data)

    # Convert processed data into DMatrix format and make prediction
    dmatrix_data = xgb.DMatrix(processed_data)
    prediction = xgb_model.predict(dmatrix_data)
    print("Prediction Array: ", prediction)

    predicted_class = int(prediction[0].argmax())
    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Disease: {disease_labels[predicted_class]}")


# Main script
if __name__ == "__main__":
    print("Please select a PDF file for prediction.")

    # Initialize Tkinter file upload dialog
    root = Tk()
    root.withdraw()  # Hide Tkinter GUI
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    root.destroy()

    if file_path:
        # Debugging to check extraction and prediction
        debug_predictions(file_path)  # Add this line for debugging
    else:
        print("No file selected. Exiting.")
