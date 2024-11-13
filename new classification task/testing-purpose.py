import pymupdf
import pandas as pd
from keras.models import load_model
import re
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scalar
model = load_model(
    r"C:\Users\syafi\Desktop\TM\Heart_Rate_Classification\new classification task\best results\cnn_model.h5"
)
with open(
    r"C:\Users\syafi\Desktop\TM\Heart_Rate_Classification\new classification task\scaler.pkl",
    "rb",
) as f:
    scaler = pickle.load(f)

    # Function to extract text from PDF file
    def extract_text_from_pdf(file_path):
        with pymupdf.open(file_path) as pdf_file:
            text = ""
            for page_num in range(pdf_file.page_count):
                page = pdf_file[page_num]
                text += page.get_text("text")  # Extract text from each page
        return text

    # Function to preprocess text and format it for CNN input
    def preprocess_pdf_text(text):
        # Extract numeric feature from text
        data_dict = {
            "Gender (0-M;1-F)": extract_numeric_feature(text, "Gender"),
            "Blood Pressure (systolic)": extract_numeric_feature(
                text, "Blood Pressure (systolic)"
            ),
            "Blood Pressure (diastolic)": extract_numeric_feature(
                text, "Blood Pressure (diastolic)"
            ),
            "Heart Rate (bpm)": extract_numeric_feature(text, "Heart Rate"),
            "Breathing Rate (brpm)": extract_numeric_feature(text, "Breathing Rate"),
            "Oxygen Saturation (%)": extract_numeric_feature(text, "Oxygen Saturation"),
            "Hemoglobin A1c (%)": extract_numeric_feature(text, "Hemoglobin A1c"),
            "HRV SDNN (ms)": extract_numeric_feature(text, "HRV SDNN"),
            "RMSSD (ms)": extract_numeric_feature(text, "RMSSD"),
            "Recovery Ability": extract_numeric_feature(text, "Recovery Ability"),
            "Mean RRi (ms)": extract_numeric_feature(text, "Mean RRi"),
            "Stress Index": extract_numeric_feature(text, "Stress Index"),
            "SNS Index": extract_numeric_feature(text, "SNS Index"),
            "PNS Index": extract_numeric_feature(text, "PNS Index"),
            "Hemoglobin (g/dl)": extract_numeric_feature(text, "Hemoglobin"),
            "SD1 (ms)": 0,  # Add any missing features with default values
            "SD2 (ms)": 0,
        }

        # Convert to DataFrame and handle missing data (set to 0)
        df = pd.DataFrame([data_dict])
        df.fillna(0, inplace=True)

        # Reorder DataFrame to match scaler's expected feature order
        df = df[scaler.feature_names_in_]

        # Scale data using the same scaler used during training
        scaled_data = scaler.transform(df)

        # Reshape for the CNN model input (batch, time_step, features)
        scaled_data = scaled_data.reshape(1, scaled_data.shape[1], 1)

        return scaled_data


# Helper function to extract numerical values for features
def extract_numeric_feature(text, keyword):
    match = re.search(f"{keyword}:?\\s*(\\d+\\.?\\d*)", text)
    return float(match.group(1)) if match else None


# Function to predict disease from processed data
def predict_disease_from_pdf(pdf_path, model):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Preprocess text to format it for CNN input
    processed_data = preprocess_pdf_text(text)

    # Make prediction
    prediction = model.predict(processed_data)
    predicted_class = prediction.argmax(axis=-1)

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
        "Aneamia",
    ]
    return disease_labels[predicted_class[0]]


# Load model and predict disease from PDF input
pdf_file_path = r"C:\Users\syafi\Desktop\work related\relate with health issue\Binah.ai Report - Anemia 1.pdf"
predicted_disease = predict_disease_from_pdf(pdf_file_path, model)
print(f"Predicted Disease: {predicted_disease}")
