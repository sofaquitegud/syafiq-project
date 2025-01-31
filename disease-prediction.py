import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tempfile
from utils import (
    preprocess_image,
    extract_text_from_pdf,
    extract_text_from_image,
    preprocess_text_to_features,
    process_extracted_features,
    render_pdf,
    correct_image_orientation,
    clean_text,
)

# Streamlit interface
st.title("Disease Prediction from PDF/Image")
option = st.radio("Select Input Method:", ["PDF Document", "Camera Image"])


def handle_pdf_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    text = extract_text_from_pdf(temp_path)
    feature_df, extracted_features = preprocess_text_to_features(text)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("PDF Preview")
        for img in render_pdf(temp_path):
            st.image(img, use_container_width=True)

    with col2:
        process_extracted_features(feature_df, extracted_features, col2)


def handle_image_upload(uploaded_image):
    img = Image.open(uploaded_image)
    img = correct_image_orientation(img)
    preprocessed_img = preprocess_image(img)
    text = extract_text_from_image(img)
    feature_df, extracted_features = preprocess_text_to_features(clean_text(text))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image Preview")
        st.image(img, caption="Original Image", use_container_width=True)
        st.image(
            preprocessed_img, caption="Preprocessed Image", use_container_width=True
        )

    with col2:
        process_extracted_features(feature_df, extracted_features, col2)


if option == "PDF Document":
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        handle_pdf_upload(uploaded_file)

elif option == "Camera Image":
    uploaded_image = st.file_uploader(
        "Upload a camera image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_image:
        handle_image_upload(uploaded_image)
