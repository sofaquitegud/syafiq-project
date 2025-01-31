# Disease Prediction from PDF/Image

## 📌 Project Overview
This project predicts diseases using extracted text from PDF documents or images. It leverages **EasyOCR** for text extraction and **XGBoost** for disease classification. The Streamlit-based UI allows users to upload a PDF or an image to analyze health-related metrics.

---

## 🏗 Project Structure
```
📂 disease-prediction
├── 📄 disease-prediction.py   # Streamlit UI
├── 📄 utils.py                # Helper functions for text extraction & ML model
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # Project documentation
```

---

## 🚀 Features
- 📄 **Extracts text from PDF files** (first 3 pages)
- 📷 **Extracts text from images** (OCR-based preprocessing)
- 📊 **Predicts disease based on extracted text**
- 🎨 **User-friendly Streamlit interface**

---

## ⚡ Quick Start

### 1️⃣ Clone the repository
```sh
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
```

### 2️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```sh
streamlit run disease-prediction.py
```

---

## 📦 Dependencies
Ensure you have the following Python packages installed:
```txt
easyocr
fitz
numpy
pandas
streamlit
xgboost
opencv-python-headless
Pillow
```
To install them, run:
```sh
pip install -r requirements.txt
```

---

## 🔬 How It Works
1. **Upload a PDF or an Image**: The app processes the input using EasyOCR (for images) or PyMuPDF (for PDFs).
2. **Text Extraction & Preprocessing**: The extracted text is cleaned and mapped to relevant health parameters.
3. **Disease Prediction**: The processed data is fed into an XGBoost model that predicts the most likely disease.
4. **Results Displayed**: The app outputs the predicted disease.

---

## 🏆 Contributors
- **Muhammad Syafiq Farhan Bin Mohd Faridz** – AI Developer Intern at TM Research & Development

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments
- **EasyOCR** for image-to-text recognition
- **XGBoost** for machine learning
- **Streamlit** for UI development

If you find this project useful, consider ⭐ starring the repository!

