# Deepfake Detection & Localization using Spatial–Frequency Features

This project detects **manipulated (deepfake) images** and highlights the **exact regions** that are likely to be fake using a dual-stream deep learning model and Grad-CAM visualization.

## 🎯 Objective

- Classify an input image as **REAL** or **FAKE**
- Localize manipulated regions using a **heatmap overlay**
- Provide an easy-to-use **Streamlit web app** for demo

## 🧠 Model Overview

The system uses a **dual-stream deep learning model**:

1. **Spatial Stream**
   - Takes the RGB image as input
   - Learns texture and visual artifacts in the spatial domain
   - Uses a CNN backbone (e.g., Xception / ResNet)

2. **Frequency Stream**
   - Converts image to frequency domain (using `FrequencyTransformer`)
   - Learns artifacts in magnitude & phase components
   - Helps detect subtle manipulations not visible in raw pixels

The outputs of both streams are fused and passed through a classifier to predict:
- `1` → FAKE  
- `0` → REAL  

For localization, **Grad-CAM** is applied on the spatial stream to generate a heatmap of suspicious regions.

## 📦 Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy, OpenCV, PIL
- Matplotlib

## 📂 Main Files

- `streamlit_app.py` – Streamlit UI for uploading an image and viewing results  
- `deepfake_localization.py` – DeepfakeLocalizer class + Grad-CAM visualization  
- `total_code.py` – Model architecture and preprocessing config  
- `training_code.ipynb` – Model training notebook  
- `requirements.txt` – List of dependencies  

## 🚀 How to Run (Locally)

1. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate  # On Linux/Mac
