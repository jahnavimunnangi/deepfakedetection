import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import torch
import time
from base64 import b64encode

# Import the deepfake localizer
from deepfake_localization import DeepfakeLocalizer

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection & Localization",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("Deepfake Detection & Localization")
st.markdown("""
This application detects manipulated images (deepfakes) and visualizes which regions of the image appear to be manipulated. 
Upload an image to see if it's real or fake.
""")

# Sidebar for model selection and settings
st.sidebar.header("Model Settings")

# Model path selection
model_path = st.sidebar.selectbox(
    "Select a model",
    ["best_model.pth", "final_model.pth"],
    help="Choose which trained model to use for detection"
)

# Heatmap intensity slider
heatmap_alpha = st.sidebar.slider(
    "Heatmap intensity",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help="Adjust the transparency of the heatmap overlay"
)

# Device selection
device = st.sidebar.radio(
    "Device",
    ["CPU", "GPU (CUDA)"],
    index=0 if not torch.cuda.is_available() else 1,
    help="Select the device to run the model on"
)
device = "cuda" if device == "GPU (CUDA)" and torch.cuda.is_available() else "cpu"

# Main app functionality
@st.cache_resource
def load_model(model_path, device):
    """Load the deepfake localizer model with caching"""
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at {model_path}. Please check the file path.")
        return None
    
    try:
        localizer = DeepfakeLocalizer(model_path=model_path, device=device)
        return localizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_image_download_link(img, filename, text):
    """Generate a link to download an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    href = f'<a href="data:file/png;base64,{b64encode(img_str).decode()}" download="{filename}">{text}</a>'
    return href

# Load model (with caching)
with st.spinner("Loading model... This may take a moment."):
    localizer = load_model(model_path, device)
    if localizer:
        st.sidebar.success(f"Model loaded successfully on {device.upper()}")
    else:
        st.sidebar.error("Failed to load model. Please check model path or try a different model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image to analyze", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    # Save image temporarily
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    image.save(temp_path)
    
    # Analysis button
    if st.button("Analyze Image"):
        try:
            if localizer:
                with st.spinner("Analyzing image..."):
                    # Time the analysis
                    start_time = time.time()
                    
                    # Analyze image
                    results = localizer.analyze_image(temp_path)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    # Display original image in first column
                    with col1:
                        st.image(
                            results['original_image'], 
                            caption=f"Original Image", 
                            use_column_width=True
                        )
                    
                    # Display heatmap or result in second column
                    with col2:
                        if results['prediction'] == 1 and results['heatmap'] is not None:
                            # Create matplotlib figure for heatmap visualization
                            fig = localizer.visualize_results(
                                results, 
                                output_path=None, 
                                alpha=heatmap_alpha
                            )
                            
                            # Convert matplotlib figure to image
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            heatmap_img = Image.open(buf)
                            
                            # Display heatmap image
                            st.image(
                                heatmap_img, 
                                caption="Deepfake Heatmap Visualization",
                                use_column_width=True
                            )
                            plt.close(fig)
                        else:
                            st.image(
                                results['original_image'],
                                caption="No manipulation detected",
                                use_column_width=True
                            )
                    
                    # Display results and confidence
                    st.subheader("Analysis Results")
                    result_col1, result_col3 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            "Prediction", 
                            "FAKE" if results['prediction'] == 1 else "REAL",
                            delta=None
                        )
                    
                    with result_col3:
                        st.metric(
                            "Processing Time", 
                            f"{processing_time:.2f} seconds",
                            delta=None
                        )
                    
                    # Additional explanation
                    if results['prediction'] == 1:
                        st.info("""
                        🔍 **Detected as FAKE**
                        
                        The heatmap shows regions that the model considers manipulated. 
                        Brighter/redder areas indicate higher probability of manipulation.
                        """)
                    else:
                        st.success("""
                        ✅ **Detected as REAL**
                        
                        The model did not detect signs of manipulation in this image.
                        """)
                    
                    # Clean up
                    os.remove(temp_path)
            else:
                st.error("Model not loaded. Please check model path.")
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            
# Add information about the model
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This application uses a dual-stream deep learning model that analyzes both spatial and frequency domain features to detect manipulated images.

The model was trained on the Hemg/deepfake-and-real-images dataset from Hugging Face.
""")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and PyTorch") 