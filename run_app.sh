#!/bin/bash

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Create temp directory if it doesn't exist
mkdir -p temp

# Check if models exist
if [ ! -f "best_model.pth" ] && [ ! -f "final_model.pth" ]; then
    echo "⚠️ Warning: Model files not found. You need at least one of:"
    echo "  - best_model.pth"
    echo "  - final_model.pth"
    
    read -p "Do you want to train a model now? This will take some time. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting model training..."
        python total_code.py
    else
        echo "Please add model files to the current directory before running the app."
        exit 1
    fi
fi

# Start the Streamlit app
echo "Starting Deepfake Detection & Localization app..."
streamlit run streamlit_app.py 