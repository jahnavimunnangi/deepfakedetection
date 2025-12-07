@echo off
SETLOCAL

:: Check if venv exists, if not create it
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing requirements...
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate
)

:: Create temp directory if it doesn't exist
IF NOT EXIST temp mkdir temp

:: Check if models exist
IF NOT EXIST best_model.pth (
    IF NOT EXIST final_model.pth (
        echo Warning: Model files not found. You need at least one of:
        echo   - best_model.pth
        echo   - final_model.pth
        
        SET /P TRAIN="Do you want to train a model now? This will take some time. (y/n) "
        IF /I "%TRAIN%" EQU "y" (
            echo Starting model training...
            python total_code.py
        ) ELSE (
            echo Please add model files to the current directory before running the app.
            EXIT /B 1
        )
    )
)

:: Start the Streamlit app
echo Starting Deepfake Detection ^& Localization app...
streamlit run streamlit_app.py

ENDLOCAL 