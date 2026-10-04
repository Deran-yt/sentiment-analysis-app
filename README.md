# Sentiment Analysis Web App

This is a Streamlit web application for Sentiment Analysis.

## Directory Structure
The structure includes:
- `data/`: Datasets used for training/testing.
- `models/`: Trained models stored as joblib files.
- `notebooks/`: Jupyter notebooks for data exploration.
- `src/`: Core Python code containing ML logic (`train.py`, `predict.py`, `preprocess.py`).
- `app.py`: Entry point for the Streamlit UI.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (optional/if applicable):
   ```bash
   python src/train.py
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
