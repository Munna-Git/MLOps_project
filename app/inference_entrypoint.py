# app/inference_entrypoint.py - WITH MLFLOW MODEL LOADING

from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
import shap
import logging
import sys
import mlflow
import mlflow.xgboost

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.clean_data import InferencePipeline

# Load environment variables
load_dotenv()

# Initialize basic Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths from environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")
INFERENCE_CLEANED_PATH = os.getenv("INFERENCE_CLEANED_PATH")

# 🔥 NEW: Option to load model from MLflow (set in .env)
USE_MLFLOW_MODEL = os.getenv("USE_MLFLOW_MODEL", "false").lower() == "true"
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "customer-churn-xgboost")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")  # Production, Staging, or None

# Check environment variables
if None in [MODEL_PATH, TEST_DATA_PATH, INFERENCE_CLEANED_PATH]:
    logging.error("Missing one or more required environment variables. Please check your .env file.")
    raise ValueError("Environment variable configuration error.")

# Load model
try:
    if USE_MLFLOW_MODEL:
        # 🔥 NEW: Load model from MLflow Model Registry
        logging.info(f"Loading model from MLflow Registry: {MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}")
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        model = mlflow.xgboost.load_model(model_uri)
        logging.info(f"✓ Model loaded from MLflow Registry: {model_uri}")
    else:
        # Original: Load model from pickle file
        logging.info(f"Loading model from local file: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info("✓ Model loaded from local file")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Initialize SHAP explainer AFTER model loads
try:
    explainer = shap.Explainer(model)
    logging.info("✓ SHAP explainer initialized")
except Exception as e:
    logging.error(f"Error initializing SHAP explainer: {e}")
    explainer = None

# Define Inference Pipeline
inference_pipeline = InferencePipeline(
    input_filepath=TEST_DATA_PATH,
    output_filepath=INFERENCE_CLEANED_PATH
)


def get_customer_data(customer_id: int) -> pd.DataFrame | None:
    """Retrieve data for a given customer ID after preprocessing."""
    logging.info(f"Retrieving data for customer ID: {customer_id}")
    try:
        # Preprocess the test dataset using InferencePipeline
        inference_pipeline.preprocess()

        # Load the cleaned dataset
        df = pd.read_csv(INFERENCE_CLEANED_PATH)

        # Ensure CustomerId column exists before filtering
        if "CustomerId" not in df.columns:
            logging.error("CustomerId column not found in cleaned dataset.")
            return None

        # Filter the DataFrame for the given customer ID
        client_data = df[df['CustomerId'] == customer_id]

        if client_data.empty:
            logging.warning(f"Customer ID {customer_id} not found.")
            return None

        # Drop the CustomerId column before prediction
        client_data = client_data.drop(columns=['CustomerId'])
        return client_data
    except Exception as e:
        logging.error(f"Error retrieving customer data: {e}")
        return None


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    model_source = "MLflow Registry" if USE_MLFLOW_MODEL else "Local Pickle File"
    model_info = f"{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}" if USE_MLFLOW_MODEL else MODEL_PATH
    
    return jsonify({
        'status': 'running',
        'message': 'Customer Churn Prediction API',
        'model_source': model_source,
        'model_info': model_info,
        'mlflow_enabled': USE_MLFLOW_MODEL,
        'endpoints': {
            'predict_with_shap': '/score/<customer_id> [GET]',
            'predict_simple': '/<customer_id> [GET]',
            'health': '/ [GET]'
        }
    }), 200


@app.route('/score/<int:customer_id>', methods=['GET'])
def score(customer_id):
    """API endpoint for scoring a customer and returning SHAP explanations."""
    client_data = get_customer_data(customer_id)

    if client_data is None:
        return jsonify({"error": "Customer not found"}), 404

    try:
        # Make prediction
        prediction = model.predict(client_data)

        # Calculate SHAP values (only if explainer is available)
        shap_values = None
        if explainer:
            shap_values = explainer(client_data)
            shap_values_list = shap_values.values.tolist()
        else:
            shap_values_list = [[]]

        feature_names = client_data.columns.tolist()

        response = {
            "customer_id": customer_id,
            "prediction": int(prediction[0]),
            "prediction_label": "Will Churn" if prediction[0] == 1 else "Will Not Churn",
            "shap_values": dict(zip(feature_names, shap_values_list[0])) if shap_values else {},
            "model_source": "MLflow Registry" if USE_MLFLOW_MODEL else "Local File"
        }

        logging.info(f"Prediction and SHAP values computed for customer ID: {customer_id}")
        return jsonify(response)

    except Exception as e:
        error_line = e.__traceback__.tb_lineno
        error_type = type(e).__name__
        error_message = str(e)
        logging.error(f"Error during prediction for customer ID {customer_id}: "
                      f"Line {error_line}, Type: {error_type}, Message: {error_message}")

        return jsonify({
            "error": f"An error occurred during prediction",
            "details": {
                "line": error_line,
                "type": error_type,
                "message": error_message
            }
        }), 500


@app.route('/<int:customer_id>', methods=['GET'])
def predict_simple(customer_id):
    """
    Simplified endpoint for quick predictions without SHAP values.
    """
    client_data = get_customer_data(customer_id)

    if client_data is None:
        return jsonify({"error": "Customer not found"}), 404

    try:
        # Make prediction
        prediction = model.predict(client_data)

        response = {
            "customer_id": customer_id,
            "prediction": int(prediction[0]),
            "prediction_label": "Will Churn" if prediction[0] == 1 else "Will Not Churn",
            "message": "Use /score/<customer_id> for SHAP explanations",
            "model_source": "MLflow Registry" if USE_MLFLOW_MODEL else "Local File"
        }

        logging.info(f"Prediction computed for customer ID: {customer_id}")
        return jsonify(response)

    except Exception as e:
        error_line = e.__traceback__.tb_lineno
        error_type = type(e).__name__
        error_message = str(e)
        logging.error(f"Error during prediction for customer ID {customer_id}: "
                      f"Line {error_line}, Type: {error_type}, Message: {error_message}")

        return jsonify({
            "error": f"An error occurred during prediction",
            "details": {
                "line": error_line,
                "type": error_type,
                "message": error_message
            }
        }), 500


if __name__ == '__main__':
    # Validate environment variables and file paths
    logging.info("="*60)
    logging.info("Starting Customer Churn Prediction API")
    logging.info("="*60)
    
    if USE_MLFLOW_MODEL:
        logging.info(f"✓ Using MLflow Model Registry")
        logging.info(f"  Model: {MLFLOW_MODEL_NAME}")
        logging.info(f"  Stage: {MLFLOW_MODEL_STAGE}")
    else:
        logging.info(f"MODEL_PATH: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            logging.error(f"Model file not found: {MODEL_PATH}")
        else:
            logging.info("✓ Model file found")
    
    logging.info(f"TEST_DATA_PATH: {TEST_DATA_PATH}")
    if not os.path.exists(TEST_DATA_PATH):
        logging.error(f"Test data file not found: {TEST_DATA_PATH}")
    else:
        logging.info("✓ Test data file found")
    
    logging.info(f"INFERENCE_CLEANED_PATH: {INFERENCE_CLEANED_PATH}")
    logging.info("="*60)
    
    # Use a safer port (5000 is standard for Flask)
    app.run(host='0.0.0.0', port=5000, debug=True)