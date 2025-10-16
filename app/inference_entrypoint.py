# app/inference_entrypoint.py - FIXED FOR PRODUCTION DEPLOYMENT

from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
import shap
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.clean_data import InferencePipeline

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔥 FIX: Get absolute paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths from environment variables with fallback to relative paths
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models/xgboost_model.pkl"))
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", os.path.join(BASE_DIR, "data/raw/InferenceData.csv"))
INFERENCE_CLEANED_PATH = os.getenv("INFERENCE_CLEANED_PATH", os.path.join(BASE_DIR, "data/cleaned/InferenceCleaned.csv"))

# MLflow options
USE_MLFLOW_MODEL = os.getenv("USE_MLFLOW_MODEL", "false").lower() == "true"

# Global variables
model = None
explainer = None
inference_pipeline = None

# 🔥 FIX: Check if required files exist
def check_required_files():
    """Check if model and data files exist"""
    required_files = {
        "Model": MODEL_PATH,
        "Test Data": TEST_DATA_PATH
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            logger.error(f"✗ {name} not found: {path}")
            missing_files.append(name)
        else:
            logger.info(f"✓ {name} found: {path}")
    
    return len(missing_files) == 0, missing_files


def initialize_app():
    """Initialize model, explainer, and pipeline with validation"""
    global model, explainer, inference_pipeline
    
    logger.info("=" * 60)
    logger.info("INITIALIZING CUSTOMER CHURN PREDICTION API")
    logger.info("=" * 60)
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Test Data Path: {TEST_DATA_PATH}")
    logger.info(f"Inference Cleaned Path: {INFERENCE_CLEANED_PATH}")
    
    # 🔥 FIX: Check if files exist
    files_ok, missing = check_required_files()
    if not files_ok:
        error_msg = f"Missing required files: {', '.join(missing)}"
        logger.error(error_msg)
        logger.error("=" * 60)
        logger.error("DEPLOYMENT INSTRUCTIONS:")
        logger.error("1. Ensure 'models/xgboost_model.pkl' is in your Git repo")
        logger.error("2. Ensure 'data/raw/InferenceData.csv' is in your Git repo")
        logger.error("3. Check your .gitignore doesn't exclude these files")
        logger.error("=" * 60)
        # Don't raise error - allow API to start but show error on requests
        return False
    
    # Load model
    try:
        if USE_MLFLOW_MODEL:
            import mlflow.xgboost
            logger.info("Loading model from MLflow Registry")
            model_name = os.getenv("MLFLOW_MODEL_NAME", "customer-churn-xgboost")
            model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
            model_uri = f"models:/{model_name}/{model_stage}"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"✓ Model loaded from MLflow: {model_uri}")
        else:
            logger.info(f"Loading model from local file")
            with open(MODEL_PATH, 'rb') as model_file:
                model = pickle.load(model_file)
            logger.info("✓ Model loaded from local file")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        return False
    
    # Initialize SHAP explainer
    try:
        explainer = shap.Explainer(model)
        logger.info("✓ SHAP explainer initialized")
    except Exception as e:
        logger.warning(f"⚠ Could not initialize SHAP explainer: {e}")
        explainer = None
    
    # Initialize inference pipeline
    try:
        # 🔥 FIX: Create cleaned directory if it doesn't exist
        os.makedirs(os.path.dirname(INFERENCE_CLEANED_PATH), exist_ok=True)
        
        inference_pipeline = InferencePipeline(
            input_filepath=TEST_DATA_PATH,
            output_filepath=INFERENCE_CLEANED_PATH
        )
        logger.info("✓ Inference pipeline initialized")
    except Exception as e:
        logger.error(f"✗ Error initializing inference pipeline: {e}")
        return False
    
    logger.info("=" * 60)
    logger.info("✓ API INITIALIZATION COMPLETE")
    logger.info("=" * 60)
    return True


def get_customer_data(customer_id: int) -> pd.DataFrame | None:
    """Retrieve data for a given customer ID after preprocessing."""
    if model is None:
        logger.error("Model not loaded. Cannot process request.")
        return None
    
    logger.info(f"Retrieving data for customer ID: {customer_id}")
    try:
        # Preprocess the test dataset using InferencePipeline
        inference_pipeline.preprocess()

        # Load the cleaned dataset
        df = pd.read_csv(INFERENCE_CLEANED_PATH)

        # Ensure CustomerId column exists before filtering
        if "CustomerId" not in df.columns:
            logger.error("CustomerId column not found in cleaned dataset.")
            return None

        # Filter the DataFrame for the given customer ID
        client_data = df[df['CustomerId'] == customer_id]

        if client_data.empty:
            logger.warning(f"Customer ID {customer_id} not found.")
            return None

        # Drop the CustomerId column before prediction
        client_data = client_data.drop(columns=['CustomerId'])
        return client_data
    except Exception as e:
        logger.error(f"Error retrieving customer data: {e}")
        return None


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not loaded"
    
    return jsonify({
        'status': 'running',
        'message': 'Customer Churn Prediction API',
        'model_status': model_status,
        'model_source': "MLflow Registry" if USE_MLFLOW_MODEL else "Local File",
        'endpoints': {
            'health': '/ [GET]',
            'predict_simple': '/<customer_id> [GET]',
            'predict_with_shap': '/score/<customer_id> [GET]'
        },
        'deployment': 'production'
    }), 200


@app.route('/score/<int:customer_id>', methods=['GET'])
def score(customer_id):
    """API endpoint for scoring a customer and returning SHAP explanations."""
    
    # 🔥 FIX: Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "details": "The model file is missing. Please check deployment configuration."
        }), 503
    
    client_data = get_customer_data(customer_id)

    if client_data is None:
        return jsonify({"error": "Customer not found"}), 404

    try:
        # Make prediction
        prediction = model.predict(client_data)

        # Calculate SHAP values (only if explainer is available)
        shap_values_dict = {}
        if explainer:
            try:
                shap_values = explainer(client_data)
                feature_names = client_data.columns.tolist()
                shap_values_list = shap_values.values[0].tolist()
                shap_values_dict = dict(zip(feature_names, shap_values_list))
            except Exception as e:
                logger.warning(f"Could not compute SHAP values: {e}")

        response = {
            "customer_id": customer_id,
            "prediction": int(prediction[0]),
            "prediction_label": "Will Churn" if prediction[0] == 1 else "Will Not Churn",
            "shap_values": shap_values_dict,
            "model_source": "MLflow Registry" if USE_MLFLOW_MODEL else "Local File"
        }

        logger.info(f"✓ Prediction for customer {customer_id}: {prediction[0]}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during prediction for customer {customer_id}: {e}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.route('/<int:customer_id>', methods=['GET'])
def predict_simple(customer_id):
    """Simplified endpoint for quick predictions without SHAP values."""
    
    # 🔥 FIX: Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "details": "The model file is missing. Please check deployment configuration."
        }), 503
    
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

        logger.info(f"✓ Prediction for customer {customer_id}: {prediction[0]}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during prediction for customer {customer_id}: {e}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": {
            "health": "/",
            "predict_simple": "/<customer_id>",
            "predict_with_shap": "/score/<customer_id>"
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), 500


# 🔥 FIX: Initialize app on import (for gunicorn)
app_initialized = initialize_app()

if not app_initialized:
    logger.warning("⚠ API started but model not loaded. Check logs for details.")


if __name__ == '__main__':
    # For local development only
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)