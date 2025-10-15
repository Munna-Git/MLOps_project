# app/inference_entrypoint.py - REFACTORED WITH PYDANTIC

from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
import shap
import logging
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import get_settings
from src.schemas.api_schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionWithSHAPResponse,
    ErrorResponse,
    HealthCheckResponse,
    create_prediction_response,
    create_shap_response,
    create_error_response
)
from src.data.clean_data import InferencePipeline
from pydantic import ValidationError

# Get validated settings
settings = get_settings()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and explainer
model = None
explainer = None
inference_pipeline = None


def initialize_app():
    """Initialize model, explainer, and pipeline with validation"""
    global model, explainer, inference_pipeline
    
    logger.info("=" * 60)
    logger.info("INITIALIZING CUSTOMER CHURN PREDICTION API")
    logger.info("=" * 60)
    logger.info(f"Model Path: {settings.model_path}")
    logger.info(f"Test Data Path: {settings.test_data_path}")
    logger.info(f"Inference Cleaned Path: {settings.inference_cleaned_path}")
    
    # Validate paths exist
    if not settings.model_path.exists():
        logger.error(f"✗ Model file not found: {settings.model_path}")
        raise FileNotFoundError(f"Model file not found: {settings.model_path}")
    
    if not settings.test_data_path.exists():
        logger.error(f"✗ Test data file not found: {settings.test_data_path}")
        raise FileNotFoundError(f"Test data file not found: {settings.test_data_path}")
    
    # Load model
    try:
        with open(settings.model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        raise
    
    # Initialize SHAP explainer
    try:
        explainer = shap.Explainer(model)
        logger.info("✓ SHAP explainer initialized")
    except Exception as e:
        logger.warning(f"⚠ Could not initialize SHAP explainer: {e}")
        explainer = None
    
    # Initialize inference pipeline
    try:
        inference_pipeline = InferencePipeline(
            input_filepath=str(settings.test_data_path),
            output_filepath=str(settings.inference_cleaned_path)
        )
        logger.info("✓ Inference pipeline initialized")
    except Exception as e:
        logger.error(f"✗ Error initializing inference pipeline: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("✓ API INITIALIZATION COMPLETE")
    logger.info("=" * 60)


def get_customer_data(customer_id: int) -> pd.DataFrame | None:
    """Retrieve and preprocess data for a given customer ID"""
    logger.info(f"Retrieving data for customer ID: {customer_id}")
    
    try:
        # Validate customer_id using Pydantic
        request_data = PredictionRequest(customer_id=customer_id)
        
        # Preprocess the test dataset
        inference_pipeline.preprocess()
        
        # Load cleaned dataset
        df = pd.read_csv(settings.inference_cleaned_path)
        
        # Check if CustomerId exists
        if "CustomerId" not in df.columns:
            logger.error("CustomerId column not found in cleaned dataset")
            return None
        
        # Filter for customer
        client_data = df[df['CustomerId'] == customer_id]
        
        if client_data.empty:
            logger.warning(f"Customer ID {customer_id} not found")
            return None
        
        # Drop CustomerId before prediction
        client_data = client_data.drop(columns=['CustomerId'])
        
        logger.info(f"✓ Retrieved data for customer {customer_id}: {client_data.shape}")
        return client_data
        
    except ValidationError as e:
        logger.error(f"Validation error for customer_id {customer_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving customer data: {e}")
        return None


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint with validated response"""
    try:
        health_response = HealthCheckResponse(
            status="running",
            message="Customer Churn Prediction API",
            version="1.0.0",
            model_loaded=model is not None,
            endpoints={
                'health': '/ [GET]',
                'predict_simple': '/<customer_id> [GET]',
                'predict_with_shap': '/score/<customer_id> [GET]'
            }
        )
        
        return jsonify(health_response.model_dump()), 200
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        error_response = create_error_response(
            error="Health check failed",
            details={"message": str(e)}
        )
        return jsonify(error_response.model_dump()), 500


@app.route('/<int:customer_id>', methods=['GET'])
def predict_simple(customer_id: int):
    """
    Simple prediction endpoint without SHAP values.
    Returns validated PredictionResponse.
    """
    try:
        # Get customer data
        client_data = get_customer_data(customer_id)
        
        if client_data is None:
            error_response = create_error_response(
                error="Customer not found",
                details={"customer_id": customer_id}
            )
            return jsonify(error_response.model_dump()), 404
        
        # Make prediction
        prediction = model.predict(client_data)[0]
        
        # Get prediction probability for confidence
        try:
            prediction_proba = model.predict_proba(client_data)[0]
            confidence = float(prediction_proba[prediction])
        except:
            confidence = None
        
        # Create validated response
        response = create_prediction_response(
            customer_id=customer_id,
            prediction=int(prediction),
            confidence=confidence,
            message="Use /score/<customer_id> for SHAP explanations"
        )
        
        logger.info(f"✓ Prediction for customer {customer_id}: {prediction}")
        return jsonify(response.model_dump()), 200
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        error_response = create_error_response(
            error="Validation failed",
            details={"validation_errors": e.errors()}
        )
        return jsonify(error_response.model_dump()), 400
        
    except Exception as e:
        logger.error(f"Prediction error for customer {customer_id}: {e}")
        error_response = create_error_response(
            error="Prediction failed",
            details={
                "customer_id": customer_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        return jsonify(error_response.model_dump()), 500


@app.route('/score/<int:customer_id>', methods=['GET'])
def score(customer_id: int):
    """
    Prediction endpoint with SHAP explanations.
    Returns validated PredictionWithSHAPResponse.
    """
    try:
        # Get customer data
        client_data = get_customer_data(customer_id)
        
        if client_data is None:
            error_response = create_error_response(
                error="Customer not found",
                details={"customer_id": customer_id}
            )
            return jsonify(error_response.model_dump()), 404
        
        # Make prediction
        prediction = model.predict(client_data)[0]
        
        # Get prediction probability for confidence
        try:
            prediction_proba = model.predict_proba(client_data)[0]
            confidence = float(prediction_proba[prediction])
        except:
            confidence = None
        
        # Calculate SHAP values
        shap_values_dict = {}
        if explainer:
            try:
                shap_values = explainer(client_data)
                feature_names = client_data.columns.tolist()
                shap_values_list = shap_values.values[0].tolist()
                shap_values_dict = dict(zip(feature_names, shap_values_list))
                logger.info(f"✓ SHAP values computed for customer {customer_id}")
            except Exception as e:
                logger.warning(f"Could not compute SHAP values: {e}")
        else:
            logger.warning("SHAP explainer not available")
        
        # Create validated response with SHAP
        response = create_shap_response(
            customer_id=customer_id,
            prediction=int(prediction),
            shap_values=shap_values_dict,
            confidence=confidence,
            top_n=5
        )
        
        logger.info(f"✓ Prediction with SHAP for customer {customer_id}: {prediction}")
        return jsonify(response.model_dump()), 200
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        error_response = create_error_response(
            error="Validation failed",
            details={"validation_errors": e.errors()}
        )
        return jsonify(error_response.model_dump()), 400
        
    except Exception as e:
        logger.error(f"Scoring error for customer {customer_id}: {e}")
        error_response = create_error_response(
            error="Scoring failed",
            details={
                "customer_id": customer_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        return jsonify(error_response.model_dump()), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with validated response"""
    error_response = create_error_response(
        error="Endpoint not found",
        details={"path": request.path}
    )
    return jsonify(error_response.model_dump()), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with validated response"""
    error_response = create_error_response(
        error="Internal server error",
        details={"message": str(error)}
    )
    return jsonify(error_response.model_dump()), 500


if __name__ == '__main__':
    try:
        # Initialize application
        initialize_app()
        
        # Run Flask app with settings
        logger.info(f"Starting API on {settings.api_host}:{settings.api_port}")
        app.run(
            host=settings.api_host,
            port=settings.api_port,
            debug=settings.api_debug
        )
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise