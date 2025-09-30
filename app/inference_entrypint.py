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

# Initialize basic Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths from environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")
INFERENCE_CLEANED_PATH = os.getenv("INFERENCE_CLEANED_PATH")

try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
        logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Initialize SHAP explainer
explainer = shap.Explainer(model)

# Define Inference Pipeline
inference_pipeline = InferencePipeline(input_filepath=TEST_DATA_PATH, output_filepath=INFERENCE_CLEANED_PATH)


def get_customer_data(customer_id: int) -> pd.DataFrame:
    """Retrieve data for a given customer ID after preprocessing."""
    logging.info(f"Retrieving data for customer ID: {customer_id}")
    try:
        # Preprocess the test dataset using InferencePipeline
        inference_pipeline.preprocess()

        # Load the cleaned dataset
        df = pd.read_csv(INFERENCE_CLEANED_PATH)

        # Filter the DataFrame for the given customer ID
        client_data = df[df['CustomerId'] == customer_id]

        # Check if client data exists
        if client_data.empty:
            logging.warning(f"Customer ID {customer_id} not found.")
            return None

        # Drop the CustomerId column before prediction
        client_data = client_data.drop(columns=['CustomerId'])
        return client_data
    except Exception as e:
        logging.error(f"Error retrieving customer data: {e}")
        return None


@app.route('/score/<int:customer_id>', methods=['GET'])
def score(customer_id):
    client_data = get_customer_data(customer_id)

    if client_data is None:
        return jsonify({"error": "Customer not found"}), 404

    try:
        # Make prediction
        prediction = model.predict(client_data)

        # Calculate SHAP values
        shap_values = explainer(client_data)

        # Convert SHAP values to a format that can be returned as JSON
        shap_values_list = shap_values.values.tolist()
        feature_names = client_data.columns.tolist()

        response = {
            "customer_id": customer_id,
            "prediction": int(prediction[0]),  # Assuming binary classification (0 or 1)
            "shap_values": dict(zip(feature_names, shap_values_list[0]))
        }

        logging.info(f"Prediction and SHAP values computed for customer ID: {customer_id}")
        return jsonify(response)

    except Exception as e:
        error_line = e.__traceback__.tb_lineno
        error_type = str(type(e).__name__)
        error_message = str(e)
        logging.error(f"Error during prediction for customer ID {customer_id}: Line {error_line}, Type: {error_type}, Message: {error_message}")

        return jsonify({"error": f"An error occurred during prediction: Line {error_line}, Type: {error_type}, Message: {error_message}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)