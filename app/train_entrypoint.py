# app/train_entrypoint.py

# Feature and Training Pipeline
from dotenv import load_dotenv
import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.clean_data import TrainingPipeline
from src.data.train_test_split import split_data, save_data
from src.models.train_model import train_xgb_model, evaluate_model, save_model


# Load environment variables
load_dotenv()

# Constants
RAW_DATA_FILEPATH = os.getenv("RAW_DATA_FILEPATH")
CLEANED_DATA_FILEPATH = os.getenv("CLEANED_DATA_FILEPATH")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")
MODEL_FILENAME = os.getenv("MODEL_FILENAME")
TEST_SIZE = os.getenv("TEST_SIZE")

# Ensure TEST_SIZE is valid float
try:
    TEST_SIZE = float(TEST_SIZE) if TEST_SIZE else 0.1
except ValueError:
    TEST_SIZE = 0.1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_directory_exists(directory: str) -> None:
    """Create directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")


def main() -> None:
    """Executes the feature preprocessing and model training pipeline."""
    # Validate environment variables
    if None in [RAW_DATA_FILEPATH, CLEANED_DATA_FILEPATH, PROCESSED_DATA_DIR, MODEL_OUTPUT_DIR, MODEL_FILENAME]:
        logging.error("Missing one or more required environment variables. Please check your .env file.")
        return

    # Ensure output directories exist
    ensure_directory_exists(PROCESSED_DATA_DIR)
    ensure_directory_exists(MODEL_OUTPUT_DIR)

    # Step 1: Clean the raw data using the TrainingPipeline
    logging.info("Running data cleaning pipeline.")
    try:
        training_pipeline = TrainingPipeline(RAW_DATA_FILEPATH, CLEANED_DATA_FILEPATH)
        training_pipeline.preprocess()
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        return

    # Step 2: Load cleaned data and split into train-test sets
    logging.info("Loading cleaned data and performing train-test split.")
    try:
        df = training_pipeline.load_cleaned_data()
        X_train, X_test, y_train, y_test = split_data(df, target="Exited", test_size=TEST_SIZE)
        save_data(X_train, y_train, "train", PROCESSED_DATA_DIR)
        save_data(X_test, y_test, "test", PROCESSED_DATA_DIR)
        logging.info(f"Train-test split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        return

    # Step 3: Train the XGBoost model
    logging.info("Starting XGBoost model training.")
    try:
        model = train_xgb_model(X_train, y_train)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return

    # Step 4: Evaluate the trained model
    logging.info("Evaluating the trained model.")
    try:
        y_pred = evaluate_model(model, X_test, y_test)
        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return

    # Step 5: Save the trained model
    logging.info("Saving the trained model.")
    try:
        # Pass only the filename since save_model already joins with MODEL_OUTPUT_DIR
        save_model(model, MODEL_FILENAME)
        logging.info(f"Model saved successfully to {os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)}.")
    except Exception as e:
        logging.error(f"Error while saving the model: {e}")
        return

    logging.info("Feature preprocessing and model training pipeline completed successfully.")


if __name__ == "__main__":
    main()
