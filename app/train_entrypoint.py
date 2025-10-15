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
import mlflow
import mlflow.xgboost


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

   # Configure MLflow
    mlflow.set_experiment("customer-churn-prediction")
    
    # Start MLflow run with a descriptive name
    run_name = f"xgboost-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        logging.info("="*60)
        logging.info(f"MLflow Run ID: {run.info.run_id}")
        logging.info(f"MLflow Run Name: {run_name}")
        logging.info("="*60)
        
        # Validate environment variables
        if None in [RAW_DATA_FILEPATH, CLEANED_DATA_FILEPATH, PROCESSED_DATA_DIR, MODEL_OUTPUT_DIR, MODEL_FILENAME]:
            logging.error("Missing one or more required environment variables. Please check your .env file.")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", "missing_env_variables")
            return

        # Log configuration parameters
        mlflow.log_param("raw_data_filepath", RAW_DATA_FILEPATH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("model_filename", MODEL_FILENAME)
        mlflow.set_tag("pipeline_version", "1.0")
        mlflow.set_tag("model_type", "XGBoost")

        # Ensure output directories exist
        ensure_directory_exists(PROCESSED_DATA_DIR)
        ensure_directory_exists(MODEL_OUTPUT_DIR)

        # Step 1: Clean the raw data using the TrainingPipeline
        logging.info("Step 1: Running data cleaning pipeline.")
        try:
            training_pipeline = TrainingPipeline(RAW_DATA_FILEPATH, CLEANED_DATA_FILEPATH)
            training_pipeline.preprocess()
            mlflow.set_tag("data_cleaning", "success")
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", "data_cleaning_error")
            mlflow.log_param("error_message", str(e))
            return

        # Step 2: Load cleaned data and split into train-test sets
        logging.info("Step 2: Loading cleaned data and performing train-test split.")
        try:
            df = training_pipeline.load_cleaned_data()
            
            # Log data statistics
            mlflow.log_param("total_samples", len(df))
            mlflow.log_param("n_features", len(df.columns) - 1)  # Exclude target
            mlflow.log_param("target_column", "Exited")
            
            # Log target distribution
            target_dist = df['Exited'].value_counts()
            mlflow.log_param("class_0_count", int(target_dist.get(0, 0)))
            mlflow.log_param("class_1_count", int(target_dist.get(1, 0)))
            mlflow.log_param("class_balance_ratio", 
                           round(target_dist.get(1, 0) / target_dist.get(0, 1), 3))
            
            X_train, X_test, y_train, y_test = split_data(df, target="Exited", test_size=TEST_SIZE)
            save_data(X_train, y_train, "train", PROCESSED_DATA_DIR)
            save_data(X_test, y_test, "test", PROCESSED_DATA_DIR)
            
            logging.info(f"Train-test split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            mlflow.set_tag("train_test_split", "success")
            
        except Exception as e:
            logging.error(f"Error during train-test split: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", "train_test_split_error")
            mlflow.log_param("error_message", str(e))
            return

        # Step 3: Train the XGBoost model
        logging.info("Step 3: Starting XGBoost model training.")
        try:
            model = train_xgb_model(X_train, y_train)
            logging.info("Model training completed successfully.")
            mlflow.set_tag("model_training", "success")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", "model_training_error")
            mlflow.log_param("error_message", str(e))
            return

        # Step 4: Evaluate the trained model
        logging.info("Step 4: Evaluating the trained model.")
        try:
            y_pred, metrics = evaluate_model(model, X_test, y_test)
            logging.info("Model evaluation completed successfully.")
            mlflow.set_tag("model_evaluation", "success")
            
            # Log confusion matrix as text artifact
            conf_matrix_str = f"Confusion Matrix:\n{metrics['confusion_matrix']}"
            with open("confusion_matrix.txt", "w") as f:
                f.write(conf_matrix_str)
            mlflow.log_artifact("confusion_matrix.txt")
            os.remove("confusion_matrix.txt")  # Clean up
            
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", "model_evaluation_error")
            mlflow.log_param("error_message", str(e))
            return

        # Step 5: Save the trained model
        logging.info("Step 5: Saving the trained model.")
        try:
            save_model(model, MODEL_FILENAME)
            logging.info(f"Model saved successfully to {os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)}.")
            mlflow.set_tag("model_saving", "success")
            mlflow.set_tag("status", "success")
        except Exception as e:
            logging.error(f"Error while saving the model: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("failure_reason", "model_saving_error")
            mlflow.log_param("error_message", str(e))
            return

        logging.info("="*60)
        logging.info("Feature preprocessing and model training pipeline completed successfully.")
        logging.info(f"MLflow UI: Run 'mlflow ui' and navigate to http://localhost:5000")
        logging.info(f"Run ID: {run.info.run_id}")
        logging.info("="*60)


if __name__ == "__main__":
    main()