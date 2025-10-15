# src/models/train_model.py

import os
import pickle
import logging
import numpy as np
from xgboost import XGBClassifier
from src.models.evaluate_model import get_model_metrics
import mlflow
import mlflow.xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATA_DIR: str = "data/processed"   # Input training data
MODEL_OUTPUT_DIR: str = "models"   # Output directory for model artifacts
MODEL_FILENAME: str = "xgboost_model.pkl"
SHAP_PLOT_FILENAME: str = "shap_summary_plot.png"  # reserved for shap_utils usage

# XGBoost parameters
XGB_PARAMS: dict = {
    "colsample_bytree": 1.0,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 1,
    "n_estimators": 150,
    "scale_pos_weight": 1,
    "subsample": 1.0,
    "random_state": 42
}


def train_xgb_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train the XGBoost model with the provided parameters."""
    logging.info("Training XGBoost model.")

    # Log hyperparameters to MLflow
    mlflow.log_params(XGB_PARAMS)
    mlflow.log_param("objective", "binary:logistic")
    mlflow.log_param("eval_metric", "logloss")


    model = XGBClassifier(
        **XGB_PARAMS,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=True)
    logging.info("XGBoost model training completed.")
    return model


def evaluate_model(model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, dict]:
    """Evaluate the model and return predictions and metrics."""
    logging.info("Evaluating model.")
    y_pred = model.predict(X_test)
    metrics = get_model_metrics(y_test, y_pred)
    
    # Log metrics to MLflow
    mlflow.log_metric("f1_score", metrics['f1_score'])
    mlflow.log_metric("f2_score", metrics['f2_score'])
    mlflow.log_metric("precision", metrics['precision'])
    mlflow.log_metric("recall", metrics['recall'])
    
    # Log classification report metrics (class-specific)
    if 'classification_report' in metrics:
        report = metrics['classification_report']
        # Log metrics for class 0 (no churn)
        if '0' in report:
            mlflow.log_metric("class_0_precision", report['0']['precision'])
            mlflow.log_metric("class_0_recall", report['0']['recall'])
            mlflow.log_metric("class_0_f1", report['0']['f1-score'])
        # Log metrics for class 1 (churn)
        if '1' in report:
            mlflow.log_metric("class_1_precision", report['1']['precision'])
            mlflow.log_metric("class_1_recall", report['1']['recall'])
            mlflow.log_metric("class_1_f1", report['1']['f1-score'])
        # Log overall accuracy
        if 'accuracy' in report:
            mlflow.log_metric("accuracy", report['accuracy'])
    
    logging.info(f"Model evaluation metrics: {metrics}")
    return y_pred, metrics



def save_model(model: XGBClassifier, filename: str) -> None:
    """Save the model to a file."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, filename)
    logging.info(f"Saving model to {model_path}.")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Log model to MLflow
    mlflow.xgboost.log_model(
        model, 
        "model",
        registered_model_name="customer-churn-xgboost"  # Optional: auto-register
    )
    
    mlflow.log_artifact(model_path, artifact_path="local_model")
    
    logging.info("Model saved locally and logged to MLflow.")



def main():
    # Paths for processed train/test data
    X_train_path = os.path.join(DATA_DIR, "X_train.pkl")
    y_train_path = os.path.join(DATA_DIR, "y_train.pkl")
    X_test_path = os.path.join(DATA_DIR, "X_test.pkl")
    y_test_path = os.path.join(DATA_DIR, "y_test.pkl")

    # Load train/test data
    logging.info("Loading processed training and testing data.")
    with open(X_train_path, "rb") as f:
        X_train = pickle.load(f)
    with open(y_train_path, "rb") as f:
        y_train = pickle.load(f)
    with open(X_test_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_test_path, "rb") as f:
        y_test = pickle.load(f)

    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Start MLflow run
    with mlflow.start_run():
        # Log dataset info
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Train model
        model = train_xgb_model(X_train, y_train)

        # Evaluate model
        y_pred, metrics = evaluate_model(model, X_test, y_test)

        # Save trained model
        save_model(model, MODEL_FILENAME)

        logging.info("Training pipeline completed successfully.")
        logging.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()