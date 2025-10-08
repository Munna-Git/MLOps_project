# src/models/train_model.py

import os
import pickle
import logging
import numpy as np
from xgboost import XGBClassifier
from src.models.evaluate_model import get_model_metrics

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
    logging.info(f"Model evaluation metrics: {metrics}")
    return y_pred, metrics


def save_model(model: XGBClassifier, filename: str) -> None:
    """Save the model to a file."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, filename)
    logging.info(f"Saving model to {model_path}.")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


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

    # Train model
    model = train_xgb_model(X_train, y_train)

    # Evaluate model
    y_pred, metrics = evaluate_model(model, X_test, y_test)

    # Save trained model
    save_model(model, MODEL_FILENAME)

    logging.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
