import os
import pickle
import logging
import numpy as np
from xgboost import XGBClassifier
from src.models.evaluate_model import get_model_metrics

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

#Constants
DATA_DIR: str = "data/processed"  # Input training data
MODEL_OUTPUT_DIR: str = "../models"   # Output directory for model artifacts
MODEL_FILENAME: str = "xgboost_model.pkl"
SHAP_PLOT_FILENAME: str = "shap_summary_plot.png"


# XGBoost parameters
XGB_PARAMS: dict = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'n_estimators': 150,
    'scale_pos_weight': 1,
    'subsample': 1.0
}


def train_xgb_model(X_train : np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train the XGBoost model with the provided parameters."""
    logging.info("Training XGBoost model. ")
    model = XGBClassifier(
        **XGB_PARAMS,
        objective = 'binary:logistic',
        eval_metric = 'logloss',
    )
    model.fit(X_train, y_train, verbose = True)
    logging.info("XGBoost model training completed")

    return model


def evaluate_model(model : XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) ->np.ndarray:
    """Evaluate the model and save the metrics and the model."""
    logging.info("Evaluating model.")
    y_pred = model.predict(X_test)

    metrics = get_model_metrics(y_test, y_pred)
    logging.info(f"Model evaluation metrics: {metrics}")

    return y_pred

def save_model(model: XGBClassifier, filename: str) -> None:
    """Save the model to a file."""
    logging.info(f"Saving model to {filename}.")
    with open(os.path.join(MODEL_OUTPUT_DIR, filename), 'wb') as f:
        pickle.dump(model, f)
