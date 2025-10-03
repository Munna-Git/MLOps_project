# src/utils/shap_utils.py


import pandas as pd
import shap
import matplotlib.pyplot as plt


def convert_boolean_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Convert all boolean columns in X_train and X_test to integers.

    Parameters:
    - X_train (pd.DataFrame): Training data.
    - X_test (pd.DataFrame): Test data.

    Returns:
    - tuple: X_train and X_test with boolean columns converted to integers.
    """
    bool_columns_train = X_train.select_dtypes('bool').columns
    bool_columns_test = X_test.select_dtypes('bool').columns

    if not bool_columns_train.empty:
        X_train = X_train.astype({col: 'int64' for col in bool_columns_train})
    if not bool_columns_test.empty:
        X_test = X_test.astype({col: 'int64' for col in bool_columns_test})

    return X_train, X_test


def save_shap_summary_plot(model: object, X_test: pd.DataFrame, output_path: str = "shap_summary_plot.png") -> None:
    """
    Generate and save a SHAP summary plot for the given model and validation data.

    Parameters:
    - model (object): The trained model for which SHAP values are calculated.
    - X_test (pd.DataFrame): The test features used to compute SHAP values.
    - output_path (str): The file path to save the SHAP summary plot.

    Returns:
    - None: Saves the summary plot as a PNG file.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_path)
    plt.close()