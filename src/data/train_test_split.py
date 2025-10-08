# src/data/train_test_split.py

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from the specified file path."""
    return pd.read_csv(filepath)


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.1, random_state: int = 42) -> tuple:
    """
    Splits the dataset into training and testing sets with stratification on the target variable.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - target (str): The name of the target column.
    - test_size (float): The proportion of the data to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    - tuple: X_train, X_test, y_train, y_test
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def save_data(X: pd.DataFrame, y: pd.Series, data_type: str, stage: str, base_directory: str = "data"):
    """
    Saves features and target variables to the specified stage in both CSV and Pickle format.

    Parameters:
    - X (pd.DataFrame): Features to save.
    - y (pd.Series): Target variable to save.
    - data_type (str): The type of data (e.g., 'train', 'test').
    - stage (str): The processing stage (e.g., 'processed').
    - base_directory (str): Base directory for saving data.
    """
    
    directory = os.path.join(base_directory, stage)
    os.makedirs(directory, exist_ok=True)

    # Save Pickle
    with open(f"{directory}/X_{data_type}.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(f"{directory}/y_{data_type}.pkl", "wb") as f:
        pickle.dump(y, f)

    # Save CSV
    X.to_csv(f"{directory}/X_{data_type}.csv", index=False)
    y.to_csv(f"{directory}/y_{data_type}.csv", index=False)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_filepath = os.path.join(base_dir, "../../data/cleaned/CustomerChurnCleaned.csv")
    target_column = "Exited"

    print("Loading the data...")
    df = load_data(input_filepath)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    print("Saving split data...")
    save_data(X_train, y_train, "train", "processed")
    save_data(X_test, y_test, "test", "processed")

    print("Train-test split completed. Data saved in 'data/processed/' directory.")


if __name__ == "__main__":
    main()
