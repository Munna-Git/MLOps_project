from abc import ABC, abstractmethod
import pandas as pd
from pandas import DataFrame
import warnings
import logging
import os

# Suppress warnings
warnings.filterwarnings('ignore')


class DataPipeline(ABC):
    def __init__(self, input_filepath: str, output_filepath: str):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def load_data(self) -> DataFrame:
        """Loads data from the input filepath."""
        return pd.read_csv(self.input_filepath)
    
    def load_cleaned_data(self) -> DataFrame:
        """Loads data from the output filepath (cleaned data)."""
        return pd.read_csv(self.output_filepath)

    def save_data(self, df: DataFrame) -> None:
        """Saves the processed data to the output filepath."""
        df.to_csv(self.output_filepath, index=False)

    def drop_columns(self, df: DataFrame) -> DataFrame:
        """Drop unnecessary columns (default behavior)."""
        columns_to_drop = ["RowNumber", "Surname", "Complain"]
        return df.drop(columns=columns_to_drop, errors='ignore')

    def rename_columns(self, df: DataFrame) -> DataFrame:
        """Rename columns for consistency."""
        # Strip extra spaces/tabs from column names
        df.columns = df.columns.str.strip()

        df.rename(columns={
            'Satisfaction Score': 'SatisfactionScore',
            'Card Type': 'CardType',
            'Point Earned': 'PointEarned'
        }, inplace=True)
        return df

    def encode_categorical_features(self, df: DataFrame) -> DataFrame:
        """Encode categorical features."""
        df = pd.get_dummies(df, columns=['Geography'], prefix='Geography')
        df = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
        card_mapping = {'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3}
        df['CardType'] = df['CardType'].map(card_mapping).fillna(-1)
        return df

    @abstractmethod
    def preprocess(self) -> None:
        """Abstract method for preprocessing data."""
        pass


class TrainingPipeline(DataPipeline):
    def drop_columns(self, df: DataFrame) -> DataFrame:
        """Drop unnecessary columns for training (remove ID fields)."""
        columns_to_drop = ["RowNumber", "Surname", "Complain", "CustomerId"]
        return df.drop(columns=columns_to_drop, errors='ignore')

    def preprocess(self) -> None:
        """Preprocess data for training."""
        df = self.load_data()
        df = self.drop_columns(df)
        df = self.rename_columns(df)
        df = self.encode_categorical_features(df)
        self.save_data(df)
        print(f"Training data saved to {self.output_filepath}")


class InferencePipeline:
    def __init__(self, input_filepath: str, output_filepath: str):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def preprocess(self) -> None:
        """
        Load raw inference data, clean it, and save to output filepath.
        Keeps CustomerId for traceability.
        """
        logging.info(f"Loading inference data from {self.input_filepath}")
        df = pd.read_csv(self.input_filepath)

        # Example cleaning (adjust this part as per your training pipeline steps)
        # ✅ Keep CustomerId
        if "CustomerId" in df.columns:
            customer_ids = df["CustomerId"]
        else:
            logging.warning("CustomerId column not found in raw inference data.")
            customer_ids = None

        # Drop irrelevant columns (but NOT CustomerId)
        drop_cols = ["Surname"]  # example
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Convert categorical features (example: Geography, Gender)
        if "Geography" in df.columns:
            df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

        # Reattach CustomerId at the start
        if customer_ids is not None:
            df.insert(0, "CustomerId", customer_ids)

        # Save cleaned inference data
        os.makedirs(os.path.dirname(self.output_filepath), exist_ok=True)
        df.to_csv(self.output_filepath, index=False)
        logging.info(f"Inference data saved to {self.output_filepath}")