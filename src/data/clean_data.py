from abc import ABC, abstractmethod
import pandas as pd
from pandas import DataFrame
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# Base Class for Data Pipelines
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
        return df.drop(columns=columns_to_drop)

    def rename_columns(self, df: DataFrame) -> DataFrame:
        """Rename columns for consistency."""
        df.rename(columns={
            'Satisfaction Score\t': 'SatisfactionScore',
            'Card Type': 'CardType',
            'Point Earned': 'PointEarned'
        }, inplace=True)
        return df

    def encode_categorical_features(self, df: DataFrame) -> DataFrame:
        """Encode categorical features."""
        df = pd.get_dummies(df, columns=['Geography'], prefix='Geography')
        df = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
        card_mapping = {'SILVER': 0, 'GOLD': 1, 'PLATINUM': 2, 'DIAMOND': 3}
        df['CardType'] = df['CardType'].map(card_mapping)
        return df

    @abstractmethod
    def preprocess(self) -> None:
        """Abstract method for preprocessing data."""
        pass


# Training Pipeline Class
class TrainingPipeline(DataPipeline):
    def drop_columns(self, df: DataFrame) -> DataFrame:
        """Drop unnecessary columns for training."""
        columns_to_drop = ["RowNumber", "Surname", "Complain", "CustomerId"]
        return df.drop(columns=columns_to_drop)

    def preprocess(self) -> None:
        """Preprocess data for training."""
        df = self.load_data()
        df = self.drop_columns(df)
        df = self.rename_columns(df)
        df = self.encode_categorical_features(df)
        self.save_data(df)
        print(f"Training data saved to {self.output_filepath}")


# Inference Pipeline Class
class InferencePipeline(DataPipeline):
    def preprocess(self) -> None:
        """Preprocess data for inference (retain `CustomerId`)."""
        df = self.load_data()
        df = self.drop_columns(df)  # Uses default behavior, retaining `Surname`
        df = self.rename_columns(df)
        df = self.encode_categorical_features(df)
        self.save_data(df)
        print(f"Inference data saved to {self.output_filepath}")