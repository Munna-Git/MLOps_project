# src/config/settings.py

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """
    Application settings with validation using Pydantic.
    Automatically loads from .env file and validates all configurations.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )
    
    # ========== File Paths ==========
    raw_data_filepath: Path = Field(
        ...,
        description="Path to raw customer churn data CSV file"
    )
    
    cleaned_data_filepath: Path = Field(
        ...,
        description="Path to save cleaned training data"
    )
    
    processed_data_dir: Path = Field(
        ...,
        description="Directory for processed train/test data"
    )
    
    model_output_dir: Path = Field(
        ...,
        description="Directory to save trained models"
    )
    
    model_filename: str = Field(
        default="xgboost_model.pkl",
        description="Filename for the trained model"
    )
    
    test_data_path: Path = Field(
        ...,
        description="Path to inference/test data CSV file"
    )
    
    inference_cleaned_path: Path = Field(
        ...,
        description="Path to save cleaned inference data"
    )
    
    model_path: Path = Field(
        ...,
        description="Full path to the trained model file"
    )
    
    # ========== Training Parameters ==========
    test_size: float = Field(
        default=0.1,
        ge=0.05,
        le=0.5,
        description="Train-test split ratio (must be between 0.05 and 0.5)"
    )
    
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    # ========== API Configuration ==========
    api_host: str = Field(
        default="0.0.0.0",
        description="Flask API host"
    )
    
    api_port: int = Field(
        default=5000,
        ge=1024,
        le=65535,
        description="Flask API port number"
    )
    
    api_debug: bool = Field(
        default=True,
        description="Enable Flask debug mode"
    )
    
    # ========== Logging Configuration ==========
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # ========== Validators ==========
    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Ensure test_size is a valid proportion"""
        if not 0.05 <= v <= 0.5:
            raise ValueError("test_size must be between 0.05 and 0.5")
        return v
    
    @field_validator(
        "raw_data_filepath",
        "test_data_path",
        mode="after"
    )
    @classmethod
    def validate_input_files_exist(cls, v: Path) -> Path:
        """Validate that input data files exist"""
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v
    
    @field_validator(
        "processed_data_dir",
        "model_output_dir",
        mode="after"
    )
    @classmethod
    def validate_or_create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("model_filename")
    @classmethod
    def validate_model_filename(cls, v: str) -> str:
        """Ensure model filename has correct extension"""
        if not v.endswith(".pkl"):
            raise ValueError("model_filename must end with .pkl")
        return v
    
    # ========== Computed Properties ==========
    @property
    def full_model_path(self) -> Path:
        """Get the complete model path"""
        return self.model_output_dir / self.model_filename
    
    def get_processed_file_path(self, data_type: str, file_type: str = "pkl") -> Path:
        """
        Get path for processed data files.
        
        Args:
            data_type: 'train' or 'test'
            file_type: 'pkl' or 'csv'
        
        Returns:
            Path object for the file
        """
        return self.processed_data_dir / f"X_{data_type}.{file_type}"


# Singleton instance
settings = Settings()


# Convenience function for getting settings
def get_settings() -> Settings:
    """Get application settings instance"""
    return settings