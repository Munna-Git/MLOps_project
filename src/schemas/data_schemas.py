# src/schemas/data_schemas.py

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
import pandas as pd


class CustomerChurnRawData(BaseModel):
    """Schema for raw customer churn data input validation"""
    
    model_config = {"arbitrary_types_allowed": True}
    
    RowNumber: int = Field(ge=1, description="Row number in dataset")
    CustomerId: int = Field(ge=10000000, le=99999999, description="Unique 8-digit customer ID")
    Surname: str = Field(min_length=2, max_length=50, description="Customer surname")
    CreditScore: int = Field(ge=300, le=850, description="Credit score (300-850)")
    Geography: Literal["France", "Germany", "Spain"] = Field(description="Customer country")
    Gender: Literal["Male", "Female"] = Field(description="Customer gender")
    Age: int = Field(ge=18, le=100, description="Customer age")
    Tenure: int = Field(ge=0, le=10, description="Years with bank (0-10)")
    Balance: float = Field(ge=0, description="Account balance")
    NumOfProducts: int = Field(ge=1, le=4, description="Number of products (1-4)")
    HasCrCard: Literal[0, 1] = Field(description="Has credit card (0/1)")
    IsActiveMember: Literal[0, 1] = Field(description="Is active member (0/1)")
    EstimatedSalary: float = Field(ge=0, le=250000, description="Estimated salary")
    Exited: Literal[0, 1] = Field(description="Customer churned (0/1)")
    Complain: Literal[0, 1] = Field(description="Customer complained (0/1)")
    SatisfactionScore: int = Field(ge=1, le=5, alias="Satisfaction Score", description="Satisfaction (1-5)")
    CardType: Literal["DIAMOND", "GOLD", "PLATINUM", "SILVER"] = Field(alias="Card Type", description="Card type")
    PointEarned: int = Field(ge=0, alias="Point Earned", description="Loyalty points")
    
    @field_validator("Surname")
    @classmethod
    def validate_surname(cls, v: str) -> str:
        """Ensure surname contains only letters"""
        if not v.replace(" ", "").replace("-", "").isalpha():
            raise ValueError("Surname must contain only letters, spaces, or hyphens")
        return v.strip()


class CustomerChurnCleanedData(BaseModel):
    """Schema for cleaned/processed data after feature engineering"""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Core features
    CreditScore: int = Field(ge=300, le=850)
    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0, le=10)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(ge=0, le=250000)
    SatisfactionScore: int = Field(ge=1, le=5)
    CardType: int = Field(ge=-1, le=3, description="Encoded card type: -1=unknown, 0=SILVER, 1=GOLD, 2=PLATINUM, 3=DIAMOND")
    PointEarned: int = Field(ge=0)
    
    # One-hot encoded geography
    Geography_France: Literal[0, 1] = Field(default=0)
    Geography_Germany: Literal[0, 1] = Field(default=0)
    Geography_Spain: Literal[0, 1] = Field(default=0)
    
    # One-hot encoded gender
    Gender_Female: Literal[0, 1] = Field(default=0)
    Gender_Male: Literal[0, 1] = Field(default=0)
    
    # Target (optional for inference data)
    Exited: Optional[Literal[0, 1]] = Field(default=None)
    
    @model_validator(mode='after')
    def validate_one_hot_encodings(self):
        """Ensure one-hot encodings are mutually exclusive"""
        # Check Geography
        geo_sum = self.Geography_France + self.Geography_Germany + self.Geography_Spain
        if geo_sum != 1:
            raise ValueError("Exactly one Geography field must be 1")
        
        # Check Gender
        gender_sum = self.Gender_Female + self.Gender_Male
        if gender_sum != 1:
            raise ValueError("Exactly one Gender field must be 1")
        
        return self


class InferenceInputData(CustomerChurnCleanedData):
    """Schema for inference input (includes CustomerId, no target)"""
    
    CustomerId: int = Field(ge=10000000, le=99999999, description="Unique 8-digit customer ID")
    Exited: Optional[Literal[0, 1]] = None  # Not required for inference


class DataFrameValidator:
    """Utility class to validate pandas DataFrames against Pydantic schemas"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, schema: type[BaseModel], 
                          sample_size: Optional[int] = None) -> tuple[bool, list[str]]:
        """
        Validate a pandas DataFrame against a Pydantic schema.
        
        Args:
            df: DataFrame to validate
            schema: Pydantic model class to validate against
            sample_size: If provided, only validate first N rows (for large datasets)
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate column names
        expected_fields = set(schema.model_fields.keys())
        actual_columns = set(df.columns)
        
        # Handle aliases
        field_aliases = {
            field_info.alias or field_name: field_name 
            for field_name, field_info in schema.model_fields.items() 
            if field_info.alias
        }
        
        # Check for missing columns
        missing_cols = expected_fields - actual_columns
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for extra columns (warning, not error)
        extra_cols = actual_columns - expected_fields
        if extra_cols:
            errors.append(f"Extra columns (will be ignored): {extra_cols}")
        
        # Validate rows
        rows_to_check = df.head(sample_size) if sample_size else df
        
        for idx, row in rows_to_check.iterrows():
            try:
                # Convert row to dict and validate
                row_dict = row.to_dict()
                schema.model_validate(row_dict)
            except Exception as e:
                errors.append(f"Row {idx} validation error: {str(e)}")
                if len(errors) >= 10:  # Limit error reporting
                    errors.append("... (more errors omitted)")
                    break
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_and_raise(df: pd.DataFrame, schema: type[BaseModel], 
                          sample_size: Optional[int] = None) -> None:
        """
        Validate DataFrame and raise exception if invalid.
        
        Args:
            df: DataFrame to validate
            schema: Pydantic model class to validate against
            sample_size: If provided, only validate first N rows
        
        Raises:
            ValueError: If validation fails
        """
        is_valid, errors = DataFrameValidator.validate_dataframe(df, schema, sample_size)
        if not is_valid:
            error_msg = "\n".join(errors)
            raise ValueError(f"DataFrame validation failed:\n{error_msg}")