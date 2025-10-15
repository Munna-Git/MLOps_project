# src/schemas/api_schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Literal, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    
    customer_id: int = Field(
        ...,
        ge=10000000,
        le=99999999,
        description="Unique 8-digit customer ID",
        examples=[15634602]
    )
    
    @field_validator("customer_id")
    @classmethod
    def validate_customer_id_length(cls, v: int) -> int:
        """Ensure customer ID is exactly 8 digits"""
        if not (10000000 <= v <= 99999999):
            raise ValueError("customer_id must be an 8-digit number")
        return v


class PredictionResponse(BaseModel):
    """Schema for simple prediction response"""
    
    customer_id: int = Field(
        ...,
        description="Customer ID that was queried",
        examples=[15634602]
    )
    
    prediction: Literal[0, 1] = Field(
        ...,
        description="Churn prediction: 0 = will not churn, 1 = will churn",
        examples=[1]
    )
    
    prediction_label: str = Field(
        ...,
        description="Human-readable prediction label",
        examples=["Will Churn"]
    )
    
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0-1)",
        examples=[0.87]
    )
    
    message: Optional[str] = Field(
        None,
        description="Additional information or instructions",
        examples=["Use /score/<customer_id> for SHAP explanations"]
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Prediction timestamp"
    )


class SHAPExplanation(BaseModel):
    """Schema for SHAP feature importance values"""
    
    feature_name: str = Field(..., description="Name of the feature")
    shap_value: float = Field(..., description="SHAP value indicating feature impact")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_name": "Age",
                "shap_value": 0.234
            }
        }


class PredictionWithSHAPResponse(BaseModel):
    """Schema for prediction response with SHAP explanations"""
    
    customer_id: int = Field(
        ...,
        description="Customer ID that was queried"
    )
    
    prediction: Literal[0, 1] = Field(
        ...,
        description="Churn prediction: 0 = will not churn, 1 = will churn"
    )
    
    prediction_label: str = Field(
        ...,
        description="Human-readable prediction label"
    )
    
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model confidence score"
    )
    
    shap_values: Dict[str, float] = Field(
        ...,
        description="SHAP values for each feature explaining the prediction"
    )
    
    top_positive_features: list[SHAPExplanation] = Field(
        default_factory=list,
        description="Top features increasing churn probability"
    )
    
    top_negative_features: list[SHAPExplanation] = Field(
        default_factory=list,
        description="Top features decreasing churn probability"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Prediction timestamp"
    )


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    
    error: str = Field(
        ...,
        description="Error type or message",
        examples=["Customer not found"]
    )
    
    details: Optional[Dict[str, any]] = Field(
        None,
        description="Additional error details"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )


class HealthCheckResponse(BaseModel):
    """Schema for health check endpoint"""
    
    status: Literal["running", "error"] = Field(
        ...,
        description="Service status",
        examples=["running"]
    )
    
    message: str = Field(
        ...,
        description="Service description",
        examples=["Customer Churn Prediction API"]
    )
    
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded"
    )
    
    endpoints: Dict[str, str] = Field(
        ...,
        description="Available API endpoints"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Health check timestamp"
    )


# Utility functions for creating responses
def create_prediction_response(
    customer_id: int,
    prediction: int,
    confidence: Optional[float] = None,
    message: Optional[str] = None
) -> PredictionResponse:
    """Helper function to create a validated prediction response"""
    
    prediction_label = "Will Churn" if prediction == 1 else "Will Not Churn"
    
    return PredictionResponse(
        customer_id=customer_id,
        prediction=prediction,
        prediction_label=prediction_label,
        confidence=confidence,
        message=message
    )


def create_shap_response(
    customer_id: int,
    prediction: int,
    shap_values: Dict[str, float],
    confidence: Optional[float] = None,
    top_n: int = 5
) -> PredictionWithSHAPResponse:
    """Helper function to create a validated SHAP response"""
    
    prediction_label = "Will Churn" if prediction == 1 else "Will Not Churn"
    
    # Sort features by SHAP value magnitude
    sorted_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Get top positive and negative contributors
    top_positive = [
        SHAPExplanation(feature_name=name, shap_value=value)
        for name, value in sorted_features if value > 0
    ][:top_n]
    
    top_negative = [
        SHAPExplanation(feature_name=name, shap_value=value)
        for name, value in sorted_features if value < 0
    ][:top_n]
    
    return PredictionWithSHAPResponse(
        customer_id=customer_id,
        prediction=prediction,
        prediction_label=prediction_label,
        confidence=confidence,
        shap_values=shap_values,
        top_positive_features=top_positive,
        top_negative_features=top_negative
    )


def create_error_response(
    error: str,
    details: Optional[Dict] = None
) -> ErrorResponse:
    """Helper function to create a validated error response"""
    
    return ErrorResponse(
        error=error,
        details=details
    )