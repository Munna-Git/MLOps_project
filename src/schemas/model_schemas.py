# src/schemas/model_schemas.py

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, Dict, Any
from datetime import datetime


class XGBoostHyperparameters(BaseModel):
    """Schema for XGBoost hyperparameters with validation"""
    
    n_estimators: int = Field(
        default=150,
        ge=10,
        le=1000,
        description="Number of boosting rounds (10-1000)"
    )
    
    max_depth: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum tree depth (1-20)"
    )
    
    learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Step size shrinkage (0.001-1.0)",
        alias="eta"
    )
    
    min_child_weight: float = Field(
        default=1.0,
        ge=0,
        le=10.0,
        description="Minimum sum of instance weight in child (0-10)"
    )
    
    subsample: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Subsample ratio of training instances (0.5-1.0)"
    )
    
    colsample_bytree: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Subsample ratio of columns (0.5-1.0)"
    )
    
    scale_pos_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Balancing of positive and negative weights (0.1-10)"
    )
    
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    objective: Literal["binary:logistic", "binary:hinge"] = Field(
        default="binary:logistic",
        description="Learning objective"
    )
    
    eval_metric: Literal["logloss", "error", "auc"] = Field(
        default="logloss",
        description="Evaluation metric"
    )
    
    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Warn if learning rate is too high or too low"""
        if v > 0.3:
            print(f"Warning: learning_rate={v} is quite high. Consider using 0.01-0.3")
        elif v < 0.01:
            print(f"Warning: learning_rate={v} is quite low. Training may be slow")
        return v
    
    @model_validator(mode='after')
    def validate_tree_parameters(self):
        """Validate tree parameter combinations"""
        if self.max_depth > 10 and self.n_estimators > 500:
            print("Warning: High max_depth with many estimators may cause overfitting")
        return self
    
    def to_xgboost_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dictionary"""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "objective": self.objective,
            "eval_metric": self.eval_metric
        }


class TrainTestSplitConfig(BaseModel):
    """Schema for train-test split configuration"""
    
    test_size: float = Field(
        default=0.1,
        ge=0.05,
        le=0.5,
        description="Proportion of dataset for test set (0.05-0.5)"
    )
    
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    stratify: bool = Field(
        default=True,
        description="Whether to stratify split based on target variable"
    )
    
    @field_validator("test_size")
    @classmethod
    def validate_test_size_proportion(cls, v: float) -> float:
        """Ensure test_size is reasonable"""
        if v < 0.1:
            print(f"Warning: test_size={v} is quite small. Consider using at least 0.1")
        elif v > 0.3:
            print(f"Warning: test_size={v} is quite large. May reduce training data")
        return v


class ModelMetrics(BaseModel):
    """Schema for model evaluation metrics"""
    
    f1_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="F1 score (harmonic mean of precision and recall)"
    )
    
    f2_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="F2 score (weighted harmonic mean favoring recall)"
    )
    
    precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Precision (positive predictive value)"
    )
    
    recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Recall (sensitivity, true positive rate)"
    )
    
    accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall accuracy"
    )
    
    roc_auc: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Area under ROC curve"
    )
    
    confusion_matrix: Optional[list[list[int]]] = Field(
        None,
        description="2x2 confusion matrix [[TN, FP], [FN, TP]]"
    )
    
    classification_report: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed classification report"
    )
    
    @model_validator(mode='after')
    def validate_metrics_consistency(self):
        """Check if metrics are consistent"""
        if self.precision == 0 and self.recall == 0:
            if self.f1_score != 0:
                raise ValueError("F1 score should be 0 when precision and recall are 0")
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return self.model_dump(exclude_none=True)
