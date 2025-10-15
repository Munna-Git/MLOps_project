# app/train_entrypoint.py - REFACTORED WITH PYDANTIC

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.settings import get_settings
from src.schemas.model_schemas import (
    ModelTrainingConfig,
    ModelMetadata,
    ModelMetrics,
    XGBoostHyperparameters
)
from src.schemas.data_schemas import DataFrameValidator, CustomerChurnRawData
from src.data.clean_data import TrainingPipeline
from src.data.train_test_split import split_data, save_data
from src.models.train_model import train_xgb_model, evaluate_model, save_model
from pydantic import ValidationError

# Get validated settings
settings = get_settings()

# Configure logging with settings
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_raw_data() -> bool:
    """Validate raw data against schema before processing"""
    logger.info("Validating raw data schema...")
    try:
        import pandas as pd
        df = pd.read_csv(settings.raw_data_filepath)
        
        # Validate first 100 rows as sample
        is_valid, errors = DataFrameValidator.validate_dataframe(
            df, 
            CustomerChurnRawData,
            sample_size=100
        )
        
        if not is_valid:
            logger.error("Raw data validation failed:")
            for error in errors[:10]:  # Show first 10 errors
                logger.error(f"  - {error}")
            return False
        
        logger.info(f"✓ Raw data validation passed ({len(df)} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        return False


def run_data_cleaning_pipeline() -> bool:
    """Execute data cleaning with error handling"""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Cleaning Pipeline")
    logger.info("=" * 60)
    
    try:
        training_pipeline = TrainingPipeline(
            str(settings.raw_data_filepath),
            str(settings.cleaned_data_filepath)
        )
        training_pipeline.preprocess()
        logger.info(f"✓ Data cleaned and saved to {settings.cleaned_data_filepath}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error during data cleaning: {e}")
        return False


def run_train_test_split() -> tuple:
    """Execute train-test split with validated config"""
    logger.info("=" * 60)
    logger.info("STEP 2: Train-Test Split")
    logger.info("=" * 60)
    
    try:
        # Load cleaned data
        training_pipeline = TrainingPipeline(
            str(settings.raw_data_filepath),
            str(settings.cleaned_data_filepath)
        )
        df = training_pipeline.load_cleaned_data()
        logger.info(f"Loaded cleaned data: {df.shape}")
        
        # Create split configuration
        from src.schemas.model_schemas import TrainTestSplitConfig
        split_config = TrainTestSplitConfig(
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=True
        )
        
        # Perform split
        X_train, X_test, y_train, y_test = split_data(
            df,
            target="Exited",
            test_size=split_config.test_size,
            random_state=split_config.random_state
        )
        
        # Save split data
        save_data(X_train, y_train, "train", str(settings.processed_data_dir))
        save_data(X_test, y_test, "test", str(settings.processed_data_dir))
        
        logger.info(f"✓ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"✓ Split data saved to {settings.processed_data_dir}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"✗ Error during train-test split: {e}")
        raise


def run_model_training(X_train, y_train) -> tuple:
    """Train model with validated hyperparameters"""
    logger.info("=" * 60)
    logger.info("STEP 3: Model Training")
    logger.info("=" * 60)
    
    try:
        # Create and validate hyperparameters
        hyperparameters = XGBoostHyperparameters(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            scale_pos_weight=1.0,
            random_state=settings.random_state
        )
        
        logger.info("Training with hyperparameters:")
        for param, value in hyperparameters.model_dump().items():
            logger.info(f"  - {param}: {value}")
        
        # Train model with timing
        start_time = time.time()
        model = train_xgb_model(X_train, y_train)
        training_duration = time.time() - start_time
        
        logger.info(f"✓ Model training completed in {training_duration:.2f}s")
        
        return model, hyperparameters, training_duration
        
    except ValidationError as e:
        logger.error(f"✗ Hyperparameter validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error during model training: {e}")
        raise


def run_model_evaluation(model, X_test, y_test) -> ModelMetrics:
    """Evaluate model and return validated metrics"""
    logger.info("=" * 60)
    logger.info("STEP 4: Model Evaluation")
    logger.info("=" * 60)
    
    try:
        y_pred, metrics_dict = evaluate_model(model, X_test, y_test)
        
        # Create validated metrics object
        metrics = ModelMetrics(
            f1_score=metrics_dict['f1_score'],
            f2_score=metrics_dict['f2_score'],
            precision=metrics_dict['precision'],
            recall=metrics_dict['recall'],
            confusion_matrix=metrics_dict['confusion_matrix'].tolist(),
            classification_report=metrics_dict['classification_report']
        )
        
        logger.info("Model Performance:")
        logger.info(f"  - F1 Score: {metrics.f1_score:.4f}")
        logger.info(f"  - F2 Score: {metrics.f2_score:.4f}")
        logger.info(f"  - Precision: {metrics.precision:.4f}")
        logger.info(f"  - Recall: {metrics.recall:.4f}")
        
        return metrics
        
    except ValidationError as e:
        logger.error(f"✗ Metrics validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Error during model evaluation: {e}")
        raise


def save_model_with_metadata(model, X_train, X_test, metrics, 
                             hyperparameters, training_duration):
    """Save model along with metadata"""
    logger.info("=" * 60)
    logger.info("STEP 5: Saving Model and Metadata")
    logger.info("=" * 60)
    
    try:
        # Create metadata
        metadata = ModelMetadata(
            model_name="XGBoost Churn Classifier",
            model_version="1.0.0",
            trained_at=datetime.now(),
            training_samples=len(X_train),
            test_samples=len(X_test),
            feature_names=X_train.columns.tolist(),
            metrics=metrics,
            hyperparameters=hyperparameters,
            training_duration_seconds=training_duration
        )
        
        # Save model
        save_model(model, settings.model_filename)
        logger.info(f"✓ Model saved to {settings.full_model_path}")
        
        # Save metadata
        metadata_path = settings.model_output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            f.write(metadata.model_dump_json(indent=2))
        logger.info(f"✓ Metadata saved to {metadata_path}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("MODEL TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(metadata.summary())
        
        return metadata
        
    except Exception as e:
        logger.error(f"✗ Error saving model: {e}")
        raise


def main() -> None:
    """Execute the complete training pipeline with Pydantic validation"""
    
    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    try:
        # Validate settings
        logger.info("Configuration:")
        logger.info(f"  - Raw Data: {settings.raw_data_filepath}")
        logger.info(f"  - Cleaned Data: {settings.cleaned_data_filepath}")
        logger.info(f"  - Processed Data Dir: {settings.processed_data_dir}")
        logger.info(f"  - Model Output: {settings.full_model_path}")
        logger.info(f"  - Test Size: {settings.test_size}")
        logger.info(f"  - Random State: {settings.random_state}")
        logger.info("")
        
        # Step 0: Validate raw data
        if not validate_raw_data():
            logger.error("Pipeline aborted due to data validation errors")
            return
        
        # Step 1: Clean data
        if not run_data_cleaning_pipeline():
            logger.error("Pipeline aborted due to cleaning errors")
            return
        
        # Step 2: Split data
        X_train, X_test, y_train, y_test = run_train_test_split()
        
        # Step 3: Train model
        model, hyperparameters, training_duration = run_model_training(X_train, y_train)
        
        # Step 4: Evaluate model
        metrics = run_model_evaluation(model, X_test, y_test)
        
        # Step 5: Save model and metadata
        save_model_with_metadata(
            model, X_train, X_test, metrics,
            hyperparameters, training_duration
        )
        
        logger.info("=" * 60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except ValidationError as e:
        logger.error("=" * 60)
        logger.error("VALIDATION ERROR")
        logger.error("=" * 60)
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()