# src/utils/data_validator.py

import yaml
import pandas as pd
from cerberus import Validator
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, schema_path="config/data_schema.yaml"):
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
        self.validator = Validator(self.schema)
        # Allow unknown fields (for columns not in schema)
        self.validator.allow_unknown = True
    
    def validate_row(self, row_dict):
        """Validate single row"""
        if self.validator.validate(row_dict):
            return True, None
        else:
            return False, self.validator.errors
    
    def validate_dataframe(self, df, sample_size=None):
        """Validate DataFrame, return valid rows and errors"""
        errors = []
        
        # Validate subset if sample_size provided
        rows_to_check = df.head(sample_size) if sample_size else df
        
        for idx, row in rows_to_check.iterrows():
            is_valid, error = self.validate_row(row.to_dict())
            if not is_valid:
                errors.append({"row": idx, "errors": error})
                if len(errors) >= 10:  # Limit errors
                    break
        
        return len(errors) == 0, errors


# Quick usage function
def validate_csv(csv_path, schema_path="config/data_schema.yaml"):
    """Validate CSV file"""
    df = pd.read_csv(csv_path)
    validator = DataValidator(schema_path)
    is_valid, errors = validator.validate_dataframe(df, sample_size=100)
    
    if is_valid:
        logger.info(f"✓ Data validation passed for {csv_path}")
    else:
        logger.error("✗ Data validation failed:")
        for error in errors[:5]:
            logger.error(f"Row {error['row']}: {error['errors']}")
    
    return is_valid, errors