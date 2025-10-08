# src/tests/feature_pipeline_test.py

import unittest
import sys
import os
import pandas as pd
import tempfile

# --- Make sure Python can find your src folder ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.clean_data import InferencePipeline


class TestInferencePipeline(unittest.TestCase):
    """Unit tests for the InferencePipeline data preprocessing."""

    def setUp(self):
        """Prepare temporary input and output CSV files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "inference_input.csv")
        self.output_file = os.path.join(self.temp_dir.name, "inference_output.csv")

        # Create mock raw inference data
        data = {
            "CustomerId": [15619304, 15584014],
            "Surname": ["Smith", "Johnson"],
            "CreditScore": [619, 608],
            "Geography": ["France", "Spain"],
            "Gender": ["Male", "Female"],
            "Age": [42, 41],
            "Tenure": [2, 1],
            "Balance": [0.0, 83807.86],
            "NumOfProducts": [1, 1],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 1],
            "EstimatedSalary": [101348.88, 112542.58],
            "CardType": ["SILVER", "GOLD"]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.input_file, index=False)

        # Initialize the pipeline
        self.pipeline = InferencePipeline(
            input_filepath=self.input_file,
            output_filepath=self.output_file
        )

    def test_preprocess_creates_cleaned_file(self):
        """Test whether preprocess() creates a cleaned output CSV."""
        self.pipeline.preprocess()
        self.assertTrue(os.path.exists(self.output_file))

    def test_output_has_expected_columns(self):
        """Test that the output CSV has encoded and cleaned columns."""
        self.pipeline.preprocess()
        df_out = pd.read_csv(self.output_file)

        # Check that Surname was dropped
        self.assertNotIn("Surname", df_out.columns)

        # Check Geography one-hot encoding
        self.assertIn("Geography_France", df_out.columns)
        self.assertIn("Geography_Spain", df_out.columns)

        # Check Gender one-hot encoding (NOT a single numeric column)
        self.assertIn("Gender_Female", df_out.columns)
        self.assertIn("Gender_Male", df_out.columns)

        # Ensure CustomerId is preserved
        self.assertIn("CustomerId", df_out.columns)

        # Check numeric features are present
        self.assertIn("CreditScore", df_out.columns)
        self.assertIn("Age", df_out.columns)
        self.assertIn("Balance", df_out.columns)

    def test_gender_is_encoded_correctly(self):
        """Test that Gender column is one-hot encoded into Gender_Female and Gender_Male."""
        self.pipeline.preprocess()
        df_out = pd.read_csv(self.output_file)
        
        # Gender should be one-hot encoded, not a single column
        self.assertIn("Gender_Female", df_out.columns)
        self.assertIn("Gender_Male", df_out.columns)
        
        # Values should be binary (0 or 1)
        self.assertTrue(df_out["Gender_Female"].isin([0, 1]).all())
        self.assertTrue(df_out["Gender_Male"].isin([0, 1]).all())
        
        # For each row, exactly one gender column should be 1
        gender_sum = df_out["Gender_Female"] + df_out["Gender_Male"]
        self.assertTrue((gender_sum == 1).all(), 
                       "Each row should have exactly one gender encoded as 1")

    def test_geography_is_encoded_correctly(self):
        """Test that Geography column is one-hot encoded."""
        self.pipeline.preprocess()
        df_out = pd.read_csv(self.output_file)
        
        # Check for Geography columns
        self.assertIn("Geography_France", df_out.columns)
        self.assertIn("Geography_Spain", df_out.columns)
        
        # Values should be binary
        self.assertTrue(df_out["Geography_France"].isin([0, 1]).all())
        self.assertTrue(df_out["Geography_Spain"].isin([0, 1]).all())

    def test_numeric_columns_preserved(self):
        """Test that numeric columns are preserved after preprocessing."""
        self.pipeline.preprocess()
        df_out = pd.read_csv(self.output_file)
        
        numeric_columns = ["CreditScore", "Age", "Tenure", "Balance", 
                          "NumOfProducts", "HasCrCard", "IsActiveMember", 
                          "EstimatedSalary"]
        
        for col in numeric_columns:
            self.assertIn(col, df_out.columns, 
                         f"Numeric column {col} should be preserved")

    def test_no_missing_values_after_preprocessing(self):
        """Test that there are no missing values in the output."""
        self.pipeline.preprocess()
        df_out = pd.read_csv(self.output_file)
        
        # Check for any NaN values
        self.assertFalse(df_out.isnull().any().any(), 
                        "Output should not contain any missing values")

    def test_correct_number_of_rows(self):
        """Test that no rows are lost during preprocessing."""
        self.pipeline.preprocess()
        df_in = pd.read_csv(self.input_file)
        df_out = pd.read_csv(self.output_file)
        
        self.assertEqual(len(df_in), len(df_out), 
                        "Number of rows should remain the same")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()