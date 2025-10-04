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
            "EstimatedSalary": [101348.88, 112542.58]
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

        # Check dropped column
        self.assertNotIn("Surname", df_out.columns)

        # Check categorical encoding
        self.assertIn("Geography_Spain", df_out.columns)
        self.assertIn("Gender", df_out.columns)  # Since it becomes numeric (0/1)

        # Ensure CustomerId is preserved
        self.assertIn("CustomerId", df_out.columns)

    def test_gender_is_encoded_correctly(self):
        """Test that Gender column is converted to numeric values."""
        self.pipeline.preprocess()
        df_out = pd.read_csv(self.output_file)
        self.assertTrue(df_out["Gender"].isin([0, 1]).all())

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
