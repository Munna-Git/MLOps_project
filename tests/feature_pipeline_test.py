import unittest
import sys
import os

# --- Add this block ---
# Get the absolute path to the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------------------

from src.data.clean_data import preprocess

class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize any required variables or state
        self.sample_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        }
        self.expected_output = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        }

    def test_preprocess(self):
        # Test the preprocess function
        result = preprocess(self.sample_data)
        self.assertEqual(result, self.expected_output)

if __name__ == '__main__':
    unittest.main()
