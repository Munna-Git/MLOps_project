import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.data.clean_data import preprocess


class TestPreprocessData(unittest.TestCase):

        def setUp(self):
            # Setup sample input DataFrame
            self.sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [10, 20, 30, 40, 50]
            })
            self.expected_output = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [10, 20, 30, 40, 50]
            })

        def test_preprocess(self):
            # Apply preprocess function
            result = preprocess(self.sample_data)
            # Compare DataFrames
            assert_frame_equal(result, self.expected_output)

if __name__ == '__main__':
    unittest.main()
