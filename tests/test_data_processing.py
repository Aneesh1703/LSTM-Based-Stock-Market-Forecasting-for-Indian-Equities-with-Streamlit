import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing import get_stock_data

class TestDataProcessing(unittest.TestCase):

    @patch('yfinance.download')
    def test_get_stock_data(self, mock_download):
        # Create a mock DataFrame to be returned by yfinance.download
        mock_data = {'Close': np.linspace(100, 150, 50)}
        mock_df = pd.DataFrame(mock_data)
        mock_download.return_value = mock_df

        # Call the function with a dummy ticker
        ticker = "DUMMY.NS"
        data = get_stock_data(ticker)

        # 1. Check if the function returns a NumPy array
        self.assertIsInstance(data, np.ndarray)

        # 2. Check if the output has one column (Close)
        self.assertEqual(data.shape[1], 1)

        # 3. Check if the number of rows is correct
        self.assertEqual(data.shape[0], 50)

        # 4. Check if the 'Close' prices are correct
        self.assertTrue(np.array_equal(data, mock_df['Close'].values.reshape(-1, 1)))

if __name__ == '__main__':
    unittest.main()
