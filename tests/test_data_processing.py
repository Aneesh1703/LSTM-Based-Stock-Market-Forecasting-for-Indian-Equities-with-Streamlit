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
        # Create a mock array and DataFrame to be returned by yfinance.download
        close_prices = np.linspace(100, 150, 50)
        mock_df = pd.DataFrame({'Close': close_prices})
        mock_download.return_value = mock_df

        # Call the function with a dummy ticker
        ticker = "DUMMY.NS"
        data = get_stock_data(ticker, sma_window=10)

        # 1. Check if the function returns a NumPy array
        self.assertIsInstance(data, np.ndarray)

        # 2. Check if the output has two columns (Close and SMA)
        self.assertEqual(data.shape[1], 2)

        # 3. Check if the number of rows is correct
        # The original DataFrame has 50 rows. After calculating a 10-day SMA,
        # the first 9 rows will have NaN values and should be dropped.
        # So, the expected number of rows is 50 - 9 = 41.
        self.assertEqual(data.shape[0], 41)

        # 4. Check if the 'Close' prices are correct
        # The first 'Close' price in the output should correspond to the 10th price in the original data
        expected_close = close_prices[9]
        self.assertAlmostEqual(data[0, 0], expected_close)

        # 5. Check if the SMA is calculated correctly
        # The first SMA value should be the average of the first 10 'Close' prices
        expected_sma = np.mean(close_prices[:10])
        self.assertAlmostEqual(data[0, 1], expected_sma)

if __name__ == '__main__':
    unittest.main()
