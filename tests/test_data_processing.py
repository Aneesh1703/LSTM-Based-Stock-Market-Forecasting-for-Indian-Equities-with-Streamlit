import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path for the test runner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import fetch_stock_data_df

class TestDataProcessing(unittest.TestCase):

    @patch('requests.get')
    def test_fetch_stock_data_df(self, mock_get):
        # Create a mock CSV response
        csv_data = "Date,Open,High,Low,Close,Adj Close,Volume\n"
        dates = pd.date_range(end=pd.to_datetime("today"), periods=50).strftime('%Y-%m-%d')
        prices = np.linspace(100, 150, 50)
        for i in range(50):
            csv_data += f"{dates[i]},{prices[i]},{prices[i]},{prices[i]},{prices[i]},{prices[i]},1000\n"

        # Configure the mock to return the CSV data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = csv_data.encode('utf-8')
        mock_get.return_value = mock_response

        # Call the function with a dummy ticker
        ticker = "DUMMY.NS"
        df = fetch_stock_data_df(ticker)

        # 1. Check if the function returns a pandas DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # 2. Check if the DataFrame has the correct number of rows
        self.assertEqual(len(df), 50)

        # 3. Check if the 'Close' column is correct
        self.assertTrue(np.allclose(df['Close'].values, prices))

if __name__ == '__main__':
    unittest.main()
