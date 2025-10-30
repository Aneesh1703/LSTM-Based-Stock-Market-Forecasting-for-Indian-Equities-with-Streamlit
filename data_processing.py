import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
import io

def get_indian_stocks():
    """Returns a list of predefined Indian stock tickers."""
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
        "TATASTEEL.NS", "AXISBANK.NS", "WIPRO.NS", "BHARTIARTL.NS", "LT.NS",
        "MARUTI.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
        "M&M.NS", "ULTRACEMCO.NS", "INDUSINDBK.NS", "POWERGRID.NS", "HCLTECH.NS"
    ]

def fetch_stock_data_df(ticker, period="6mo"):
    """Fetches historical stock data and returns a pandas DataFrame."""
    try:
        end_date = pd.to_datetime("today")
        if period == "6mo":
            start_date = end_date - pd.DateOffset(months=6)
        elif period == "2y":
            start_date = end_date - pd.DateOffset(years=2)
        elif period == "31d":
            start_date = end_date - pd.DateOffset(days=31)
        else:
            raise ValueError("Unsupported period")

        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history&includeAdjustedClose=true"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        csv_content = response.content.decode('utf-8')
        stock_df = pd.read_csv(io.StringIO(csv_content), index_col='Date', parse_dates=True)

        if stock_df.empty:
            print(f"No data found for {ticker}")
            return None
        return stock_df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def create_training_data(data, time_step=30):
    """Creates training data sequences."""
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
