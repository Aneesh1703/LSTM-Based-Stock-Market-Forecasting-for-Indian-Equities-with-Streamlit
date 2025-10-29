import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_indian_stocks():
    """Returns a list of predefined Indian stock tickers."""
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "ITC.NS",
        "TATASTEEL.NS", "AXISBANK.NS", "WIPRO.NS", "BHARTIARTL.NS", "LT.NS",
        "MARUTI.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
        "M&M.NS", "ULTRACEMCO.NS", "INDUSINDBK.NS", "POWERGRID.NS", "HCLTECH.NS"
    ]

def get_stock_data(ticker, period="2y", interval="1d", sma_window=10):
    """Fetches historical stock data for a given ticker and calculates SMA."""
    try:
        # Let yfinance handle the session automatically
        stock_df = yf.download(ticker, period=period, interval=interval)
        if stock_df.empty:
            print(f"No data found for {ticker}")
            return None

        # Calculate Simple Moving Average
        stock_df['SMA'] = stock_df['Close'].rolling(window=sma_window).mean()

        # Drop rows with NaN values
        stock_df.dropna(inplace=True)

        # Return Close and SMA as a NumPy array
        return stock_df[['Close', 'SMA']].values

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def create_training_data(data, time_step=30):
    """Creates training data sequences from multi-feature data."""
    X, y = [], []
    for i in range(time_step, len(data)):
        # Input sequence includes all features
        X.append(data[i-time_step:i, :])
        # Output is the next day's closing price (the first column)
        y.append(data[i, 0])
    return np.array(X), np.array(y)
