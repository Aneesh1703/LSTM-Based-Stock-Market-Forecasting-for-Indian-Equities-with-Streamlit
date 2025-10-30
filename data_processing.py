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

def get_stock_data(ticker, period="2y", interval="1d"):
    """Fetches historical stock data for a given ticker."""
    try:
        stock = yf.download(ticker, period=period, interval=interval)
        if stock.empty:
            print(f"No data found for {ticker}")
            return None
        return stock["Close"].dropna().values.reshape(-1, 1)
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
