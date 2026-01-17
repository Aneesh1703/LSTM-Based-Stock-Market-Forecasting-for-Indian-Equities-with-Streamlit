import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

from data_processing import get_indian_stocks, fetch_stock_data_df

# Load trained model and scaler
MODEL_PATH = os.path.join("models", "lstm_stock_model.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler not found. Please train the model first.")
    st.stop()

try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error")
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Streamlit UI
st.title("Indian Stock Market Prediction (LSTM)")

tickers = get_indian_stocks()
selected_stock = st.selectbox("Select a stock:", tickers)

if st.button("Predict"):
    # Fetch the historical data as a DataFrame
    stock_df = fetch_stock_data_df(selected_stock, period="6mo")

    if stock_df is None:
        st.error("Failed to fetch stock data. The chart cannot be displayed.")
    else:
        st.subheader("Historical Stock Prices")
        st.line_chart(stock_df['Close'])

        # Prepare data for prediction
        data = stock_df['Close'].values.reshape(-1, 1)

        if data.shape[0] < 30:
            st.warning("Not enough data for prediction (need at least 30 days).")
        else:
            try:
                # Scale the data and prepare the input for the model
                scaled_data = scaler.transform(data)
                test_input = scaled_data[-30:].reshape(1, 30, 1) # 1 feature

                # Make a prediction
                pred_scaled = model.predict(test_input)

                # Inverse transform the prediction
                pred = scaler.inverse_transform(pred_scaled)

                st.success(f"Predicted Stock Price for {selected_stock}: ₹{pred[0][0]:.2f}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
