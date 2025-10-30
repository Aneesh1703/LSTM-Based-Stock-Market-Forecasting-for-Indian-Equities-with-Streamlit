# AI-Powered Stock Price Predictor

## Overview

This project is an AI-powered stock price prediction application that forecasts future stock prices using a Long Short-Term Memory (LSTM) neural network. The application is built with Python and features a user-friendly web interface created with Streamlit. The codebase is structured for maintainability and includes unit tests to ensure reliability.

## Key Features

- **LSTM-Based Prediction Model**: Utilizes an LSTM model to forecast stock prices based on historical data.
- **Interactive Web Interface**: A Streamlit application allows users to select from a list of Indian stocks and receive real-time price predictions.
- **Data Visualization**: Displays historical stock price data in an interactive chart, providing users with visual context for the predictions.
- **Modular and Maintainable Code**: The project is organized into distinct modules for data processing, model training, and application logic, following software engineering best practices.
- **Unit Tested**: Includes a suite of unit tests to ensure the reliability and correctness of the data processing pipeline.

## Project Structure

```
.
├── models/
│   ├── lstm_stock_model.keras
│   └── scaler.pkl
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data_processing.py
│   └── model_training.py
├── tests/
│   └── test_data_processing.py
├── README.md
└── requirements.txt
```

## How to Get Started

### Prerequisites

- Python 3.8+
- Pip for package management

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Model Training

The project includes pre-trained model and scaler files. However, if you wish to retrain the model with the latest data, you can run the training script:

```sh
python -m src.model_training
```

_Note: The `yfinance` library can occasionally be unreliable. If you encounter data fetching errors during training, please try running the script again after a short while._

### Running the Application

To launch the Streamlit web interface, run the following command:

```sh
streamlit run src/app.py
```

You can then access the application in your web browser at `http://localhost:8501`.

## How to Run Tests

To ensure the data processing logic is functioning correctly, you can run the included unit tests:

```sh
python -m unittest discover tests
```
