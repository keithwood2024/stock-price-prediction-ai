# Stock Price Prediction using AI Models

**CAP 4630 - Intro to Artificial Intelligence**  
**Final Project**

---

## 📊 Project Overview

This project implements a comprehensive stock price prediction system using multiple artificial intelligence models. The system fetches historical stock data, creates technical indicators, trains three different AI models, and generates visualizations and performance metrics for presentation.

### 🎯 Objectives

- **Problem**: Predict future stock prices based on historical data and technical indicators
- **Why Important**: Accurate stock price prediction helps investors make informed decisions and understand market trends
- **Goal**: Compare multiple AI models (Linear Regression, Random Forest, LSTM) to find the best approach for stock price prediction

---

## 🗂️ System Description

### Dataset
- **Source**: Yahoo Finance API (yfinance)
- **Data Type**: Time series financial data
- **Features**: 
  - Historical prices (Open, High, Low, Close)
  - Trading volume
  - Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
  - Lagged features

### AI Models

1. **Linear Regression**
   - Simple baseline model
   - Assumes linear relationship between features and price
   - Fast training and inference

2. **Random Forest Regressor**
   - Ensemble learning method
   - Handles non-linear relationships
   - Robust to outliers

3. **LSTM (Long Short-Term Memory)**
   - Deep learning model for time series
   - Captures temporal dependencies
   - Can learn long-term patterns

### Why These Models?

- **Linear Regression**: Provides a simple baseline to compare against
- **Random Forest**: Excellent for tabular data with non-linear patterns
- **LSTM**: Specifically designed for sequential/time series data, ideal for stock prices

---

## 🔧 Methodology / Pipeline

### 1. Data Collection
- Fetch historical stock data from Yahoo Finance
- Date range: Customizable (default: 2019-2024)
- Stocks: Any ticker symbol (AAPL, GOOGL, TSLA, etc.)

### 2. Data Preprocessing
- Handle missing values
- Create technical indicators:
  - Moving Averages (7, 21, 50-day)
  - Exponential Moving Averages (12, 26-day)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Volatility measures
  - Lagged features

### 3. Feature Engineering
- 25+ technical features created
- Normalization using MinMaxScaler
- Train/test split (80/20)

### 4. Model Training
- Linear Regression: Trained on scaled features
- Random Forest: 100 estimators, parallel processing
- LSTM: 3-layer architecture with dropout, 60-day lookback

### 5. Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Model fit (closer to 1 is better)
- **MAPE** (Mean Absolute Percentage Error): Percentage error

### 6. Visualization
- Historical price trends
- Technical indicators
- Actual vs predicted prices
- Model performance comparison

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python stock_prediction.py
```

---

## 📈 Usage

### Basic Usage

The default configuration predicts **Apple (AAPL)** stock prices from **2019-2024**:

```python
python stock_prediction.py
```

### Customize Stock and Date Range

Edit the `main()` function in `stock_prediction.py`:

```python
# Configuration
TICKER = 'AAPL'  # Change to: GOOGL, TSLA, MSFT, AMZN, etc.
START_DATE = '2019-01-01'
END_DATE = '2024-10-01'
```

### Popular Stock Tickers to Try
- **AAPL**: Apple Inc.
- **GOOGL**: Alphabet (Google)
- **TSLA**: Tesla
- **MSFT**: Microsoft
- **AMZN**: Amazon
- **META**: Meta (Facebook)
- **NVDA**: NVIDIA

---

## 📊 Output Files

The system generates the following files for your presentation:

1. **`{TICKER}_history.png`** - Historical price chart with moving averages and volume
2. **`{TICKER}_technical_indicators.png`** - RSI, MACD, and Bollinger Bands
3. **`{TICKER}_predictions_comparison.png`** - Actual vs predicted prices for all models
4. **`{TICKER}_metrics_comparison.png`** - Performance metrics comparison
5. **`{TICKER}_report.txt`** - Detailed text report with all metrics

---

## 🎯 Implementation Results

### Expected Performance

Based on typical stock prediction tasks, you should expect:

- **Linear Regression**: Baseline performance, R² ~ 0.85-0.92
- **Random Forest**: Better performance, R² ~ 0.90-0.96
- **LSTM**: Best for time series, R² ~ 0.92-0.98

### Key Observations

**What Works Well:**
- All models capture general trends effectively
- Technical indicators improve prediction accuracy
- LSTM excels at capturing temporal patterns

**Challenges:**
- Stock markets are inherently unpredictable
- External factors (news, events) not captured
- High volatility periods are harder to predict

---

## 🔮 Future Improvements

1. **Sentiment Analysis** - Incorporate news headlines and social media sentiment
2. **More Features** - Add economic indicators (GDP, unemployment, interest rates)
3. **Ensemble Methods** - Combine multiple models for better predictions
4. **Real-time Prediction** - Build a web dashboard for live predictions
5. **Multiple Stocks** - Analyze portfolio of stocks simultaneously
6. **Deep Learning** - Try GRU, Transformer, or attention mechanisms

---

## 🛠️ Troubleshooting

### Common Issues

**Issue: "No module named 'yfinance'"**
```bash
pip install yfinance
```

**Issue: LSTM takes too long to train**
- Reduce `epochs` parameter (default: 30)
- Use smaller date range
- Reduce `lookback` parameter

**Issue: Memory error with LSTM**
- Reduce batch size
- Use smaller date range
- Close other applications

**Issue: No data found for ticker**
- Verify ticker symbol is correct
- Check if market was open during date range
- Try a different stock

---

## 📚 Technical Details

### Project Structure

```
Stocks/
│
├── stock_prediction.py      # Main script
├── requirements.txt          # Dependencies
├── README.md                # This file
│
└── Output files (generated):
    ├── AAPL_history.png
    ├── AAPL_technical_indicators.png
    ├── AAPL_predictions_comparison.png
    ├── AAPL_metrics_comparison.png
    └── AAPL_report.txt
```

### Model Architectures

**LSTM Architecture:**
```
Layer 1: LSTM(50 units, return_sequences=True) + Dropout(0.2)
Layer 2: LSTM(50 units, return_sequences=True) + Dropout(0.2)
Layer 3: LSTM(50 units) + Dropout(0.2)
Output: Dense(1 unit)

Optimizer: Adam
Loss: Mean Squared Error
Early Stopping: Patience=5
```

**Random Forest:**
```
Estimators: 100
Random State: 42
Parallel Processing: All CPU cores
```

---

## 📖 References

- **Yahoo Finance API**: https://pypi.org/project/yfinance/
- **Technical Indicators**: https://www.investopedia.com/
- **LSTM for Time Series**: Hochreiter & Schmidhuber (1997)
- **Random Forest**: Breiman (2001)

---

## 👥 Team Members
- **Keith Wood**
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]
- [Team Member 5]
