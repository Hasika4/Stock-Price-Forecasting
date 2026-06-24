# Stock Price Forecasting

## Overview
End-to-end time series forecasting pipeline built on 4,900+ daily 
Google stock price observations, comparing a statistical baseline 
(ARIMA) against a hybrid deep learning architecture (1D CNN-LSTM).

## Key Results
| Model             | MAE    | RMSE   | MAPE  |
|-------------------|--------|--------|-------|
| ARIMA(0,1,0)+Drift| $2.01  | $2.98  | 1.25% |
| CNN-LSTM          | $2.03  | $2.99  | 1.26% |

> CNN-LSTM performance closely matched the ARIMA baseline — consistent 
> with the Efficient Market Hypothesis on 1-step-ahead forecasting tasks.

## Methodology

### 1. Preprocessing
- Applied log-transformation to stabilise variance
- First-differencing to achieve stationarity
- Stationarity dual-verified via concurrent ADF and KPSS tests

### 2. ARIMA Baseline
- Diagnosed ACF/PACF plots on differenced series
- Selected optimal parameters via AIC grid search across 16 
  candidate models 
- Confirmed ARIMA(0,1,0) with Drift as optimal
- Captured statistically significant daily drift of ~0.08% (p=0.002)
- Residual diagnostics: Ljung-Box, Jarque-Bera, Heteroskedasticity
- Evaluated via walk-forward out-of-sample testing on 60-day window

### 3. Hybrid CNN-LSTM
- 15-step chronological sliding windows to 3D input [Samples, 15, 1]
- 1D Conv layer for localised pattern extraction
- LSTM layer for sequential temporal dependencies
- Dropout (0.1) on both layers for regularisation
- Leakage-free 1-step-ahead backtest on strict 60-day held-out window
- Predictions inverse-transformed from log-returns to dollar scale

## Key Finding
The CNN-LSTM achieved near-identical performance to the ARIMA 
baseline (MAPE 1.26% vs 1.25%), consistent with the Efficient Market 
Hypothesis — stock price series approximate a random walk, limiting 
the advantage of complex architectures over statistical baselines on 
1-step-ahead forecasting tasks.

## Tech Stack
Python · TensorFlow/Keras · Statsmodels · Scikit-learn · 
Pandas · NumPy · Matplotlib · Seaborn · Plotly

## Dataset
Google historical stock price data — 4,900+ daily OHLCV observations
