# NVDA Time Series Analysis

Predictive modelling and time series analysis of NVIDIA (NVDA) daily stock returns using machine learning regression models and ARIMA, covering the period January 2021 to March 2026.

---

## Project Structure

```
├── nvda-time-series-analysis.ipynb        # Main analysis notebook
├── data/
│   └── nvda_raw.csv   # Saved dataset (auto-generated on first run)
└── README.md
```

---

## Objective

The project investigates whether next-day NVDA log returns can be predicted from historical price, volume, and macroeconomic features. It evaluates four regression models of increasing complexity alongside an ARIMA time series model, and assesses results in the context of the Efficient Market Hypothesis.

---

## Data

- **Source:** Yahoo Finance via `yfinance`, ticker `NVDA`
- **Period:** January 2021 to March 2026 (1,297 trading days)
- **Columns:** Open, High, Low, Close, Volume (OHLCV)
- **Adjustment:** `auto_adjust=True` accounts for the 10-for-1 stock split in June 2024

The dataset is saved to `data/nvda_raw.csv` on first run and loaded from file on subsequent runs to ensure reproducibility, since yfinance prices can be retroactively adjusted.

---

## Notebook Structure

### 1. Data Import and Cleaning
Downloads or loads NVDA data, removes duplicates, forward-fills any missing values, and verifies no nulls remain.

### 2. Data Inspection
Checks shape, dtypes, date range, and descriptive statistics before any transformations.

### 3. Feature Engineering
Constructs the following features from raw OHLCV data:

| Feature | Description |
|---|---|
| `Log_Return` | Daily log return — target variable |
| `Lag_1`, `Lag_2`, `Lag_3` | Lagged log returns |
| `Volatility_10`, `Volatility_20` | Rolling standard deviation of returns |
| `BB_Pct` | Bollinger Band percentage position |
| `Volume_Change` | Day-over-day volume change |
| `SP500_Return` | Daily S&P 500 log return |
| `VIX` | CBOE Volatility Index (fear gauge) |

### 4. Exploratory Data Analysis
- Close price and log return time series plots
- Return distribution (histogram with KDE)
- Correlation heatmap across all features
- ADF stationarity tests on Close prices, log returns, S&P 500 returns, and VIX

### 5. Time Series Diagnostics
- ACF and PACF plots for log returns and squared returns
- Ljung-Box test at lags 5, 10, 20 on returns and squared returns
- Jarque-Bera and Shapiro-Wilk normality tests
- ARCH-LM test for volatility clustering

### 6. Modelling
Chronological 80/20 train-test split. Four models evaluated:

| Model | Notes |
|---|---|
| Linear Regression | Baseline, no hyperparameters |
| Ridge Regression | Alpha selected via RidgeCV; optimal α=100 |
| Random Forest | Tuned with RandomizedSearchCV + TimeSeriesSplit |
| Gradient Boosting | Two-grid tuning strategy, n_iter=50 on Grid 2 |

StandardScaler fitted on training data only and applied to linear models. Tree-based models received raw features.

### 7. ARIMA Analysis
- Grid search over p ∈ {0..3}, q ∈ {0..3} with d=0 fixed (returns are stationary)
- Model selected by AIC: **ARIMA(1,0,0)**
- Residual diagnostics: Ljung-Box, ARCH-LM
- 30-day out-of-sample return forecast with 95% confidence intervals

### 8. Evaluation and Discussion
Model comparison by MAE, RMSE, and R². Residual plots across all four models.

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.019691 | 0.027641 | 0.0042 |
| Optimized Ridge | 0.019608 | 0.027573 | 0.0090 |
| Random Forest | 0.019561 | 0.027554 | 0.0104 |
| Gradient Boosting | 0.019375 | 0.027441 | 0.0185 |

All models explain less than 2% of next-day return variance. This is consistent with the weak form of the Efficient Market Hypothesis — past price information carries no exploitable linear signal for return prediction. The ARIMA(1,0,0) forecast converges immediately to the unconditional mean (~0.2% daily), confirming the same conclusion from a time series perspective.

---

## Key Findings

- Log returns are stationary, serially uncorrelated, and non-normal (fat tails, kurtosis 6.64)
- Squared returns show mild ARCH effects at short lags, indicating volatility clustering
- No regression model achieves meaningful predictive power regardless of complexity
- ARIMA residuals pass the white noise test but remain non-normal, consistent with equity return stylised facts
- The natural next step is a GARCH(1,1) model targeting conditional variance rather than the conditional mean

---

## Dependencies

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
scipy==1.13.1
statsmodels==0.14.2
scikit-learn==1.4.2
yfinance==0.2.38
jupyter==1.0.0
```

Install with:

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

---

## Reproducibility
```bash
# 1. Clone the repo
git clone https://github.com/AssylbekO/NVIDIA-Stock-Return-Analysis.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
# On first run, data is downloaded automatically from Yahoo Finance and saved to data/nvda_raw.csv
jupyter notebook main.ipynb
```
