# QF5210 Project Requirements: Crude Oil Price Forecasting

> **Source note.** This file reproduces the oil price forecasting option of the course brief as it was provided. The description of the alternative option, the trading strategy project, is not part of this text.

---

## Overview

For group project (25%), students may also choose to work on **oil price forecasting for the next 1-60 days ahead** as an alternative to the trading strategy project.

## Project Objective

Each group will develop a **data-driven forecasting system for crude oil prices (and/or returns) using daily data**. The project should include at least one major crude oil benchmark, such as WTI or Brent, and may incorporate additional oil benchmarks, energy indices, futures, ETFs, commodity prices, or relevant financial and macroeconomic variables. All market variables used in the main analysis should have sufficient historical observations to permit meaningful model training and out-of-sample evaluation.

Each group should develop models and methods that can **automatically update forecasts as new daily observations become available**. Students are encouraged to compare conventional time-series models with modern machine-learning or AI approaches.

Any programming language (R, Python, C, Rust, Julia, etc.) is allowed.

## Possible Forecasting Approaches

Examples include:

- Classical time-series models, including ARIMA, VAR, GARCH, and related models
- Machine-learning methods such as random forests, gradient boosting, and XGBoost
- Deep-learning models such as LSTM, Transformer, or other sequence models
- Regime-switching models capturing different oil-market conditions
- Multi-factor forecasting models incorporating equity, commodity, interest-rate, exchange-rate, inventory, or macroeconomic variables
- News- or sentiment-based forecasting using macroeconomic and geopolitical information
- Hybrid or ensemble forecasting methods combining multiple models

Groups are encouraged to justify their choice of forecasting target, predictors, methodology, and benchmark models.

---

## Evaluation Emphasis: Genuine Out-of-Sample Forecasting

A key emphasis of this project is **genuine out-of-sample forecasting performance**. The test period must not be used for model estimation or parameter selection, and the evaluation design should avoid look-ahead bias and data leakage.

Forecasting performance should be evaluated using multiple complementary metrics, including:

### 1. Forecast accuracy
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE), where appropriate

### 2. Directional accuracy
- Percentage of days for which the model correctly predicts the direction of the price/return movement
- Precision, recall, or F1 score for positive/negative movements, where appropriate

### 3. Performance relative to benchmarks
- Comparison against simple benchmark models such as random walk/no-change, historical mean, AR/ARIMA, or other appropriate baselines
- Relative improvement in MAE/RMSE over the benchmark

### 4. Forecast robustness
- Performance across different market regimes, particularly high- versus low-volatility periods
- Stability of forecasting performance across different out-of-sample windows

### 5. Economic value of forecasts
- Groups are encouraged to translate their forecasts into a simple trading rule and report resulting P&L, annualized return, Sharpe ratio, maximum drawdown, and transaction costs.
- The emphasis remains on forecasting; trading performance serves as an additional measure of whether the forecasts contain economically useful information.

**A model will not be considered successful solely because it achieves a low in-sample forecasting error.** Out-of-sample performance, robustness, and comparison against meaningful benchmarks will receive substantially greater weight.

---

## Data Submission

The oil price series and other major variables used in the analysis, together with their daily observations, **must be submitted in CSV format along with the project presentation**. The group should clearly document:

- the data sources
- variable definitions
- sample period
- training / validation / test split
- data preprocessing procedures
