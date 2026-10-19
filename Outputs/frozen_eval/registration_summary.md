# Out-of-sample performance summary: oil price forecasting

**Test design.** Models were estimated only on data up to 2024-12-31 and then applied day by day from 2025-01-02 to 2026-10-18 (452 forecast dates), with no refitting. Forecasts are expected cumulative log returns, converted to prices. Benchmarks: no change (random walk), historical mean, direct AR. RMSE vs no change below 1 means better than the no-change forecast; DM p is the Diebold-Mariano p-value against no change (HAC, h-1 lags); band coverage is the share of outcomes inside the model's +-1 sd band (68% if well calibrated). Window and weight choices come from the notebooks' own 2022-2023 validation; some design decisions were made after seeing later data, so the test is clean for estimation but not pristine.

## Accuracy by horizon

| Series | Model | Type | Test period | RMSE vs no change h=1 | MAPE % h=1 | Direction % h=1 | DM p h=1 | Band coverage % h=1 | F1 up h=1 | Forecast dates h=1 | RMSE vs no change h=5 | MAPE % h=5 | Direction % h=5 | DM p h=5 | Band coverage % h=5 | F1 up h=5 | RMSE vs no change h=20 | MAPE % h=20 | Direction % h=20 | DM p h=20 | Band coverage % h=20 | F1 up h=20 | RMSE vs no change h=60 | MAPE % h=60 | Direction % h=60 | DM p h=60 | Band coverage % h=60 | F1 up h=60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Brent futures | No change | benchmark | 2025-01-02 to 2026-10-18 | 1.000 | 1.99 |  |  |  |  | 451 | 1.000 | 4.57 |  |  |  |  | 1.000 | 8.50 |  |  |  |  | 1.000 | 15.78 |  |  |  |  |
| Brent futures | Regime LASSO | model | 2025-01-02 to 2026-10-18 | 1.000 | 1.99 | 50 | 1.00 | 75 | 0.28 | 451 | 0.999 | 4.56 | 50 | 0.31 | 75 | 0.08 | 1.000 | 8.50 | 54 | 1.00 | 79 | 0.12 | 1.000 | 15.78 |  | 1.00 | 76 |  |
| Brent futures | XGBoost | model | 2025-01-02 to 2026-10-18 | 1.000 | 1.99 | 54 | 0.19 | 75 | 0.67 | 451 | 1.000 | 4.57 | 45 | 0.15 | 75 | 0.24 | 1.001 | 8.50 | 49 | 0.59 | 80 | 0.41 | 1.011 | 16.05 | 51 | 0.48 | 77 | 0.53 |
| Brent futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-18 | 1.000 | 1.99 | 51 | 0.94 | 72 | 0.52 | 451 | 1.000 | 4.57 | 48 | 0.55 | 70 | 0.48 | 1.000 | 8.50 | 53 | 0.98 | 71 | 0.52 | 1.000 | 15.78 | 52 | 0.77 | 70 | 0.50 |
| Brent futures | MS-AR | model | 2025-01-02 to 2026-10-18 | 1.004 | 2.01 | 48 | 0.50 | 76 | 0.55 | 451 | 1.014 | 4.63 | 48 | 0.27 | 76 | 0.59 | 1.011 | 8.68 | 48 | 0.31 | 83 | 0.58 | 0.979 | 15.56 | 55 | 0.31 | 77 | 0.64 |
| Brent futures | MS-ARX | model | 2025-01-02 to 2026-10-18 | 1.000 | 2.00 | 46 | 0.87 | 73 | 0.53 | 451 | 1.005 | 4.59 | 43 | 0.22 | 67 | 0.53 | 1.002 | 8.51 | 43 | 0.76 | 71 | 0.53 | 0.997 | 15.90 | 46 | 0.83 | 66 | 0.57 |
| Brent futures | Historical mean | benchmark | 2025-01-02 to 2026-10-18 | 1.000 | 1.99 | 54 | 0.57 |  | 0.70 | 451 | 1.000 | 4.57 | 51 | 0.50 |  | 0.67 | 1.000 | 8.50 | 49 | 0.57 |  | 0.66 | 1.001 | 15.76 | 52 | 0.60 |  |  |
| Brent futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-18 | 1.000 | 1.99 | 50 | 0.89 |  | 0.51 | 451 | 1.000 | 4.57 | 53 | 0.46 |  | 0.67 | 1.000 | 8.50 | 49 | 0.82 |  | 0.55 | 1.001 | 15.77 | 49 | 0.24 |  | 0.44 |
| WTI futures | No change | benchmark | 2025-01-02 to 2026-10-18 | 1.000 | 2.07 |  |  |  |  | 451 | 1.000 | 4.82 |  |  |  |  | 1.000 | 8.54 |  |  |  |  | 1.000 | 16.05 |  |  |  |  |
| WTI futures | Regime LASSO | model | 2025-01-02 to 2026-10-18 | 1.000 | 2.07 | 49 | 1.00 | 74 | 0.59 | 451 | 1.000 | 4.82 |  | 1.00 | 74 |  | 1.000 | 8.54 |  | 1.00 | 71 |  | 1.000 | 16.05 | 44 | 1.00 | 62 | 0.30 |
| WTI futures | XGBoost | model | 2025-01-02 to 2026-10-18 | 0.998 | 2.07 | 51 | 0.07 | 74 | 0.56 | 451 | 0.995 | 4.78 | 56 | 0.41 | 72 | 0.59 | 1.000 | 8.54 | 50 | 0.99 | 71 | 0.64 | 1.000 | 16.08 | 43 | 0.91 | 61 | 0.60 |
| WTI futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-18 | 1.000 | 2.08 | 51 | 0.96 | 72 | 0.50 | 451 | 1.001 | 4.82 | 50 | 0.57 | 68 | 0.49 | 1.000 | 8.53 | 54 | 0.89 | 75 | 0.51 | 1.000 | 16.05 | 51 | 0.49 | 71 | 0.48 |
| WTI futures | MS-AR | model | 2025-01-02 to 2026-10-18 | 1.001 | 2.08 | 48 | 0.75 | 67 | 0.52 | 451 | 1.004 | 4.86 | 50 | 0.69 | 64 | 0.52 | 1.031 | 9.33 | 43 | 0.13 | 69 | 0.46 | 0.993 | 17.71 | 49 | 0.92 | 64 | 0.58 |
| WTI futures | MS-ARX | model | 2025-01-02 to 2026-10-18 | 1.001 | 2.08 | 47 | 0.50 | 72 | 0.50 | 451 | 1.005 | 4.84 | 41 | 0.01 | 72 | 0.48 | 1.004 | 8.60 | 41 | 0.13 | 78 | 0.50 | 1.001 | 16.11 | 45 | 0.81 | 72 | 0.55 |
| WTI futures | Historical mean | benchmark | 2025-01-02 to 2026-10-18 | 1.000 | 2.07 | 53 | 0.80 |  | 0.69 | 451 | 0.999 | 4.83 | 50 | 0.76 |  | 0.66 | 0.997 | 8.75 | 46 | 0.83 |  | 0.63 | 0.989 | 16.93 | 46 | 0.80 |  | 0.63 |
| WTI futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-18 | 1.000 | 2.07 | 52 | 0.77 |  | 0.69 | 451 | 1.009 | 4.86 | 52 | 0.64 |  | 0.57 | 0.997 | 8.74 | 46 | 0.85 |  | 0.63 | 0.988 | 16.93 | 46 | 0.79 |  | 0.63 |


## Robustness (RMSE / RMSE of no change; below 1 is better)

| Series | Model | Horizon (days) | High-volatility RMSE ratio | Low-volatility RMSE ratio | 2025 RMSE ratio | 2026 RMSE ratio | Worst window RMSE ratio | Best window RMSE ratio | Windows better than no change |
|---|---|---|---|---|---|---|---|---|---|
| Brent futures | Regime LASSO | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | Regime LASSO | 5 | 0.998 | 1.000 | 1.000 | 0.998 | 1.000 | 0.997 | 4 of 4 |
| Brent futures | Regime LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | Regime LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | XGBoost | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 3 of 4 |
| Brent futures | XGBoost | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1 of 4 |
| Brent futures | XGBoost | 20 | 1.001 | 0.997 | 0.997 | 1.001 | 1.003 | 0.994 | 1 of 4 |
| Brent futures | XGBoost | 60 | 1.023 | 0.995 | 0.994 | 1.019 | 1.037 | 0.954 | 1 of 4 |
| Brent futures | ARMA-GARCH | 1 | 1.000 | 1.000 | 1.001 | 1.000 | 1.004 | 0.997 | 2 of 4 |
| Brent futures | ARMA-GARCH | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 0.999 | 1 of 4 |
| Brent futures | ARMA-GARCH | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 3 of 4 |
| Brent futures | ARMA-GARCH | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | MS-AR | 1 | 1.005 | 1.001 | 0.996 | 1.006 | 1.017 | 0.988 | 3 of 4 |
| Brent futures | MS-AR | 5 | 1.017 | 1.004 | 1.005 | 1.018 | 1.030 | 0.996 | 1 of 4 |
| Brent futures | MS-AR | 20 | 1.009 | 1.034 | 1.025 | 1.009 | 1.058 | 1.007 | 0 of 4 |
| Brent futures | MS-AR | 60 | 0.965 | 0.995 | 1.002 | 0.966 | 1.095 | 0.940 | 2 of 4 |
| Brent futures | MS-ARX | 1 | 1.000 | 1.000 | 0.999 | 1.000 | 1.001 | 0.997 | 2 of 4 |
| Brent futures | MS-ARX | 5 | 1.008 | 0.996 | 1.001 | 1.007 | 1.012 | 0.993 | 1 of 4 |
| Brent futures | MS-ARX | 20 | 1.001 | 1.015 | 1.021 | 0.999 | 1.065 | 0.997 | 1 of 4 |
| Brent futures | MS-ARX | 60 | 0.990 | 1.005 | 1.018 | 0.986 | 1.226 | 0.951 | 2 of 4 |
| Brent futures | Historical mean | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Historical mean | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Historical mean | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1 of 4 |
| Brent futures | Historical mean | 60 | 1.001 | 1.000 | 1.000 | 1.001 | 1.002 | 0.993 | 3 of 4 |
| Brent futures | AR (direct) | 1 | 1.000 | 1.000 | 1.001 | 0.999 | 1.003 | 0.997 | 1 of 4 |
| Brent futures | AR (direct) | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | AR (direct) | 20 | 1.000 | 1.001 | 1.001 | 1.000 | 1.002 | 1.000 | 1 of 4 |
| Brent futures | AR (direct) | 60 | 1.002 | 1.000 | 1.001 | 1.001 | 1.002 | 0.998 | 2 of 4 |
| WTI futures | Regime LASSO | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Regime LASSO | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Regime LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Regime LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | XGBoost | 1 | 0.998 | 0.996 | 0.998 | 0.998 | 1.001 | 0.996 | 3 of 4 |
| WTI futures | XGBoost | 5 | 0.996 | 0.987 | 0.994 | 0.995 | 1.014 | 0.983 | 2 of 4 |
| WTI futures | XGBoost | 20 | 1.000 | 0.999 | 0.999 | 1.000 | 1.002 | 0.998 | 2 of 4 |
| WTI futures | XGBoost | 60 | 0.999 | 1.001 | 1.000 | 1.000 | 1.007 | 0.998 | 2 of 4 |
| WTI futures | ARMA-GARCH | 1 | 1.000 | 1.002 | 1.004 | 0.999 | 1.005 | 0.996 | 1 of 4 |
| WTI futures | ARMA-GARCH | 5 | 1.002 | 0.999 | 0.999 | 1.002 | 1.003 | 0.993 | 1 of 4 |
| WTI futures | ARMA-GARCH | 20 | 1.000 | 0.998 | 0.998 | 1.000 | 1.002 | 0.995 | 3 of 4 |
| WTI futures | ARMA-GARCH | 60 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 0.998 | 3 of 4 |
| WTI futures | MS-AR | 1 | 1.000 | 1.007 | 1.004 | 1.000 | 1.008 | 0.995 | 2 of 4 |
| WTI futures | MS-AR | 5 | 0.996 | 1.032 | 1.027 | 0.995 | 1.038 | 0.988 | 1 of 4 |
| WTI futures | MS-AR | 20 | 0.994 | 1.284 | 1.230 | 0.990 | 1.383 | 0.984 | 2 of 4 |
| WTI futures | MS-AR | 60 | 0.979 | 1.008 | 1.065 | 0.953 | 1.560 | 0.902 | 2 of 4 |
| WTI futures | MS-ARX | 1 | 1.001 | 1.000 | 1.000 | 1.001 | 1.002 | 0.999 | 2 of 4 |
| WTI futures | MS-ARX | 5 | 1.007 | 0.999 | 1.001 | 1.007 | 1.009 | 0.997 | 1 of 4 |
| WTI futures | MS-ARX | 20 | 1.003 | 1.010 | 1.009 | 1.003 | 1.035 | 1.002 | 0 of 4 |
| WTI futures | MS-ARX | 60 | 1.000 | 1.003 | 1.009 | 0.997 | 1.093 | 0.985 | 2 of 4 |
| WTI futures | Historical mean | 1 | 1.000 | 1.000 | 1.002 | 0.999 | 1.003 | 0.999 | 1 of 4 |
| WTI futures | Historical mean | 5 | 0.998 | 1.004 | 1.008 | 0.995 | 1.018 | 0.992 | 1 of 4 |
| WTI futures | Historical mean | 20 | 0.988 | 1.068 | 1.054 | 0.986 | 1.129 | 0.969 | 1 of 4 |
| WTI futures | Historical mean | 60 | 0.983 | 0.996 | 1.018 | 0.974 | 1.291 | 0.934 | 1 of 4 |
| WTI futures | AR (direct) | 1 | 1.000 | 1.000 | 1.002 | 0.999 | 1.003 | 0.999 | 2 of 4 |
| WTI futures | AR (direct) | 5 | 1.008 | 1.016 | 0.978 | 1.021 | 1.035 | 0.920 | 1 of 4 |
| WTI futures | AR (direct) | 20 | 0.988 | 1.067 | 1.053 | 0.987 | 1.124 | 0.972 | 1 of 4 |
| WTI futures | AR (direct) | 60 | 0.982 | 0.996 | 1.017 | 0.973 | 1.290 | 0.935 | 1 of 4 |

