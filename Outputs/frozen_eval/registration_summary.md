# Out-of-sample performance summary: oil price forecasting

**Test design.** Models were estimated only on data up to 2024-12-31 and then applied day by day from 2025-01-02 to 2026-10-29 (462 forecast dates), with no refitting. Forecasts are expected cumulative log returns, converted to prices. Benchmarks: no change (random walk), historical mean, direct AR. RMSE vs no change below 1 means better than the no-change forecast; DM p is the Diebold-Mariano p-value against no change (HAC, h-1 lags); band coverage is the share of outcomes inside the model's +-1 sd band (68% if well calibrated). Window and weight choices come from the notebooks' own 2022-2023 validation; some design decisions were made after seeing later data, so the test is clean for estimation but not pristine.

## Accuracy by horizon

| Series | Model | Type | Test period | RMSE vs no change h=1 | MAPE % h=1 | Direction % h=1 | DM p h=1 | Band coverage % h=1 | F1 up h=1 | Forecast dates h=1 | RMSE vs no change h=5 | MAPE % h=5 | Direction % h=5 | DM p h=5 | Band coverage % h=5 | F1 up h=5 | RMSE vs no change h=20 | MAPE % h=20 | Direction % h=20 | DM p h=20 | Band coverage % h=20 | F1 up h=20 | RMSE vs no change h=60 | MAPE % h=60 | Direction % h=60 | DM p h=60 | Band coverage % h=60 | F1 up h=60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Brent futures | No change | benchmark | 2025-01-02 to 2026-10-29 | 1.000 | 2.00 |  |  |  |  | 461 | 1.000 | 4.55 |  |  |  |  | 1.000 | 8.50 |  |  |  |  | 1.000 | 15.99 |  |  |  |  |
| Brent futures | Regime LASSO | model | 2025-01-02 to 2026-10-29 | 1.000 | 2.00 | 50 | 1.00 | 75 | 0.28 | 461 | 0.999 | 4.54 | 49 | 0.31 | 75 | 0.08 | 1.000 | 8.50 | 52 | 1.00 | 80 | 0.12 | 1.000 | 15.99 |  | 1.00 | 77 |  |
| Brent futures | XGBoost | model | 2025-01-02 to 2026-10-29 | 1.000 | 2.00 | 50 | 0.99 | 75 | 0.55 | 461 | 1.003 | 4.56 | 48 | 0.03 | 75 | 0.42 | 1.003 | 8.52 | 44 | 0.07 | 80 | 0.58 | 1.000 | 16.00 | 57 | 0.97 | 77 | 0.47 |
| Brent futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-29 | 1.000 | 2.00 | 52 | 0.80 | 72 | 0.53 | 461 | 1.000 | 4.55 | 48 | 0.57 | 70 | 0.48 | 1.000 | 8.50 | 53 | 0.95 | 71 | 0.52 | 1.000 | 15.99 | 51 | 0.48 | 70 | 0.51 |
| Brent futures | MS-AR | model | 2025-01-02 to 2026-10-29 | 1.001 | 2.01 | 48 | 0.61 | 77 | 0.56 | 461 | 1.006 | 4.57 | 49 | 0.38 | 77 | 0.60 | 1.004 | 8.55 | 49 | 0.50 | 83 | 0.60 | 0.992 | 15.89 | 53 | 0.49 | 77 | 0.63 |
| Brent futures | MS-ARX | model | 2025-01-02 to 2026-10-29 | 1.002 | 2.02 | 46 | 0.76 | 77 | 0.49 | 461 | 1.007 | 4.58 | 47 | 0.26 | 75 | 0.56 | 1.003 | 8.53 | 47 | 0.55 | 82 | 0.58 | 0.992 | 15.90 | 53 | 0.45 | 74 | 0.63 |
| Brent futures | Historical mean | benchmark | 2025-01-02 to 2026-10-29 | 1.000 | 2.00 | 54 | 0.52 |  | 0.70 | 461 | 1.000 | 4.55 | 52 | 0.43 |  | 0.68 | 1.000 | 8.50 | 50 | 0.51 |  | 0.67 | 1.001 | 15.98 | 50 | 0.54 |  |  |
| Brent futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-29 | 1.000 | 2.00 | 51 | 0.76 |  | 0.52 | 461 | 1.000 | 4.55 | 54 | 0.41 |  | 0.68 | 1.000 | 8.50 | 49 | 0.81 |  | 0.55 | 1.001 | 15.99 | 49 | 0.15 |  | 0.45 |
| WTI futures | No change | benchmark | 2025-01-02 to 2026-10-29 | 1.000 | 2.09 |  |  |  |  | 461 | 1.000 | 4.78 |  |  |  |  | 1.000 | 8.56 |  |  |  |  | 1.000 | 16.18 |  |  |  |  |
| WTI futures | Regime LASSO | model | 2025-01-02 to 2026-10-29 | 1.000 | 2.09 | 49 | 1.00 | 74 | 0.59 | 461 | 1.000 | 4.78 |  | 1.00 | 74 |  | 1.000 | 8.56 |  | 1.00 | 72 |  | 1.000 | 16.18 | 36 | 1.00 | 63 | 0.36 |
| WTI futures | XGBoost | model | 2025-01-02 to 2026-10-29 | 0.999 | 2.08 | 52 | 0.74 | 74 | 0.57 | 461 | 1.007 | 4.79 | 52 | 0.35 | 71 | 0.55 | 1.021 | 8.76 | 45 | 0.22 | 67 | 0.53 | 1.000 | 16.19 | 47 | 0.69 | 60 | 0.57 |
| WTI futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-29 | 1.000 | 2.09 | 51 | 0.84 | 72 | 0.50 | 461 | 1.001 | 4.79 | 50 | 0.58 | 68 | 0.49 | 1.000 | 8.55 | 54 | 0.96 | 74 | 0.52 | 1.000 | 16.18 | 51 | 0.33 | 71 | 0.49 |
| WTI futures | MS-AR | model | 2025-01-02 to 2026-10-29 | 1.000 | 2.09 | 48 | 0.99 | 67 | 0.51 | 461 | 1.000 | 4.79 | 50 | 0.97 | 65 | 0.52 | 1.011 | 8.90 | 42 | 0.24 | 74 | 0.45 | 0.991 | 16.85 | 49 | 0.79 | 68 | 0.59 |
| WTI futures | MS-ARX | model | 2025-01-02 to 2026-10-29 | 1.001 | 2.09 | 47 | 0.34 | 66 | 0.50 | 461 | 1.001 | 4.80 | 49 | 0.77 | 65 | 0.51 | 1.012 | 8.91 | 43 | 0.23 | 73 | 0.45 | 0.991 | 16.85 | 50 | 0.80 | 68 | 0.60 |
| WTI futures | Historical mean | benchmark | 2025-01-02 to 2026-10-29 | 1.000 | 2.08 | 53 | 0.73 |  | 0.69 | 461 | 0.999 | 4.80 | 50 | 0.70 |  | 0.67 | 0.996 | 8.74 | 48 | 0.76 |  | 0.64 | 0.986 | 16.97 | 48 | 0.74 |  | 0.64 |
| WTI futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-29 | 1.000 | 2.08 | 52 | 0.73 |  | 0.69 | 461 | 1.010 | 4.82 | 53 | 0.61 |  | 0.57 | 0.996 | 8.73 | 47 | 0.76 |  | 0.64 | 0.985 | 16.97 | 48 | 0.73 |  | 0.64 |


## Robustness (RMSE / RMSE of no change; below 1 is better)

| Series | Model | Horizon (days) | High-volatility RMSE ratio | Low-volatility RMSE ratio | 2025 RMSE ratio | 2026 RMSE ratio | Worst window RMSE ratio | Best window RMSE ratio | Windows better than no change |
|---|---|---|---|---|---|---|---|---|---|
| Brent futures | Regime LASSO | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | Regime LASSO | 5 | 0.998 | 1.000 | 1.000 | 0.998 | 1.000 | 0.997 | 4 of 4 |
| Brent futures | Regime LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | Regime LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | XGBoost | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 2 of 4 |
| Brent futures | XGBoost | 5 | 1.004 | 0.999 | 1.000 | 1.005 | 1.009 | 0.999 | 1 of 4 |
| Brent futures | XGBoost | 20 | 1.004 | 1.000 | 0.999 | 1.004 | 1.004 | 0.998 | 1 of 4 |
| Brent futures | XGBoost | 60 | 1.001 | 0.998 | 0.998 | 1.001 | 1.005 | 0.992 | 3 of 4 |
| Brent futures | ARMA-GARCH | 1 | 1.000 | 1.000 | 1.001 | 0.999 | 1.003 | 0.998 | 2 of 4 |
| Brent futures | ARMA-GARCH | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | ARMA-GARCH | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 3 of 4 |
| Brent futures | ARMA-GARCH | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | MS-AR | 1 | 1.002 | 1.000 | 0.997 | 1.003 | 1.007 | 0.995 | 3 of 4 |
| Brent futures | MS-AR | 5 | 1.007 | 1.001 | 1.001 | 1.007 | 1.015 | 0.999 | 2 of 4 |
| Brent futures | MS-AR | 20 | 1.002 | 1.014 | 1.008 | 1.003 | 1.026 | 0.997 | 1 of 4 |
| Brent futures | MS-AR | 60 | 0.988 | 0.998 | 1.000 | 0.988 | 1.037 | 0.981 | 2 of 4 |
| Brent futures | MS-ARX | 1 | 1.001 | 1.004 | 0.996 | 1.003 | 1.004 | 0.993 | 2 of 4 |
| Brent futures | MS-ARX | 5 | 1.009 | 1.001 | 1.002 | 1.009 | 1.016 | 0.999 | 1 of 4 |
| Brent futures | MS-ARX | 20 | 1.002 | 1.014 | 1.009 | 1.002 | 1.031 | 0.998 | 1 of 4 |
| Brent futures | MS-ARX | 60 | 0.987 | 0.998 | 1.001 | 0.987 | 1.046 | 0.980 | 2 of 4 |
| Brent futures | Historical mean | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | Historical mean | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Historical mean | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 2 of 4 |
| Brent futures | Historical mean | 60 | 1.001 | 1.000 | 1.000 | 1.001 | 1.002 | 0.993 | 3 of 4 |
| Brent futures | AR (direct) | 1 | 0.999 | 1.000 | 1.001 | 0.999 | 1.003 | 0.997 | 2 of 4 |
| Brent futures | AR (direct) | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | AR (direct) | 20 | 1.000 | 1.001 | 1.001 | 1.000 | 1.002 | 1.000 | 1 of 4 |
| Brent futures | AR (direct) | 60 | 1.002 | 1.000 | 1.001 | 1.001 | 1.003 | 0.998 | 1 of 4 |
| WTI futures | Regime LASSO | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Regime LASSO | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Regime LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Regime LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | XGBoost | 1 | 1.000 | 0.994 | 0.998 | 0.999 | 1.003 | 0.992 | 2 of 4 |
| WTI futures | XGBoost | 5 | 1.014 | 0.981 | 0.989 | 1.014 | 1.016 | 0.985 | 2 of 4 |
| WTI futures | XGBoost | 20 | 1.018 | 1.047 | 1.008 | 1.024 | 1.114 | 0.964 | 1 of 4 |
| WTI futures | XGBoost | 60 | 1.001 | 1.000 | 1.000 | 1.000 | 1.004 | 0.999 | 3 of 4 |
| WTI futures | ARMA-GARCH | 1 | 0.999 | 1.002 | 1.004 | 0.998 | 1.006 | 0.996 | 1 of 4 |
| WTI futures | ARMA-GARCH | 5 | 1.002 | 0.999 | 0.999 | 1.002 | 1.003 | 0.995 | 1 of 4 |
| WTI futures | ARMA-GARCH | 20 | 1.000 | 0.998 | 0.998 | 1.000 | 1.001 | 0.995 | 3 of 4 |
| WTI futures | ARMA-GARCH | 60 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 0.998 | 3 of 4 |
| WTI futures | MS-AR | 1 | 1.000 | 1.001 | 1.001 | 1.000 | 1.002 | 0.997 | 2 of 4 |
| WTI futures | MS-AR | 5 | 0.997 | 1.009 | 1.008 | 0.997 | 1.022 | 0.992 | 2 of 4 |
| WTI futures | MS-AR | 20 | 0.997 | 1.122 | 1.098 | 0.995 | 1.181 | 0.992 | 2 of 4 |
| WTI futures | MS-AR | 60 | 0.989 | 0.994 | 1.019 | 0.977 | 1.266 | 0.949 | 2 of 4 |
| WTI futures | MS-ARX | 1 | 1.002 | 1.001 | 1.000 | 1.002 | 1.004 | 0.996 | 2 of 4 |
| WTI futures | MS-ARX | 5 | 0.999 | 1.009 | 1.009 | 0.999 | 1.024 | 0.994 | 2 of 4 |
| WTI futures | MS-ARX | 20 | 0.997 | 1.122 | 1.097 | 0.996 | 1.183 | 0.993 | 2 of 4 |
| WTI futures | MS-ARX | 60 | 0.989 | 0.994 | 1.019 | 0.977 | 1.268 | 0.949 | 2 of 4 |
| WTI futures | Historical mean | 1 | 1.000 | 1.000 | 1.002 | 0.999 | 1.004 | 0.998 | 1 of 4 |
| WTI futures | Historical mean | 5 | 0.997 | 1.004 | 1.008 | 0.995 | 1.017 | 0.991 | 1 of 4 |
| WTI futures | Historical mean | 20 | 0.986 | 1.068 | 1.054 | 0.985 | 1.137 | 0.968 | 1 of 4 |
| WTI futures | Historical mean | 60 | 0.978 | 0.996 | 1.018 | 0.970 | 1.287 | 0.934 | 1 of 4 |
| WTI futures | AR (direct) | 1 | 1.000 | 1.000 | 1.002 | 0.999 | 1.003 | 0.999 | 2 of 4 |
| WTI futures | AR (direct) | 5 | 1.009 | 1.016 | 0.978 | 1.021 | 1.046 | 0.940 | 3 of 4 |
| WTI futures | AR (direct) | 20 | 0.987 | 1.067 | 1.053 | 0.985 | 1.132 | 0.970 | 1 of 4 |
| WTI futures | AR (direct) | 60 | 0.976 | 0.996 | 1.017 | 0.969 | 1.285 | 0.935 | 1 of 4 |

