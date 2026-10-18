# Out-of-sample performance summary: oil price forecasting

**Test design.** Models were estimated only on data up to 2024-12-31 and then applied day by day from 2025-01-02 to 2026-10-12 (447 forecast dates), with no refitting. Forecasts are expected cumulative log returns, converted to prices. Benchmarks: no change (random walk), historical mean, direct AR. RMSE vs no change below 1 means better than the no-change forecast; DM p is the Diebold-Mariano p-value against no change (HAC, h-1 lags); band coverage is the share of outcomes inside the model's +-1 sd band (68% if well calibrated). Window and weight choices come from the notebooks' own 2022-2023 validation; some design decisions were made after seeing later data, so the test is clean for estimation but not pristine.

## Accuracy by horizon

| Series | Model | Type | Test period | RMSE vs no change h=1 | MAPE % h=1 | Direction % h=1 | DM p h=1 | Band coverage % h=1 | F1 up h=1 | Forecast dates h=1 | RMSE vs no change h=5 | MAPE % h=5 | Direction % h=5 | DM p h=5 | Band coverage % h=5 | F1 up h=5 | RMSE vs no change h=20 | MAPE % h=20 | Direction % h=20 | DM p h=20 | Band coverage % h=20 | F1 up h=20 | RMSE vs no change h=60 | MAPE % h=60 | Direction % h=60 | DM p h=60 | Band coverage % h=60 | F1 up h=60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Brent futures | No change | benchmark | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 |  |  |  |  | 446 | 1.000 | 4.58 |  |  |  |  | 1.000 | 8.56 |  |  |  |  | 1.000 | 15.83 |  |  |  |  |
| Brent futures | LASSO | model | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 | 54 | 1.00 | 70 | 0.65 | 446 | 1.000 | 4.58 |  | 1.00 | 68 |  | 1.000 | 8.56 |  | 1.00 | 68 |  | 1.000 | 15.83 | 53 | 1.00 | 64 | 0.43 |
| Brent futures | LASSO 1-SE | model | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 | 54 | 1.00 | 70 | 0.65 | 446 | 1.000 | 4.58 |  | 1.00 | 68 |  | 1.000 | 8.56 |  | 1.00 | 68 |  | 1.000 | 15.83 | 53 | 1.00 | 64 | 0.43 |
| Brent futures | PCA reg. | model | 2025-01-02 to 2026-10-12 | 0.999 | 2.02 | 51 | 0.91 | 67 | 0.54 | 446 | 1.035 | 4.73 | 55 | 0.28 | 66 | 0.56 | 1.005 | 8.62 | 57 | 0.88 | 71 | 0.58 | 1.000 | 15.51 | 59 | 0.99 | 71 | 0.58 |
| Brent futures | Targeted PCA | model | 2025-01-02 to 2026-10-12 | 1.001 | 2.00 | 56 | 0.84 | 68 | 0.61 | 446 | 1.055 | 4.77 | 51 | 0.08 | 65 | 0.54 | 1.037 | 8.86 | 52 | 0.38 | 69 | 0.57 | 1.085 | 18.46 | 60 | 0.42 | 59 | 0.67 |
| Brent futures | LASSO regime | model | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 | 54 | 1.00 | 70 | 0.53 | 446 | 1.000 | 4.58 |  | 1.00 | 68 |  | 1.000 | 8.56 |  | 1.00 | 68 |  | 1.000 | 15.83 | 53 | 1.00 | 64 | 0.43 |
| Brent futures | Combo (equal wt) | model | 2025-01-02 to 2026-10-12 | 0.998 | 2.00 | 53 | 0.38 | 70 | 0.58 | 446 | 1.004 | 4.59 | 51 | 0.62 | 68 | 0.53 | 0.999 | 8.51 | 56 | 0.91 | 70 | 0.59 | 1.003 | 15.91 | 63 | 0.86 | 65 | 0.67 |
| Brent futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 | 51 | 0.97 | 72 | 0.52 | 446 | 1.000 | 4.58 | 48 | 0.56 | 70 | 0.48 | 1.000 | 8.56 | 52 | 0.99 | 70 | 0.51 | 1.000 | 15.83 | 52 | 0.69 | 70 | 0.50 |
| Brent futures | MS-AR | model | 2025-01-02 to 2026-10-12 | 1.004 | 2.02 | 48 | 0.51 | 76 | 0.55 | 446 | 1.015 | 4.65 | 47 | 0.26 | 76 | 0.58 | 1.012 | 8.74 | 48 | 0.31 | 83 | 0.58 | 0.978 | 15.59 | 55 | 0.30 | 76 | 0.64 |
| Brent futures | MS-ARX | model | 2025-01-02 to 2026-10-12 | 1.000 | 2.01 | 46 | 0.88 | 72 | 0.54 | 446 | 1.004 | 4.60 | 43 | 0.32 | 68 | 0.53 | 1.002 | 8.57 | 43 | 0.76 | 70 | 0.53 | 0.995 | 15.90 | 47 | 0.76 | 65 | 0.57 |
| Brent futures | Historical mean | benchmark | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 | 54 | 0.58 |  | 0.70 | 446 | 1.000 | 4.58 | 50 | 0.55 |  | 0.67 | 1.000 | 8.56 | 49 | 0.57 |  | 0.66 | 1.001 | 15.81 | 52 | 0.62 |  |  |
| Brent futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-12 | 1.000 | 2.00 | 49 | 0.92 |  | 0.50 | 446 | 1.000 | 4.58 | 53 | 0.49 |  | 0.66 | 1.000 | 8.56 | 49 | 0.80 |  | 0.56 | 1.001 | 15.82 | 49 | 0.22 |  | 0.43 |
| WTI futures | No change | benchmark | 2025-01-02 to 2026-10-12 | 1.000 | 2.07 |  |  |  |  | 446 | 1.000 | 4.80 |  |  |  |  | 1.000 | 8.57 |  |  |  |  | 1.000 | 16.13 |  |  |  |  |
| WTI futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-12 | 1.000 | 2.07 | 51 | 0.96 | 72 | 0.50 | 446 | 1.001 | 4.80 | 50 | 0.60 | 69 | 0.49 | 1.000 | 8.57 | 54 | 0.86 | 74 | 0.51 | 1.000 | 16.13 | 51 | 0.44 | 71 | 0.49 |
| WTI futures | Historical mean | benchmark | 2025-01-02 to 2026-10-12 | 1.000 | 2.07 | 53 | 0.72 |  | 0.69 | 446 | 1.000 | 4.80 | 49 | 0.72 |  | 0.66 | 0.999 | 8.61 | 46 | 0.69 |  | 0.63 | 0.996 | 16.31 | 45 | 0.70 |  | 0.63 |
| WTI futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-12 | 0.995 | 2.07 | 49 | 0.48 |  | 0.50 | 446 | 1.001 | 4.81 | 49 | 0.57 |  | 0.56 | 0.999 | 8.61 | 46 | 0.69 |  | 0.63 | 0.996 | 16.31 | 45 | 0.69 |  | 0.63 |


## Robustness (RMSE / RMSE of no change; below 1 is better)

| Series | Model | Horizon (days) | High-volatility RMSE ratio | Low-volatility RMSE ratio | 2025 RMSE ratio | 2026 RMSE ratio | Worst window RMSE ratio | Best window RMSE ratio | Windows better than no change |
|---|---|---|---|---|---|---|---|---|---|
| Brent futures | LASSO | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | LASSO 1-SE | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO 1-SE | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO 1-SE | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO 1-SE | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | PCA reg. | 1 | 0.999 | 0.995 | 0.993 | 1.001 | 1.006 | 0.979 | 2 of 4 |
| Brent futures | PCA reg. | 5 | 1.051 | 0.967 | 0.949 | 1.065 | 1.098 | 0.903 | 2 of 4 |
| Brent futures | PCA reg. | 20 | 1.008 | 0.981 | 0.971 | 1.011 | 1.037 | 0.864 | 1 of 4 |
| Brent futures | PCA reg. | 60 | 1.007 | 0.990 | 0.973 | 1.013 | 1.015 | 0.872 | 2 of 4 |
| Brent futures | Targeted PCA | 1 | 1.004 | 0.988 | 0.996 | 1.003 | 1.019 | 0.976 | 2 of 4 |
| Brent futures | Targeted PCA | 5 | 1.068 | 0.999 | 0.985 | 1.079 | 1.141 | 0.975 | 2 of 4 |
| Brent futures | Targeted PCA | 20 | 1.038 | 1.032 | 1.029 | 1.038 | 1.098 | 0.997 | 1 of 4 |
| Brent futures | Targeted PCA | 60 | 1.152 | 0.996 | 1.011 | 1.122 | 1.366 | 0.946 | 1 of 4 |
| Brent futures | LASSO regime | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO regime | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO regime | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO regime | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Combo (equal wt) | 1 | 0.998 | 0.995 | 0.995 | 0.998 | 1.003 | 0.990 | 3 of 4 |
| Brent futures | Combo (equal wt) | 5 | 1.007 | 0.989 | 0.982 | 1.012 | 1.027 | 0.971 | 3 of 4 |
| Brent futures | Combo (equal wt) | 20 | 1.000 | 0.990 | 0.979 | 1.002 | 1.008 | 0.957 | 3 of 4 |
| Brent futures | Combo (equal wt) | 60 | 1.011 | 0.994 | 0.984 | 1.013 | 1.048 | 0.946 | 3 of 4 |
| Brent futures | ARMA-GARCH | 1 | 1.000 | 1.000 | 1.001 | 1.000 | 1.004 | 0.997 | 1 of 4 |
| Brent futures | ARMA-GARCH | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 0.999 | 1 of 4 |
| Brent futures | ARMA-GARCH | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 3 of 4 |
| Brent futures | ARMA-GARCH | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 3 of 4 |
| Brent futures | MS-AR | 1 | 1.004 | 1.001 | 0.996 | 1.006 | 1.021 | 0.983 | 3 of 4 |
| Brent futures | MS-AR | 5 | 1.017 | 1.004 | 1.005 | 1.019 | 1.043 | 0.994 | 2 of 4 |
| Brent futures | MS-AR | 20 | 1.009 | 1.034 | 1.025 | 1.009 | 1.056 | 1.007 | 0 of 4 |
| Brent futures | MS-AR | 60 | 0.964 | 0.995 | 1.002 | 0.965 | 1.094 | 0.941 | 2 of 4 |
| Brent futures | MS-ARX | 1 | 1.000 | 1.000 | 0.999 | 1.000 | 1.001 | 0.997 | 3 of 4 |
| Brent futures | MS-ARX | 5 | 1.006 | 0.996 | 1.001 | 1.005 | 1.013 | 0.994 | 2 of 4 |
| Brent futures | MS-ARX | 20 | 1.001 | 1.015 | 1.021 | 0.999 | 1.062 | 0.996 | 1 of 4 |
| Brent futures | MS-ARX | 60 | 0.988 | 1.005 | 1.018 | 0.983 | 1.226 | 0.948 | 2 of 4 |
| Brent futures | Historical mean | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | Historical mean | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | Historical mean | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1 of 4 |
| Brent futures | Historical mean | 60 | 1.001 | 1.000 | 1.000 | 1.001 | 1.002 | 0.993 | 3 of 4 |
| Brent futures | AR (direct) | 1 | 1.000 | 1.000 | 1.001 | 0.999 | 1.004 | 0.997 | 1 of 4 |
| Brent futures | AR (direct) | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | AR (direct) | 20 | 1.000 | 1.001 | 1.001 | 1.000 | 1.002 | 1.000 | 1 of 4 |
| Brent futures | AR (direct) | 60 | 1.002 | 1.000 | 1.001 | 1.001 | 1.002 | 0.998 | 1 of 4 |
| WTI futures | ARMA-GARCH | 1 | 1.000 | 1.002 | 1.004 | 0.999 | 1.005 | 0.996 | 1 of 4 |
| WTI futures | ARMA-GARCH | 5 | 1.002 | 0.999 | 0.999 | 1.002 | 1.002 | 0.994 | 1 of 4 |
| WTI futures | ARMA-GARCH | 20 | 1.000 | 0.998 | 0.998 | 1.001 | 1.002 | 0.995 | 3 of 4 |
| WTI futures | ARMA-GARCH | 60 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 0.998 | 3 of 4 |
| WTI futures | Historical mean | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1 of 4 |
| WTI futures | Historical mean | 5 | 1.000 | 1.001 | 1.001 | 0.999 | 1.003 | 0.998 | 1 of 4 |
| WTI futures | Historical mean | 20 | 0.997 | 1.014 | 1.011 | 0.996 | 1.027 | 0.993 | 1 of 4 |
| WTI futures | Historical mean | 60 | 0.995 | 0.997 | 1.001 | 0.993 | 1.068 | 0.984 | 1 of 4 |
| WTI futures | AR (direct) | 1 | 0.993 | 1.007 | 1.013 | 0.989 | 1.017 | 0.978 | 1 of 4 |
| WTI futures | AR (direct) | 5 | 1.001 | 1.000 | 1.001 | 1.001 | 1.002 | 0.999 | 2 of 4 |
| WTI futures | AR (direct) | 20 | 0.997 | 1.014 | 1.011 | 0.996 | 1.027 | 0.993 | 1 of 4 |
| WTI futures | AR (direct) | 60 | 0.994 | 0.997 | 1.001 | 0.993 | 1.067 | 0.984 | 1 of 4 |

