# Out-of-sample performance summary: oil price forecasting

**Test design.** Models were estimated only on data up to 2024-12-31 and then applied day by day from 2025-01-02 to 2026-10-16 (451 forecast dates), with no refitting. Forecasts are expected cumulative log returns, converted to prices. Benchmarks: no change (random walk), historical mean, direct AR. RMSE vs no change below 1 means better than the no-change forecast; DM p is the Diebold-Mariano p-value against no change (HAC, h-1 lags); band coverage is the share of outcomes inside the model's +-1 sd band (68% if well calibrated). Window and weight choices come from the notebooks' own 2022-2023 validation; some design decisions were made after seeing later data, so the test is clean for estimation but not pristine.

## Accuracy by horizon

| Series | Model | Type | Test period | RMSE vs no change h=1 | MAPE % h=1 | Direction % h=1 | DM p h=1 | Band coverage % h=1 | F1 up h=1 | Forecast dates h=1 | RMSE vs no change h=5 | MAPE % h=5 | Direction % h=5 | DM p h=5 | Band coverage % h=5 | F1 up h=5 | RMSE vs no change h=20 | MAPE % h=20 | Direction % h=20 | DM p h=20 | Band coverage % h=20 | F1 up h=20 | RMSE vs no change h=60 | MAPE % h=60 | Direction % h=60 | DM p h=60 | Band coverage % h=60 | F1 up h=60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Brent futures | No change | benchmark | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 |  |  |  |  | 450 | 1.000 | 4.58 |  |  |  |  | 1.000 | 8.50 |  |  |  |  | 1.000 | 15.79 |  |  |  |  |
| Brent futures | LASSO | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 54 | 1.00 | 70 | 0.65 | 450 | 1.000 | 4.58 |  | 1.00 | 68 |  | 1.000 | 8.50 |  | 1.00 | 69 |  | 1.000 | 15.79 | 54 | 1.00 | 64 | 0.45 |
| Brent futures | LASSO 1-SE | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 54 | 1.00 | 70 | 0.65 | 450 | 1.000 | 4.58 |  | 1.00 | 68 |  | 1.000 | 8.50 |  | 1.00 | 69 |  | 1.000 | 15.79 | 54 | 1.00 | 64 | 0.45 |
| Brent futures | PCA reg. | model | 2025-01-02 to 2026-10-16 | 0.999 | 2.01 | 51 | 0.92 | 68 | 0.54 | 450 | 1.034 | 4.72 | 55 | 0.29 | 66 | 0.57 | 1.006 | 8.58 | 57 | 0.87 | 72 | 0.59 | 1.001 | 15.52 | 59 | 0.96 | 71 | 0.57 |
| Brent futures | Targeted PCA | model | 2025-01-02 to 2026-10-16 | 1.001 | 1.99 | 56 | 0.86 | 68 | 0.61 | 450 | 1.054 | 4.77 | 51 | 0.08 | 65 | 0.54 | 1.037 | 8.80 | 53 | 0.38 | 69 | 0.58 | 1.083 | 18.35 | 60 | 0.43 | 60 | 0.67 |
| Brent futures | LASSO regime | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 54 | 1.00 | 70 | 0.53 | 450 | 1.000 | 4.58 |  | 1.00 | 68 |  | 1.000 | 8.50 |  | 1.00 | 69 |  | 1.000 | 15.79 | 54 | 1.00 | 64 | 0.45 |
| Brent futures | Combo (equal wt) | model | 2025-01-02 to 2026-10-16 | 0.998 | 1.99 | 53 | 0.37 | 70 | 0.58 | 450 | 1.003 | 4.58 | 52 | 0.64 | 68 | 0.54 | 0.999 | 8.45 | 56 | 0.91 | 70 | 0.60 | 1.003 | 15.87 | 63 | 0.86 | 65 | 0.68 |
| Brent futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 51 | 0.96 | 72 | 0.52 | 450 | 1.000 | 4.58 | 48 | 0.55 | 70 | 0.48 | 1.000 | 8.50 | 53 | 1.00 | 71 | 0.51 | 1.000 | 15.79 | 51 | 0.83 | 70 | 0.50 |
| Brent futures | MS-AR | model | 2025-01-02 to 2026-10-16 | 1.004 | 2.01 | 48 | 0.51 | 76 | 0.55 | 450 | 1.014 | 4.64 | 48 | 0.27 | 76 | 0.59 | 1.012 | 8.68 | 48 | 0.31 | 83 | 0.58 | 0.978 | 15.56 | 55 | 0.30 | 76 | 0.64 |
| Brent futures | MS-ARX | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 46 | 0.88 | 72 | 0.53 | 450 | 1.005 | 4.60 | 43 | 0.23 | 67 | 0.53 | 1.002 | 8.51 | 43 | 0.76 | 71 | 0.53 | 0.997 | 15.90 | 46 | 0.82 | 65 | 0.57 |
| Brent futures | Historical mean | benchmark | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 54 | 0.55 |  | 0.70 | 450 | 1.000 | 4.58 | 51 | 0.50 |  | 0.67 | 1.000 | 8.50 | 49 | 0.57 |  | 0.66 | 1.001 | 15.78 | 52 | 0.61 |  |  |
| Brent futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-16 | 1.000 | 2.00 | 50 | 0.91 |  | 0.51 | 450 | 1.000 | 4.58 | 53 | 0.46 |  | 0.67 | 1.000 | 8.51 | 49 | 0.81 |  | 0.55 | 1.001 | 15.78 | 49 | 0.25 |  | 0.44 |
| WTI futures | No change | benchmark | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 |  |  |  |  | 450 | 1.000 | 4.82 |  |  |  |  | 1.000 | 8.55 |  |  |  |  | 1.000 | 16.07 |  |  |  |  |
| WTI futures | LASSO | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 | 51 | 0.79 | 71 | 0.55 | 450 | 1.000 | 4.82 |  | 1.00 | 69 |  | 1.000 | 8.55 | 54 | 1.00 | 70 |  | 1.000 | 16.07 |  | 1.00 | 65 |  |
| WTI futures | LASSO 1-SE | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 | 51 | 1.00 | 71 | 0.55 | 450 | 1.000 | 4.82 |  | 1.00 | 69 |  | 1.000 | 8.55 | 54 | 1.00 | 70 |  | 1.000 | 16.07 |  | 1.00 | 65 |  |
| WTI futures | PCA reg. | model | 2025-01-02 to 2026-10-16 | 1.015 | 2.10 | 53 | 0.15 | 71 | 0.50 | 450 | 1.021 | 4.86 | 57 | 0.37 | 67 | 0.56 | 1.017 | 8.71 | 50 | 0.33 | 69 | 0.44 | 0.961 | 16.01 | 57 | 0.45 | 62 | 0.62 |
| WTI futures | Targeted PCA | model | 2025-01-02 to 2026-10-16 | 1.013 | 2.09 | 55 | 0.26 | 71 | 0.53 | 450 | 1.000 | 4.86 | 54 | 0.99 | 68 | 0.54 | 1.002 | 8.79 | 47 | 0.95 | 68 | 0.47 | 0.976 | 16.78 | 55 | 0.62 | 62 | 0.61 |
| WTI futures | LASSO regime | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 | 51 | 0.79 | 71 | 0.55 | 450 | 1.000 | 4.82 |  | 1.00 | 69 |  | 1.000 | 8.55 | 54 | 1.00 | 70 |  | 1.000 | 16.07 |  | 1.00 | 65 |  |
| WTI futures | Combo (equal wt) | model | 2025-01-02 to 2026-10-16 | 1.001 | 2.07 | 54 | 0.71 | 72 | 0.52 | 450 | 0.995 | 4.78 | 58 | 0.46 | 70 | 0.57 | 0.997 | 8.54 | 52 | 0.69 | 71 | 0.48 | 0.978 | 15.86 | 57 | 0.17 | 65 | 0.62 |
| WTI futures | ARMA-GARCH | model | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 | 51 | 0.99 | 72 | 0.50 | 450 | 1.001 | 4.82 | 50 | 0.57 | 68 | 0.49 | 1.000 | 8.55 | 54 | 0.88 | 74 | 0.50 | 1.000 | 16.07 | 51 | 0.49 | 71 | 0.48 |
| WTI futures | MS-AR | model | 2025-01-02 to 2026-10-16 | 1.001 | 2.08 | 48 | 0.75 | 67 | 0.52 | 450 | 1.004 | 4.86 | 50 | 0.69 | 64 | 0.52 | 1.031 | 9.34 | 43 | 0.13 | 69 | 0.46 | 0.993 | 17.74 | 49 | 0.92 | 64 | 0.59 |
| WTI futures | MS-ARX | model | 2025-01-02 to 2026-10-16 | 1.001 | 2.07 | 47 | 0.49 | 72 | 0.50 | 450 | 1.005 | 4.84 | 41 | 0.01 | 72 | 0.49 | 1.004 | 8.61 | 41 | 0.13 | 78 | 0.50 | 1.001 | 16.13 | 45 | 0.82 | 72 | 0.55 |
| WTI futures | Historical mean | benchmark | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 | 53 | 0.75 |  | 0.69 | 450 | 0.999 | 4.84 | 50 | 0.77 |  | 0.66 | 0.997 | 8.76 | 46 | 0.84 |  | 0.63 | 0.989 | 16.96 | 46 | 0.80 |  | 0.63 |
| WTI futures | AR (direct) | benchmark | 2025-01-02 to 2026-10-16 | 1.000 | 2.07 | 52 | 0.71 |  | 0.69 | 450 | 1.009 | 4.86 | 52 | 0.64 |  | 0.57 | 0.997 | 8.76 | 46 | 0.85 |  | 0.63 | 0.989 | 16.96 | 46 | 0.79 |  | 0.63 |


## Robustness (RMSE / RMSE of no change; below 1 is better)

| Series | Model | Horizon (days) | High-volatility RMSE ratio | Low-volatility RMSE ratio | 2025 RMSE ratio | 2026 RMSE ratio | Worst window RMSE ratio | Best window RMSE ratio | Windows better than no change |
|---|---|---|---|---|---|---|---|---|---|
| Brent futures | LASSO | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | LASSO | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | LASSO 1-SE | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 of 4 |
| Brent futures | LASSO 1-SE | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO 1-SE | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO 1-SE | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | PCA reg. | 1 | 1.000 | 0.995 | 0.993 | 1.001 | 1.008 | 0.981 | 2 of 4 |
| Brent futures | PCA reg. | 5 | 1.050 | 0.967 | 0.949 | 1.064 | 1.089 | 0.884 | 2 of 4 |
| Brent futures | PCA reg. | 20 | 1.009 | 0.981 | 0.971 | 1.011 | 1.045 | 0.864 | 1 of 4 |
| Brent futures | PCA reg. | 60 | 1.011 | 0.990 | 0.973 | 1.016 | 1.014 | 0.872 | 2 of 4 |
| Brent futures | Targeted PCA | 1 | 1.003 | 0.988 | 0.996 | 1.003 | 1.018 | 0.973 | 2 of 4 |
| Brent futures | Targeted PCA | 5 | 1.068 | 0.999 | 0.985 | 1.078 | 1.134 | 0.974 | 2 of 4 |
| Brent futures | Targeted PCA | 20 | 1.038 | 1.032 | 1.029 | 1.038 | 1.108 | 0.996 | 1 of 4 |
| Brent futures | Targeted PCA | 60 | 1.149 | 0.996 | 1.011 | 1.120 | 1.367 | 0.944 | 1 of 4 |
| Brent futures | LASSO regime | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | LASSO regime | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO regime | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| Brent futures | LASSO regime | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Combo (equal wt) | 1 | 0.998 | 0.995 | 0.995 | 0.998 | 1.003 | 0.990 | 3 of 4 |
| Brent futures | Combo (equal wt) | 5 | 1.007 | 0.989 | 0.982 | 1.011 | 1.025 | 0.969 | 3 of 4 |
| Brent futures | Combo (equal wt) | 20 | 1.000 | 0.990 | 0.979 | 1.001 | 1.009 | 0.956 | 2 of 4 |
| Brent futures | Combo (equal wt) | 60 | 1.011 | 0.994 | 0.984 | 1.013 | 1.049 | 0.946 | 3 of 4 |
| Brent futures | ARMA-GARCH | 1 | 1.000 | 1.000 | 1.001 | 1.000 | 1.004 | 0.997 | 2 of 4 |
| Brent futures | ARMA-GARCH | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 0.999 | 1 of 4 |
| Brent futures | ARMA-GARCH | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 3 of 4 |
| Brent futures | ARMA-GARCH | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | MS-AR | 1 | 1.004 | 1.001 | 0.996 | 1.006 | 1.016 | 0.988 | 3 of 4 |
| Brent futures | MS-AR | 5 | 1.017 | 1.004 | 1.005 | 1.018 | 1.033 | 0.996 | 1 of 4 |
| Brent futures | MS-AR | 20 | 1.009 | 1.034 | 1.025 | 1.009 | 1.058 | 1.007 | 0 of 4 |
| Brent futures | MS-AR | 60 | 0.965 | 0.995 | 1.002 | 0.965 | 1.095 | 0.939 | 2 of 4 |
| Brent futures | MS-ARX | 1 | 1.000 | 1.000 | 0.999 | 1.000 | 1.001 | 0.997 | 2 of 4 |
| Brent futures | MS-ARX | 5 | 1.008 | 0.996 | 1.001 | 1.007 | 1.012 | 0.993 | 1 of 4 |
| Brent futures | MS-ARX | 20 | 1.001 | 1.015 | 1.021 | 0.999 | 1.065 | 0.997 | 1 of 4 |
| Brent futures | MS-ARX | 60 | 0.990 | 1.005 | 1.018 | 0.985 | 1.226 | 0.949 | 2 of 4 |
| Brent futures | Historical mean | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Historical mean | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| Brent futures | Historical mean | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 | 1.000 | 1 of 4 |
| Brent futures | Historical mean | 60 | 1.001 | 1.000 | 1.000 | 1.001 | 1.002 | 0.993 | 3 of 4 |
| Brent futures | AR (direct) | 1 | 1.000 | 1.000 | 1.001 | 0.999 | 1.003 | 0.997 | 1 of 4 |
| Brent futures | AR (direct) | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3 of 4 |
| Brent futures | AR (direct) | 20 | 1.000 | 1.001 | 1.001 | 1.000 | 1.002 | 1.000 | 1 of 4 |
| Brent futures | AR (direct) | 60 | 1.002 | 1.000 | 1.001 | 1.001 | 1.002 | 0.998 | 2 of 4 |
| WTI futures | LASSO | 1 | 1.000 | 0.999 | 0.998 | 1.000 | 1.002 | 0.996 | 2 of 4 |
| WTI futures | LASSO | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | LASSO | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| WTI futures | LASSO | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | LASSO 1-SE | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | LASSO 1-SE | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | LASSO 1-SE | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| WTI futures | LASSO 1-SE | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | PCA reg. | 1 | 1.020 | 0.987 | 0.998 | 1.020 | 1.030 | 0.988 | 1 of 4 |
| WTI futures | PCA reg. | 5 | 1.024 | 1.012 | 0.990 | 1.033 | 1.059 | 0.902 | 1 of 4 |
| WTI futures | PCA reg. | 20 | 1.014 | 1.040 | 1.073 | 1.006 | 1.109 | 0.945 | 2 of 4 |
| WTI futures | PCA reg. | 60 | 0.937 | 0.988 | 1.010 | 0.935 | 1.169 | 0.929 | 2 of 4 |
| WTI futures | Targeted PCA | 1 | 1.016 | 0.997 | 1.007 | 1.014 | 1.032 | 0.982 | 1 of 4 |
| WTI futures | Targeted PCA | 5 | 0.989 | 1.042 | 0.994 | 1.002 | 1.043 | 0.916 | 2 of 4 |
| WTI futures | Targeted PCA | 20 | 0.992 | 1.073 | 1.113 | 0.980 | 1.130 | 0.974 | 3 of 4 |
| WTI futures | Targeted PCA | 60 | 0.961 | 0.992 | 1.046 | 0.937 | 1.374 | 0.921 | 2 of 4 |
| WTI futures | LASSO regime | 1 | 1.000 | 0.999 | 0.998 | 1.000 | 1.002 | 0.996 | 2 of 4 |
| WTI futures | LASSO regime | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | LASSO regime | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2 of 4 |
| WTI futures | LASSO regime | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 of 4 |
| WTI futures | Combo (equal wt) | 1 | 1.003 | 0.993 | 0.996 | 1.003 | 1.003 | 0.990 | 1 of 4 |
| WTI futures | Combo (equal wt) | 5 | 0.993 | 1.001 | 0.988 | 0.998 | 1.007 | 0.959 | 2 of 4 |
| WTI futures | Combo (equal wt) | 20 | 0.996 | 1.008 | 1.019 | 0.993 | 1.031 | 0.968 | 3 of 4 |
| WTI futures | Combo (equal wt) | 60 | 0.969 | 0.988 | 0.991 | 0.971 | 1.042 | 0.960 | 2 of 4 |
| WTI futures | ARMA-GARCH | 1 | 1.000 | 1.002 | 1.004 | 0.999 | 1.005 | 0.996 | 1 of 4 |
| WTI futures | ARMA-GARCH | 5 | 1.002 | 0.999 | 0.999 | 1.002 | 1.003 | 0.993 | 1 of 4 |
| WTI futures | ARMA-GARCH | 20 | 1.000 | 0.998 | 0.998 | 1.000 | 1.002 | 0.995 | 3 of 4 |
| WTI futures | ARMA-GARCH | 60 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 0.998 | 3 of 4 |
| WTI futures | MS-AR | 1 | 1.000 | 1.007 | 1.004 | 1.000 | 1.008 | 0.995 | 2 of 4 |
| WTI futures | MS-AR | 5 | 0.996 | 1.032 | 1.027 | 0.995 | 1.038 | 0.987 | 1 of 4 |
| WTI futures | MS-AR | 20 | 0.994 | 1.284 | 1.230 | 0.990 | 1.383 | 0.984 | 2 of 4 |
| WTI futures | MS-AR | 60 | 0.979 | 1.008 | 1.065 | 0.953 | 1.560 | 0.902 | 2 of 4 |
| WTI futures | MS-ARX | 1 | 1.001 | 1.000 | 1.000 | 1.001 | 1.002 | 0.999 | 2 of 4 |
| WTI futures | MS-ARX | 5 | 1.007 | 0.999 | 1.001 | 1.007 | 1.009 | 0.997 | 1 of 4 |
| WTI futures | MS-ARX | 20 | 1.003 | 1.010 | 1.009 | 1.003 | 1.035 | 1.002 | 0 of 4 |
| WTI futures | MS-ARX | 60 | 1.000 | 1.003 | 1.009 | 0.997 | 1.093 | 0.984 | 2 of 4 |
| WTI futures | Historical mean | 1 | 1.000 | 1.000 | 1.002 | 0.999 | 1.003 | 0.999 | 2 of 4 |
| WTI futures | Historical mean | 5 | 0.998 | 1.004 | 1.008 | 0.996 | 1.018 | 0.991 | 1 of 4 |
| WTI futures | Historical mean | 20 | 0.988 | 1.068 | 1.054 | 0.986 | 1.129 | 0.969 | 1 of 4 |
| WTI futures | Historical mean | 60 | 0.983 | 0.996 | 1.018 | 0.974 | 1.291 | 0.934 | 1 of 4 |
| WTI futures | AR (direct) | 1 | 0.999 | 1.000 | 1.002 | 0.999 | 1.003 | 0.999 | 2 of 4 |
| WTI futures | AR (direct) | 5 | 1.008 | 1.016 | 0.978 | 1.021 | 1.032 | 0.920 | 1 of 4 |
| WTI futures | AR (direct) | 20 | 0.988 | 1.067 | 1.053 | 0.987 | 1.124 | 0.972 | 1 of 4 |
| WTI futures | AR (direct) | 60 | 0.982 | 0.996 | 1.017 | 0.974 | 1.290 | 0.935 | 1 of 4 |

