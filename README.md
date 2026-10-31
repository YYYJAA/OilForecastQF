# OilForecastQF: automated Brent and WTI price forecasting

NUS QF5210 group project (oil price forecasting option). Every day the pipeline downloads fresh market data, runs saved
models on it, logs the forecasts, and emails the team a report. A web app shows the same results. Models are retrained
once a quarter, never daily.

## What it does

```
Every day (GitHub Actions)                              Every quarter
  1. download the last 90 days of data, merge into CSVs   retrain every model on past data only,
  2. run the saved models, forecast 1 to 60 days ahead    commit the new models,
  3. commit the data and the forecast log                 email what changed
  4. email the forecast report to the team
```

- **Series:** Brent and WTI front-month futures (Yahoo Finance), with FRED macro series and EIA data as predictors.
- **Horizons:** 1, 5, 10, 20, 40 and 60 trading days ahead (the frozen test also scores 2, 3, etc.).
- **Every model is judged against "no change"** (the random walk), plus a historical-mean and a direct AR benchmark.

## Models

| Plugin | Model | Idea |
|---|---|---|
| `lasso_regime` | Regime-dependent LASSO | Shrinks predictors to zero; effects can differ in calm and turbulent markets |
| `xgboost` | XGBoost | Shallow trees on a PCA-reduced predictor set plus regime inputs |
| `arma_garch` | ARMA-GARCH | Near-zero mean; its value is the volatility band |
| `msar` | MS-AR, MS-ARX | Two-regime Markov-switching autoregression, with and without outside inputs |

Each model is a plugin (`train`, `predict`, `backtest`), so adding one means writing one class. The email and the web
app build their columns from whatever models are in the forecast log. Details: [PIPELINE.md](PIPELINE.md).

## How we test

Models are estimated on data up to 2024-12-31 and then applied day by day from 2025-01-02 **with their parameters
frozen**. We report accuracy against no change (RMSE ratio, Diebold-Mariano test), direction hit-rate, how often the
actual outcome falls inside the model's 68% band, robustness by volatility regime and year, and a simple trading rule
with costs. Results are in `Outputs/frozen_eval/` (open `report.html`).

**Honest summary:** no model reliably beats "no change" on price accuracy. The useful, well-calibrated part is the
uncertainty band.

## Web app

`app/` is a read-only [Streamlit](https://streamlit.io) app: today's forecasts and signals, the frozen-test track record,
price history and forecast-vs-actual views. It reads files the daily bot commits, so it never downloads data or trains models.

```bash
streamlit run app/Home.py
```

To deploy for free: share.streamlit.io, repo `OilForecastQF`, branch `main`, main file `app/Home.py`, Python 3.11, then
set sharing to public. It redeploys on every push, so it updates with the daily run.

## Run it yourself

```bash
pip install -r requirements.txt
python scripts/update_data.py             # download and merge the latest data
python scripts/train_models.py            # first time, or to retrain
python scripts/run_daily.py --backfill 30 # forecast today and seed the log with past days
python scripts/send_email.py --dry-run    # writes email_preview/, sends nothing
python scripts/frozen_eval.py             # re-run the frozen-model evaluation
```

For GitHub Actions the repo needs two secrets, `GMAIL_USER` and `GMAIL_APP_PASSWORD` (a Gmail app password).
Both workflows can also be started by hand from the Actions tab.

## Repository layout

| Path | Content |
|---|---|
| `forecast/` | package: data loading, features, regimes, model plugins, reporting |
| `scripts/` | entry points: download, train, daily run, email, frozen evaluation |
| `app/` | Streamlit web app |
| `models/` | saved models (`*.joblib`) and readable summaries |
| `Data/raw/` | downloaded CSVs (updated by the bot) |
| `Outputs/` | `latest_forecasts.csv`, `forecast_log.csv`, and `frozen_eval/` results |
| `.github/workflows/` | the daily and quarterly workflows |
| `*.ipynb` | the team's research notebooks (data download, EDA, and each model family) |
| `PIPELINE.md`, `PROJECT_REQUIREMENTS.md` | pipeline details and the project brief |

## Known limits

- Futures series are unadjusted continuous contracts, so contract rolls show up as jumps in returns.
- The latest day's futures price is provisional until the next day's refresh corrects it.
- WTI futures are used from June 2020 (the price went negative in April 2020).
- Model settings were chosen on Brent and reused for WTI.
- Some design choices were made after seeing data from 2025 onward, so the frozen test is clean for estimation but not pristine.
