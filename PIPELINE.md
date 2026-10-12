# Automated forecasting pipeline

```
Every weekday (GitHub Actions, daily_update.yml)
  1. scripts/update_data.py   download last 90 days of every series, merge into Data/raw/*.csv
  2. scripts/run_daily.py     load saved models, forecast all horizons, append Outputs/forecast_log.csv
  3. commit                   data CSVs + forecast log
  4. scripts/send_email.py    forecast table, chart, last-7-days actual vs forecast, track record

Every quarter (quarterly_retrain.yml)
  scripts/update_data.py -> scripts/train_models.py -> commit models/ -> scripts/send_retrain_email.py
```

**Both workflows can also be started by hand:** GitHub > Actions > pick the workflow > *Run workflow*.
The daily one can skip the email or backfill the log; the retrain one can limit the plugins or skip the email.

**Emails** go to the five NUS addresses listed in `forecast/mailer.py` (cc yja210199@gmail.com); edit `DEFAULT_TO` there to change the list.
The retrain email lists the new training window, which variables/penalties/factor counts changed, how the
forecasts changed (old vs new model on the same data) and the regime-model parameters.

Models are **not** retrained daily: `models/<plugin>.joblib` holds the fitted models; each day only
the new data is fed through them (features, regime filter, prediction). A quarterly retrain re-selects
hyper-parameters (LASSO penalty, number of PCA factors, HMM parameters) on past data only.
Refit frequency was tested on 2016-2026 (refit every 21 / 63 / 252 days): 63 vs 21 days changed RMSE by at most
1.8% with no systematic pattern, so quarterly is enough.

## Folders
| Path | Content |
|---|---|
| `Data/raw/` | `yahoo_daily`, `yahoo_extra_daily`, `fred_daily`, `weekly_raw`, `eia_wti_futures` CSVs |
| `forecast/` | package: `data`, `features`, `regime` (HMM), `models`, `registry`, `plugins/`, `reporting` |
| `models/` | saved models (`*.joblib`), readable summaries (`*_meta.json`), `training_history.csv` |
| `Outputs/` | `latest_forecasts.csv`, `forecast_log.csv` (every forecast ever made) |
| `scripts/` | entry points: `update_data`, `run_daily`, `send_email`, `train_models`, `send_retrain_email` |

## Models in the pipeline

Model families are fitted separately and share input data and pipeline utilities.
Forecast combinations are explicitly defined within a family, such as the LASSO
equal-weight combination.

| Plugin | Models | Series | Notes |
|---|---|---|---|
| `lasso_family` | No change, LASSO, LASSO 1-SE, PCA, targeted PCA, regime LASSO, equal-weight combo | `cfg.target` (Brent) | from `lasso.ipynb`; HMM regime features |
| `arma_garch` | ARMA-GARCH (MA(2) mean, GARCH(1,1)-t) | Brent and WTI futures | from `gp_arma_garch.ipynb`; forecasts about no change, its value is the volatility band (all horizons) |

## MS-AR research notebook

`msar.ipynb` sits alongside `gp_arma_garch.ipynb` and `lasso.ipynb`. It contains all
MS-AR/MS-ARX model code, configuration, backtesting, numerical checks, tables and
plots. Open it from `OilForecastQF` and run all cells. It uses
`Data/raw/yahoo_daily.csv` and saves evidence under `Outputs/msar/baseline/` and
`Outputs/msar/extended/`.

The default run loads existing results and verifies them. Set `RERUN_MODEL = True`
in the setup cell to refit the baseline and follow-up in order after updating the
source CSV. MS-AR is not yet a registered daily forecasting plugin. Its reused
2024+ test comparison remains exploratory. Earlier implementation files are
archived outside the repository at `../tmp/msar_before_notebook_only/`.

## How models share tables and charts
* All models report, for one target series, the expected log return over *h* trading days and a 1-sd
  uncertainty (`sigma`, with `sigma_kind` = `rv20` for recent volatility or `garch`).
* The e-mail has **one panel per target series** (prices of Brent and WTI cannot share an axis).
* Tables: price per model and horizon with the change vs "No change" (grey when about zero), the +-1 sd
  price range per model, last-7-days actual vs forecast, track record (RMSE vs no change; share of
  outcomes inside the +-1 sd band).
* Chart: models whose forecast equals no change are merged into one line; model-based bands (GARCH) are
  shaded and the recent-volatility band is dotted. "No change" is added from the price data when no
  plugin supplies it.

## Adding a model (ARIMA, GARCH, ML, ...)
1. Create `forecast/plugins/<name>.py` with a class derived from `ForecastPlugin` (see `forecast/registry.py`):
   `train(inputs, cfg)`, `predict(artifact, inputs, cfg)`, `backtest(inputs, cfg, n_days, refit_every)`.
   `predict` returns one row per (model, horizon) with columns
   `asof_date, model, horizon, forecast_return, sigma` and optionally `target` (series name, default
   `cfg.target`) and `sigma_kind`.
2. Register it: `register(MyPlugin())` at the bottom of the file, and import it in `forecast/plugins/__init__.py`.
3. Run `python scripts/train_models.py --plugins <name>` once, then `python scripts/run_daily.py --backfill 30`.

The daily run, the forecast log and the e-mail pick it up automatically: the e-mail builds its columns
from whatever models are in the log. Put display names in `forecast/reporting.py` (`LABELS`).

## Secrets (GitHub: Settings > Secrets and variables > Actions)
`GMAIL_USER` and `GMAIL_APP_PASSWORD` (a Gmail app password). Recipients are set in `daily_update.yml`.

## Running locally
```bash
python scripts/update_data.py
python scripts/train_models.py            # first time, or to retrain
python scripts/run_daily.py --backfill 30 # --backfill seeds the log with past forecasts
python scripts/send_email.py --dry-run    # writes email_preview/, sends nothing
```

## Known limits
* The `lasso_family` target is `brent_fut`, an unadjusted continuous front-month futures series (contract rolls appear as jumps).
* EIA's futures-contract files stop in April 2024, so the curve-slope feature is unavailable.
* Weekly crude stocks come from EIA (the FRED series is no longer served).
