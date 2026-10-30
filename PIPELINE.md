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
| `lasso_regime` | No change, regime-dependent LASSO (shrunk by half; calm / turbulent interactions) | Brent and WTI futures | from `lasson.ipynb`; replaces the earlier `lasso_family`; band = standardised-error spread x today's volatility |
| `xgboost` | XGBoost (depth-3 trees, PCA variable selection plus P(turbulent) and a regime label, shrunk by half) | Brent and WTI futures | from `xgb_1030.ipynb` (earlier `xgbn.ipynb`); needs `xgboost`; calendar = days both futures trade; band = standardised-error spread x today's volatility |
| `arma_garch` | ARMA-GARCH (MA(2) mean, GARCH(1,1)-t) | Brent and WTI futures | from `gp_arma_garch.ipynb`; forecasts about no change, its value is the volatility band (all horizons) |
| `msar` | MS-AR, MS-ARX (2-regime Markov-switching AR on a common Brent/WTI calendar, 1,260-day window; MS-ARX adds the other benchmark's % return, DXY return and Brent-WTI spread; both forecasts shrunk by half) | Brent and WTI futures | from `msar_common.ipynb` (replaces `msar.ipynb`); needs statsmodels >= 0.15; band = simulated sd |

## MS-AR notebooks

`msar_common.ipynb` is the current specification and is what the `msar` plugin (`forecast/plugins/msar.py`) implements; the plugin's
forecasts match the notebook's own functions to the last digit at the same origin. It uses a common Brent / WTI calendar, a 1,260-day
window for both models, a 0.5 shrink and 2,000 simulated paths (config: `msar_*`). The earlier `msar.ipynb` (63-day refits,
756-day window and 0.25 weight for MS-ARX) is kept for reference only. The quarterly retrain re-estimates the regime and AR
parameters; between retrains the regime filter is run forward with fixed parameters.

### Brent and WTI
`cfg.targets` (default Brent and WTI futures) lists the series the regime LASSO, XGBoost and MS-AR plugins forecast; ARMA-GARCH
uses `cfg.arma_garch_targets`. Every model has its own fit per series and the e-mail shows one panel per series.
* The notebooks' settings (LASSO penalties are re-picked by cross-validation, but the window of 1260 days, the shrink
  weights, the regime model and the feature set) were chosen on **Brent**. They are applied to WTI
  unchanged; they have not been validated for WTI.
* WTI settled at -37.63 on 2020-04-20, which would put a -60% "return" into estimation windows. WTI's own price series is
  therefore used only from `cfg.series_start["wti_fut"]` = 2020-06-01 (about 6 years; shorter windows are accepted down
  to 80% of the nominal window). WTI as a *predictor* of Brent is unaffected. The historical-mean and AR benchmarks use the
  same start.
* For MS-ARX, "the other benchmark" is WTI when forecasting Brent and Brent when forecasting WTI.

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

## Web app (Streamlit)
`app/` is a read-only viewer over files the daily bot already commits (`Data/raw`, `Outputs/`, `models/*_meta.json`); it never downloads data or trains.
Pages: Today (`app/Home.py`), Track record, Explore, About. Light dependencies only (`app/requirements.txt`).

Run locally (in the `cs422` env): `streamlit run app/Home.py`

**Deploy (free, Streamlit Community Cloud):** share.streamlit.io > sign in with GitHub > Create app > repo `OilForecastQF`, branch `main`,
main file `app/Home.py`, Advanced settings > Python 3.11 > Deploy. Then Settings > Sharing > make it public ("anyone with the link").
**Daily refresh:** the bot pushes to `main` every day and Streamlit Cloud redeploys on each push, so new data appears automatically
(the sidebar shows when the data was last updated; "Refresh data" reloads the files). Free apps sleep after about 12 hours without visitors; the
first visit shows a "wake up" button.
Extra files for the app, written by `scripts/frozen_eval.py`: `Outputs/frozen_eval/equity_curves.csv`, `rolling_forecasts.csv`.

## Known limits
* Every plugin's target is an unadjusted continuous front-month futures series (contract rolls appear as jumps).
* EIA's futures-contract files stop in April 2024, so the curve-slope feature is unavailable.
* Weekly crude stocks come from EIA (the FRED series is no longer served).
