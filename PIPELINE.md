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
| `lasso_family` | No change, LASSO, LASSO 1-SE, PCA, targeted PCA, regime LASSO, equal-weight combo | Brent and WTI futures | from `lasso.ipynb`; HMM regime features; fitted separately per series |
| `arma_garch` | ARMA-GARCH (MA(2) mean, GARCH(1,1)-t) | Brent and WTI futures | from `gp_arma_garch.ipynb`; forecasts about no change, its value is the volatility band (all horizons) |
| `msar` | MS-AR, MS-ARX (2-regime Markov-switching AR; MS-ARX adds the other benchmark's lagged price change, DXY return and Brent-WTI spread; forecast scaled by 0.25) | Brent and WTI futures | from `msar.ipynb`; needs statsmodels >= 0.15; band = simulated sd |

## MS-AR research notebook

`msar.ipynb` sits alongside `gp_arma_garch.ipynb` and `lasso.ipynb`. It contains all
MS-AR/MS-ARX model code, configuration, backtesting, numerical checks, tables and
plots. Open it from `OilForecastQF` and run all cells. It uses
`Data/raw/yahoo_daily.csv` and saves evidence under `Outputs/msar/baseline/` and
`Outputs/msar/extended/`.

The default run loads existing results and verifies them. Set `RERUN_MODEL = True`
in the setup cell to refit the baseline and follow-up in order after updating the
source CSV. MS-AR and MS-ARX **are** registered as the `msar` plugin (`forecast/plugins/msar.py`, a port of the
notebook's estimation, exact-mean recursion and simulation; its 2026-10-09 forecasts match the notebook's saved
results to three decimals). The notebook's window and mean-weight choices were selected on 2022-2023 validation and are
fixed in `forecast/config.py` (`msar_models`); the quarterly retrain only re-estimates the regime and AR parameters.
Re-selecting the specification still means re-running the notebook with `RERUN_MODEL = True`. The notebook's own
2024+ test comparison (reused, exploratory) found both models no better than no change. Earlier implementation files are
archived outside the repository at `../tmp/msar_before_notebook_only/`.

## Frozen-model evaluation and economic value (`scripts/frozen_eval.py`)
Answers: "how would these models have done if they stopped learning on a fixed date?" (config: `frozen_cutoff`,
default 2024-12-31).
* Every plugin is trained **only on data up to the cutoff**; from the next day on, the frozen parameters are applied
  daily to the data known on each date (features and regime filter continue, nothing is refitted).
* Forecasts since the cutoff are scored against outcomes: RMSE vs no change, direction, +-1 sd band coverage,
  Diebold-Mariano p-value (HAC, h-1 lags).
* **Economic value:** a simple rule turns each forecast into a position: long if the forecast return is above
  `trade_band` x the 1-sd of the h-day return, short if below minus that, otherwise flat. Overlapping tranches (each day
  opens 1/h of capital and holds it h days), `cost_bps` per unit of turnover. Reported: annualised return, volatility,
  Sharpe, max drawdown, turnover, costs paid, days in market, against buy and hold.
* Evaluated horizons: 1, 2, 3, 5, 10, 20, 40, 60 days (tables). The charts are short-horizon and rolling: a
  `frozen_chart_horizon`-day (3) forecast is made every day and drawn against what happened, with the ARMA-GARCH band
  and how often the realised move fell inside it.
* Also scored against two benchmarks that are NOT production models: historical mean and a direct AR (both estimated on
  data up to the cutoff). Metrics per model and horizon: RMSE / MAE / MAPE, direction hit-rate, precision / recall / F1,
  Diebold-Mariano p-value, +-1 sd band coverage; robustness by volatility regime (threshold = median trailing
  volatility up to the cutoff), by calendar year and by four consecutive windows of the test period.
* **Project outputs** (`Outputs/frozen_eval/`): `report.html` (standalone report with every table and chart),
  `registration_accuracy.csv` / `registration_robustness.csv` / `registration_summary.md` (spreadsheet-ready), plus the raw
  `metrics.csv`, `trading.csv`, `robustness.csv`. The e-mail keeps its shorter "Frozen-model check" section (rolling
  chart, accuracy, direction, band coverage, trading rule); MAPE, F1, benchmarks and the robustness tables are in the
  report only. The signal summary also reads its track record from `metrics.csv`.

## Signal summary (top of the e-mail, `forecast/signals.py`)
For `signal_horizon` (1) trading day (the horizon where the frozen test shows the best records; skill fades quickly beyond it): BUY / SELL if a model's expected move exceeds `trade_band` x its 1-sd noise
(the same rule as the trading test), else HOLD. **The track record comes from the frozen test, not from the model:**
Significant = direction right significantly more than 50% (one-sided p < 0.05, using n / h effective observations because
h-day returns overlap) AND RMSE below no change; Suggestive = p < 0.10, or hit-rate > 55% with RMSE below no change;
None yet otherwise. The consensus is LEAN BUY / LEAN SELL when at least two models agree and outnumber the other side,
otherwise NEUTRAL.
* Runs on the 1st of each month, when missing, or on demand ("run_frozen_eval" in the manual-run dialog) - about
  1-2 minutes.
* This is a deliberately stale model (skill plus ageing). It is not the production model.
* Caveat for the report: window / weight choices come from the notebooks' own validation (all before 2025), but some
  design decisions were made after seeing later data, so the test is clean for estimation but not pristine.

### Brent and WTI
`cfg.targets` (default Brent and WTI futures) lists the series the LASSO-family and MS-AR plugins forecast; ARMA-GARCH
uses `cfg.arma_garch_targets`. Every model has its own fit per series and the e-mail shows one panel per series.
* The notebooks' settings (LASSO penalties are re-picked by cross-validation, but the window of 1260 days, the MS-AR
  window and the 0.25 mean weight, the HMM and the feature set) were chosen on **Brent**. They are applied to WTI
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

## Known limits
* The `lasso_family` target is `brent_fut`, an unadjusted continuous front-month futures series (contract rolls appear as jumps).
* EIA's futures-contract files stop in April 2024, so the curve-slope feature is unavailable.
* Weekly crude stocks come from EIA (the FRED series is no longer served).
