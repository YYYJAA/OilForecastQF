"""Model plugin interface.

A plugin is a family of models that share data preparation and training. Adding a model later
(ARIMA, GARCH, a tree-based ML model, ...) means writing one class with these three methods and
adding it to REGISTRY. The daily run, the forecast log and the e-mail need no other change.

    train(inputs, cfg)                       -> artifact (a picklable dict)  [quarterly]
    predict(artifact, inputs, cfg)           -> DataFrame, one row per (model, horizon)  [daily]
    backtest(inputs, cfg, n_days, refit_every)-> same format, for past origins (seeds the log)
    describe_change(old, new, inputs, cfg)   -> dict of tables for the retrain e-mail (optional)

Output DataFrame columns: asof_date, model, horizon, forecast_return, sigma
    forecast_return: expected log return from asof_date to asof_date + horizon trading days
    sigma:           optional 1-sd uncertainty of that return (NaN if the model has none)
"""
from typing import Dict


class ForecastPlugin:
    name = "base"

    def train(self, inputs, cfg) -> dict:
        raise NotImplementedError

    def predict(self, artifact: dict, inputs, cfg):
        raise NotImplementedError

    def backtest(self, inputs, cfg, n_days: int, refit_every: int = 5):
        raise NotImplementedError

    def rolling_predict(self, artifact: dict, inputs, cfg, origins):
        """Forecasts at many past origins using FIXED parameters (the frozen-model evaluation). Default: call
        predict() on the inputs known at each origin (always correct, can be slow); plugins may override."""
        from .data import truncate_inputs
        import pandas as pd
        parts = [self.predict(artifact, truncate_inputs(inputs, d), cfg) for d in origins]
        return pd.concat(parts, ignore_index=True)

    def describe_change(self, old, new, inputs, cfg) -> dict:
        """What changed between two artifacts (old may be None on the first training), for the retrain
        e-mail. Return {"headline": [str, ...], "tables": [{"title": str, "header": [...], "rows": [[...]]}]}."""
        return {"headline": [f"Trained through {new.get('train_end')}."], "tables": []}


REGISTRY: Dict[str, ForecastPlugin] = {}


def register(plugin: ForecastPlugin):
    REGISTRY[plugin.name] = plugin
    return plugin
