"""Importing this package registers every available plugin."""
from . import lasso_regime  # noqa: F401   (replaces the earlier lasso_family set)
from . import arma_garch    # noqa: F401
from . import msar          # noqa: F401
from . import xgboost_trees  # noqa: F401
