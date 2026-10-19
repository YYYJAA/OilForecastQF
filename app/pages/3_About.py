"""About: how it works, model cards, evaluation design, limits and data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from lib import data as D
from lib import ui

ui.setup("About · Oil Forecast Monitor", "ℹ️")
ui.sidebar()
st.title("About")

st.markdown("Models are retrained **once a quarter**; in between they are only run forward on new data with fixed parameters.")

ms = D.meta()
st.subheader("The models")
CARDS = {
    "lasso_family": ("LASSO family (LASSO, LASSO 1-SE, regime LASSO, PCA, targeted PCA, equal-weight combination)",
                     "Regress future returns on lagged returns, volatility, term structure, macro and calendar variables, with a hidden "
                     "volatility-regime probability. LASSO shrinks useless predictors to zero; for oil it usually keeps none, which means "
                     "'no change'. The PCA variants compress the predictors into a few factors."),
    "arma_garch": ("ARMA-GARCH", "A small ARMA for the mean and a GARCH(1,1) with fat tails for the changing volatility. Its value is the "
                                 "uncertainty band: the mean is close to zero."),
    "msar": ("MS-AR / MS-ARX", "Markov-switching autoregression: returns follow a calm or a turbulent regime, with probabilities updated daily. "
                               "MS-ARX adds external predictors and shrinks the forecast towards zero."),
}
for key, (title, text) in CARDS.items():
    with st.expander(title):
        st.write(text)
        m = ms.get(key, {})
        if m:
            st.caption(f"Last retrained {str(m.get('trained_at', 'n/a'))[:10]}, on data through {m.get('train_end', 'n/a')}.")

st.subheader("How we evaluate")
summ = D.frozen("summary.json")
st.markdown(f"""
* **Frozen test.** Models fitted on data to **{summ.get('cutoff', '2024-12-31')}** are applied daily from **{summ.get('first_origin', '2025-01-02')}** with no
  refitting; forecasts are compared with what happened, 1 to 60 trading days ahead.
* **Benchmarks.** *No change* (random walk), historical mean and a direct AR model. RMSE ratio below 1 beats no change; Diebold-Mariano p-values
  test whether a difference is real.
* **Bands.** A ±1 sd band should contain about 68% of outcomes.
* **Signals.** BUY / SELL when the expected move exceeds 0.1 × its noise, else HOLD. *Track record* comes from the frozen test, not from the model's own claim.
* **Economic value.** A simple long / short rule on the forecasts, with trading costs, against buy-and-hold.
""")

st.subheader("Limits to keep in mind")
st.markdown("""
* Futures series are unadjusted continuous contracts, so they **jump when the contract rolls**; spot (FRED) has no jumps but arrives about a week late.
* The latest day's futures price is provisional until the close and is corrected by the next day's refresh.
* WTI futures are used from June 2020 (negative price in April 2020).
* Window and weight choices were made on 2022–2023 validation data and some design decisions were made after seeing later data: the test is
  clean for estimation, not pristine.
* Several models forecast exactly 'no change'; the uncertainty band is where most of the evidence lies.
""")
