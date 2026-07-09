"""
Black-Scholes Options Valuation Platform — Streamlit Dashboard.

Professional interactive dashboard for European option pricing, Greeks analysis,
implied volatility, Monte Carlo benchmarking, and scenario testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.binomial_tree import BinomialTreePricer
from src.black_scholes import BlackScholesModel, OptionType
from src.greeks import GreeksCalculator
from src.implied_vol import ImpliedVolatilitySolver, SolverMethod
from src.monte_carlo import MonteCarloPricer
from src.scenario_analysis import ScenarioEngine
from src.volatility import VolatilityCalculator

try:
  from data.market_data import (
    AlpacaMarketData,
    AuthenticationError,
    MarketDataError,
  )
  MARKET_DATA_AVAILABLE = True
except ImportError:
  MARKET_DATA_AVAILABLE = False

try:
  from ml import ReturnPredictor, WalkForwardBacktester, XGBoostConfig
  from ml.features import FeatureEngineer

  def _xgboost_available() -> bool:
    try:
      from xgboost import XGBRegressor  # noqa: F401
      return True
    except Exception:
      return False

  ML_AVAILABLE = _xgboost_available()
  ML_IMPORT_ERROR = None if ML_AVAILABLE else (
    "XGBoost could not be loaded. On macOS run: brew install libomp"
  )
except ImportError as exc:
  ML_AVAILABLE = False
  ML_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
  page_title="Black-Scholes Options Platform",
  page_icon="📈",
  layout="wide",
  initial_sidebar_state="expanded",
)

CHART_TEMPLATE = "plotly_white"


def _option_type_from_str(s: str) -> OptionType:
  return OptionType.CALL if s.lower() == "call" else OptionType.PUT


def _metric_row(cols, labels, values, formats=None):
  formats = formats or ["${:.4f}"] * len(labels)
  for col, label, val, fmt in zip(cols, labels, values, formats):
    col.metric(label, fmt.format(val) if isinstance(val, (int, float)) else val)


# ---------------------------------------------------------------------------
# Sidebar: Option Inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Option Inputs")

option_type_str = st.sidebar.selectbox("Option Type", ["Call", "Put"])
spot = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=100.0, step=1.0)
strike = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
time_to_expiry = st.sidebar.number_input("Time to Expiry (Years)", min_value=0.001, value=1.0, step=0.1, format="%.3f")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.005, format="%.4f")
dividend_yield = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0, value=0.0, step=0.005, format="%.4f")
volatility = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.20, step=0.01, format="%.4f")

option_type = _option_type_from_str(option_type_str)

st.title("Black-Scholes Options Valuation Platform")
st.caption(
  "European option pricing with Greeks, implied volatility, Monte Carlo benchmarking, "
  "and scenario analysis. Continuous compounding assumed throughout."
)

# ---------------------------------------------------------------------------
# Section 2: Valuation Results
# ---------------------------------------------------------------------------
st.header("Valuation Results")

bs_price = BlackScholesModel.price(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
)
intrinsic = BlackScholesModel.intrinsic_value(spot, strike, option_type)
time_value = bs_price - intrinsic

col1, col2, col3, col4 = st.columns(4)
_metric_row(
  [col1, col2, col3, col4],
  ["Black-Scholes Price", "Intrinsic Value", "Time Value", "Moneyness"],
  [bs_price, intrinsic, time_value, spot / strike],
  ["${:.4f}", "${:.4f}", "${:.4f}", "{:.4f}"],
)

# Binomial comparison
bin_result = BinomialTreePricer.price(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, num_steps=200, dividend_yield=dividend_yield
)
col5, col6 = st.columns(2)
col5.metric("Binomial Tree (200 steps)", f"${bin_result.price:.4f}")
col6.metric("Binomial vs BS", f"${abs(bin_result.price - bs_price):.6f}")

# Option value vs stock price
st.subheader("Option Value vs Stock Price")
spot_range = np.linspace(spot * 0.6, spot * 1.4, 100)
option_values = BlackScholesModel.price_vectorized(
  spot_range, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
)
payoff = BlackScholesModel.payoff_at_expiry(spot_range, strike, option_type)

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=spot_range, y=option_values, name="BS Option Value", line=dict(color="#2563eb", width=2)))
fig_price.add_trace(go.Scatter(x=spot_range, y=payoff, name="Expiry Payoff", line=dict(color="#dc2626", dash="dash")))
fig_price.add_vline(x=strike, line_dash="dot", annotation_text=f"K={strike}")
fig_price.add_vline(x=spot, line_dash="dot", line_color="green", annotation_text=f"S={spot}")
fig_price.update_layout(
  template=CHART_TEMPLATE,
  xaxis_title="Stock Price",
  yaxis_title="Option Value ($)",
  height=400,
  legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Greeks Analysis
# ---------------------------------------------------------------------------
st.header("Greeks Analysis")

greeks = GreeksCalculator.compute(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
)

g1, g2, g3, g4, g5 = st.columns(5)
_metric_row(
  [g1, g2, g3, g4, g5],
  ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ/day)", "Rho (ρ)"],
  [greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho],
  ["{:.4f}", "{:.6f}", "{:.4f}", "{:.4f}", "{:.4f}"],
)

greek_tabs = st.tabs(["Delta", "Gamma", "Vega", "Theta", "Rho"])
greek_names = ["delta", "gamma", "vega", "theta", "rho"]
greek_labels = ["Delta (Δ)", "Gamma (Γ)", "Vega (ν per 1%)", "Theta (Θ/day)", "Rho (ρ per 1%)"]

for tab, gname, glabel in zip(greek_tabs, greek_names, greek_labels):
  with tab:
    curve = GreeksCalculator.compute_curve(
      spot_range, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield, gname
    )
    fig_greek = go.Figure()
    fig_greek.add_trace(go.Scatter(x=spot_range, y=curve, line=dict(color="#7c3aed", width=2)))
    fig_greek.add_vline(x=spot, line_dash="dot", line_color="green")
    fig_greek.update_layout(
      template=CHART_TEMPLATE,
      xaxis_title="Stock Price",
      yaxis_title=glabel,
      height=350,
    )
    st.plotly_chart(fig_greek, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Implied Volatility
# ---------------------------------------------------------------------------
st.header("Implied Volatility")

iv_col1, iv_col2, iv_col3 = st.columns(3)
with iv_col1:
  market_price = st.number_input("Market Option Price", min_value=0.01, value=round(bs_price, 4), step=0.1)
with iv_col2:
  iv_method = st.selectbox("Solver Method", ["Newton-Raphson", "Brent"])
with iv_col3:
  initial_guess = st.number_input("Initial Vol Guess (NR)", min_value=0.01, value=0.20, step=0.01)

method = SolverMethod.NEWTON_RAPHSON if iv_method == "Newton-Raphson" else SolverMethod.BRENT

try:
  iv_result = ImpliedVolatilitySolver.solve(
    market_price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield, method, initial_guess
  )
  iv_c1, iv_c2, iv_c3, iv_c4 = st.columns(4)
  iv_c1.metric("Implied Volatility", f"{iv_result.implied_vol:.4%}")
  iv_c2.metric("Converged", "Yes" if iv_result.converged else "No")
  iv_c3.metric("Iterations", iv_result.iterations)
  iv_c4.metric("Final Error", f"${iv_result.final_error:.8f}")
except ValueError as e:
  st.error(f"IV Solver Error: {e}")

# Volatility smile
st.subheader("Volatility Smile")
smile_col1, smile_col2 = st.columns(2)
with smile_col1:
  smile_skew = st.slider("Skew", -0.3, 0.3, -0.1, 0.01)
with smile_col2:
  smile_curvature = st.slider("Smile Curvature", 0.0, 0.2, 0.05, 0.01)

smile_fn = VolatilityCalculator.generate_synthetic_smile(spot, volatility, smile_skew, smile_curvature)
strikes_smile = np.linspace(strike * 0.7, strike * 1.3, 20)
ivs_smile = [smile_fn(k) for k in strikes_smile]

fig_smile = go.Figure()
fig_smile.add_trace(go.Scatter(x=strikes_smile, y=ivs_smile, mode="lines+markers", name="IV Smile"))
fig_smile.add_vline(x=spot, line_dash="dot", annotation_text="ATM")
fig_smile.update_layout(
  template=CHART_TEMPLATE,
  xaxis_title="Strike Price",
  yaxis_title="Implied Volatility",
  height=350,
)
st.plotly_chart(fig_smile, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Volatility Analytics (Surface)
# ---------------------------------------------------------------------------
st.header("Volatility Analytics")

expiry_range = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
strike_range_surf = np.linspace(strike * 0.8, strike * 1.2, 15)
vol_surface = np.zeros((len(expiry_range), len(strike_range_surf)))

for i, T in enumerate(expiry_range):
  for j, K in enumerate(strike_range_surf):
    moneyness = K / spot - 1.0
    vol_surface[i, j] = max(volatility + smile_skew * moneyness + smile_curvature * moneyness**2, 0.01)

strike_mesh, expiry_mesh, _ = VolatilityCalculator.volatility_surface_grid(
  spot, strike_range_surf, expiry_range, vol_surface
)

fig_surface = go.Figure(
  data=[
    go.Surface(
      x=strike_mesh,
      y=expiry_mesh,
      z=vol_surface,
      colorscale="Viridis",
      colorbar=dict(title="σ"),
    )
  ]
)
fig_surface.update_layout(
  template=CHART_TEMPLATE,
  title="Volatility Surface (Synthetic)",
  scene=dict(xaxis_title="Strike", yaxis_title="Expiry (Yrs)", zaxis_title="Implied Vol"),
  height=500,
)
st.plotly_chart(fig_surface, use_container_width=True)

# Payoff diagram
st.subheader("Payoff Diagram")
premium = st.slider("Premium Paid", 0.0, float(bs_price * 2), float(bs_price), 0.01)
payoff_pnl = BlackScholesModel.payoff_at_expiry(spot_range, strike, option_type, premium)

fig_payoff = go.Figure()
fig_payoff.add_trace(go.Scatter(x=spot_range, y=payoff_pnl, fill="tozeroy", name="P&L at Expiry"))
fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
fig_payoff.add_vline(x=strike, line_dash="dot")
fig_payoff.update_layout(
  template=CHART_TEMPLATE,
  xaxis_title="Stock Price at Expiry",
  yaxis_title="Profit / Loss ($)",
  height=350,
)
st.plotly_chart(fig_payoff, use_container_width=True)

# Time decay
st.subheader("Time Decay (Theta Path)")
time_decay = ScenarioEngine.time_sensitivity(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield=dividend_yield
)
fig_decay = go.Figure()
fig_decay.add_trace(
  go.Scatter(x=time_decay.factor_values * 365, y=time_decay.option_prices, name="Option Value", line=dict(color="#ea580c"))
)
fig_decay.update_layout(
  template=CHART_TEMPLATE,
  xaxis_title="Days to Expiration",
  yaxis_title="Option Value ($)",
  height=350,
)
st.plotly_chart(fig_decay, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 6: Monte Carlo Simulation
# ---------------------------------------------------------------------------
st.header("Monte Carlo Simulation")

mc_col1, mc_col2, mc_col3 = st.columns(3)
with mc_col1:
  num_sims = st.selectbox("Simulations", [10_000, 50_000, 100_000, 500_000], index=2)
with mc_col2:
  use_antithetic = st.checkbox("Antithetic Variates", value=True)
with mc_col3:
  mc_seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

mc_result = MonteCarloPricer.price(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type,
  num_sims, dividend_yield, use_antithetic, int(mc_seed), return_paths=True,
)

mc_m1, mc_m2, mc_m3, mc_m4 = st.columns(4)
mc_m1.metric("MC Price", f"${mc_result.price:.4f}")
mc_m2.metric("BS Price", f"${bs_price:.4f}")
mc_m3.metric("Difference", f"${abs(mc_result.price - bs_price):.4f}")
mc_m4.metric("Std Error", f"${mc_result.standard_error:.4f}")

st.info(
  f"95% Confidence Interval: [${mc_result.confidence_interval_95[0]:.4f}, "
  f"${mc_result.confidence_interval_95[1]:.4f}]"
)

if mc_result.terminal_prices is not None:
  fig_mc = make_subplots(rows=1, cols=2, subplot_titles=("Terminal Price Distribution", "Discounted Payoff Distribution"))

  fig_mc.add_trace(
    go.Histogram(x=mc_result.terminal_prices, nbinsx=60, name="S_T", marker_color="#2563eb"),
    row=1, col=1,
  )
  fig_mc.add_trace(
    go.Histogram(x=mc_result.payoffs, nbinsx=60, name="Payoff", marker_color="#16a34a"),
    row=1, col=2,
  )
  fig_mc.update_layout(template=CHART_TEMPLATE, height=400, showlegend=False)
  st.plotly_chart(fig_mc, use_container_width=True)

# MC convergence
st.subheader("Monte Carlo Convergence")
conv_counts = [1000, 5000, 10000, 25000, 50000, 100000]
counts, mc_prices, mc_errors = MonteCarloPricer.convergence_analysis(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, conv_counts, dividend_yield
)

fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
fig_conv.add_trace(go.Scatter(x=counts, y=mc_prices, name="MC Price", mode="lines+markers"))
fig_conv.add_hline(y=bs_price, line_dash="dash", name="BS Price", annotation_text="BS")
fig_conv.add_trace(go.Scatter(x=counts, y=mc_errors, name="Std Error", mode="lines+markers", line=dict(color="red")), secondary_y=True)
fig_conv.update_layout(template=CHART_TEMPLATE, xaxis_title="Simulations", height=350)
fig_conv.update_yaxes(title_text="Price ($)")
fig_conv.update_yaxes(title_text="Std Error", secondary_y=True)
st.plotly_chart(fig_conv, use_container_width=True)

# Binomial convergence
st.subheader("Binomial Tree Convergence")
step_counts = [10, 25, 50, 100, 200, 500]
_, bin_prices, bin_errors = BinomialTreePricer.convergence_analysis(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, step_counts, dividend_yield, bs_price
)

fig_bin = go.Figure()
fig_bin.add_trace(go.Scatter(x=step_counts, y=bin_prices, mode="lines+markers", name="Binomial"))
fig_bin.add_hline(y=bs_price, line_dash="dash", annotation_text="BS")
fig_bin.update_layout(template=CHART_TEMPLATE, xaxis_title="Tree Steps", yaxis_title="Price ($)", height=350)
st.plotly_chart(fig_bin, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 7: Scenario Testing
# ---------------------------------------------------------------------------
st.header("Scenario Testing")

scenarios = ScenarioEngine.run_scenarios(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield=dividend_yield
)
scenario_df = ScenarioEngine.scenarios_to_dataframe(scenarios)
st.dataframe(scenario_df.style.format({
  "Spot": "${:.2f}",
  "Volatility": "{:.2%}",
  "Time (Yrs)": "{:.3f}",
  "Rate": "{:.2%}",
  "Price": "${:.4f}",
  "Delta": "{:.4f}",
  "P&L": "${:.4f}",
}), use_container_width=True, hide_index=True)

# Sensitivity tabs
sens_tab1, sens_tab2, sens_tab3, sens_tab4 = st.tabs(["Spot", "Volatility", "Time", "Rate"])

with sens_tab1:
  sens = ScenarioEngine.spot_sensitivity(
    spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield=dividend_yield
  )
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=sens.factor_values, y=sens.option_prices, name="Price"))
  fig.update_layout(template=CHART_TEMPLATE, xaxis_title="Spot Price", yaxis_title="Option Price", height=350)
  st.plotly_chart(fig, use_container_width=True)

with sens_tab2:
  sens = ScenarioEngine.volatility_sensitivity(
    spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield=dividend_yield
  )
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=sens.factor_values, y=sens.option_prices, name="Price"))
  fig.update_layout(template=CHART_TEMPLATE, xaxis_title="Volatility", yaxis_title="Option Price", height=350)
  st.plotly_chart(fig, use_container_width=True)

with sens_tab3:
  sens = ScenarioEngine.time_sensitivity(
    spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield=dividend_yield
  )
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=sens.factor_values * 365, y=sens.option_prices, name="Price"))
  fig.update_layout(template=CHART_TEMPLATE, xaxis_title="Days to Expiry", yaxis_title="Option Price", height=350)
  st.plotly_chart(fig, use_container_width=True)

with sens_tab4:
  sens = ScenarioEngine.rate_sensitivity(
    spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield=dividend_yield
  )
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=sens.factor_values, y=sens.option_prices, name="Price"))
  fig.update_layout(template=CHART_TEMPLATE, xaxis_title="Risk-Free Rate", yaxis_title="Option Price", height=350)
  st.plotly_chart(fig, use_container_width=True)

# Scenario heatmap
st.subheader("Spot × Volatility Scenario Grid")
spot_shocks = np.linspace(-0.3, 0.3, 13)
vol_shocks = np.linspace(-0.10, 0.10, 11)
s_mesh, v_mesh, p_surface = ScenarioEngine.scenario_grid(
  spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, spot_shocks, vol_shocks, dividend_yield
)

fig_heat = go.Figure(
  data=go.Heatmap(
    x=[f"{s:+.0%}" for s in spot_shocks],
    y=[f"{v:+.0%}" for v in vol_shocks],
    z=p_surface,
    colorscale="RdYlGn",
    colorbar=dict(title="Price"),
  )
)
fig_heat.update_layout(
  template=CHART_TEMPLATE,
  xaxis_title="Spot Shock",
  yaxis_title="Vol Shock",
  height=400,
)
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 8: Data Import & Market Data
# ---------------------------------------------------------------------------
st.header("Data Import & Historical Volatility")

# --- Alpaca live market data ---
st.subheader("Alpaca Market Data")

if MARKET_DATA_AVAILABLE:
  md_col1, md_col2, md_col3 = st.columns(3)
  with md_col1:
    ticker = st.text_input("Ticker Symbol", value="AAPL", key="alpaca_ticker")
  with md_col2:
    hist_days = st.number_input("History (days)", min_value=30, max_value=3650, value=252, key="hist_days")
  with md_col3:
    risk_free_for_md = st.number_input(
      "Risk-Free Rate (for BS bridge)",
      min_value=0.0,
      max_value=0.20,
      value=float(risk_free_rate),
      step=0.001,
      format="%.4f",
      key="md_risk_free",
    )

  fetch_hist = st.button("Fetch Historical Prices", key="fetch_hist")
  fetch_spot = st.button("Fetch Current Price", key="fetch_spot")
  fetch_chain = st.button("Fetch Options Chain", key="fetch_chain")

  if fetch_hist or fetch_spot or fetch_chain:
    try:
      md_client = AlpacaMarketData.from_env()
      symbol = md_client.validate_symbol(ticker)

      if fetch_hist:
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(hist_days))
        hist_df = md_client.get_historical_prices(symbol, start=start)
        st.session_state["alpaca_hist"] = hist_df
        st.session_state["alpaca_symbol"] = symbol
        closes = md_client.get_close_series(hist_df)
        vol_stats = VolatilityCalculator.historical_volatility(closes)
        st.session_state["suggested_vol"] = vol_stats.annualized_volatility
        st.success(
          f"Loaded {len(hist_df)} bars for {symbol}. "
          f"Annualized vol: {vol_stats.annualized_volatility:.2%}"
        )

      if fetch_spot:
        live_spot = md_client.get_current_price(symbol)
        st.session_state["suggested_spot"] = live_spot
        st.success(f"{symbol} current price: ${live_spot:.2f}")

      if fetch_chain:
        chain_df = md_client.get_options_chain(symbol)
        st.session_state["alpaca_chain"] = chain_df
        st.success(f"Loaded {len(chain_df)} option contracts for {symbol}")

    except AuthenticationError:
      st.error(
        "Alpaca credentials not configured. Copy `.env.example` to `.env` "
        "and set ALPACA_API_KEY and ALPACA_SECRET_KEY."
      )
    except MarketDataError as exc:
      st.error(f"Market data error: {exc}")

  if "alpaca_hist" in st.session_state:
    hist_df = st.session_state["alpaca_hist"]
    fig_alpaca = go.Figure()
    fig_alpaca.add_trace(
      go.Scatter(x=hist_df.index, y=hist_df["close"], name="Close", line=dict(color="#2563eb"))
    )
    fig_alpaca.update_layout(
      template=CHART_TEMPLATE,
      title=f"{st.session_state.get('alpaca_symbol', ticker)} Price History (Alpaca)",
      xaxis_title="Date",
      yaxis_title="Close ($)",
      height=350,
    )
    st.plotly_chart(fig_alpaca, use_container_width=True)

  if "alpaca_chain" in st.session_state:
    chain_df = st.session_state["alpaca_chain"]
    st.dataframe(
      chain_df.head(50).style.format({
        "strike": "{:.2f}",
        "bid": "{:.2f}",
        "ask": "{:.2f}",
        "mid": "{:.2f}",
        "implied_volatility": "{:.2%}",
      }),
      use_container_width=True,
      hide_index=True,
    )

    contract_idx = st.selectbox(
      "Select contract for Black-Scholes pricing",
      range(min(len(chain_df), 50)),
      format_func=lambda i: (
        f"{chain_df.iloc[i]['symbol']} | K={chain_df.iloc[i]['strike']:.2f} | "
        f"{chain_df.iloc[i]['option_type'].upper()} | mid=${chain_df.iloc[i]['mid']:.2f}"
      ),
      key="contract_select",
    )

    if st.button("Price Selected Contract with Black-Scholes", key="price_contract"):
      try:
        md_client = AlpacaMarketData.from_env()
        bs_inputs = md_client.prepare_black_scholes_inputs(
          ticker,
          chain_df.iloc[contract_idx],
          risk_free_rate=risk_free_for_md,
        )
        p = bs_inputs["option_params"]
        opt_t = bs_inputs["option_type"]
        model_price = BlackScholesModel.price(
          p.spot, p.strike, p.time_to_expiry, p.risk_free_rate, p.volatility, opt_t, p.dividend_yield
        )
        mkt = bs_inputs["market_price"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Price (BS)", f"${model_price:.4f}")
        c2.metric("Market Mid", f"${mkt:.4f}")
        c3.metric("Spot", f"${p.spot:.2f}")
        c4.metric("Hist. Vol", f"{bs_inputs['historical_volatility']:.2%}" if bs_inputs["historical_volatility"] else "N/A")

        st.json({
          "strike": p.strike,
          "time_to_expiry_years": round(p.time_to_expiry, 4),
          "volatility": round(p.volatility, 4),
          "risk_free_rate": p.risk_free_rate,
          "contract": bs_inputs["contract_symbol"],
        })
      except MarketDataError as exc:
        st.error(f"Pricing error: {exc}")
else:
  st.warning("Market data module not available. Install alpaca-py: pip install alpaca-py")

st.divider()
st.subheader("CSV Upload")

uploaded_file = st.file_uploader("Upload CSV (Date, Close columns)", type=["csv"])

if uploaded_file is not None:
  try:
    df_upload = pd.read_csv(uploaded_file)
    st.dataframe(df_upload.head(10), use_container_width=True)

    date_col = st.selectbox("Date Column", df_upload.columns.tolist(), index=0)
    price_col = st.selectbox("Price Column", df_upload.columns.tolist(),
                              index=df_upload.columns.tolist().index("Close") if "Close" in df_upload.columns else 0)

    df_upload[date_col] = pd.to_datetime(df_upload[date_col], errors="coerce")
    df_upload = df_upload.dropna(subset=[date_col, price_col]).set_index(date_col)
    prices = VolatilityCalculator.validate_price_data(df_upload[price_col])

    vol_stats = VolatilityCalculator.historical_volatility(prices)
    roll_window = st.slider("Rolling Volatility Window (days)", 5, 60, 21)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Daily Volatility", f"{vol_stats.daily_volatility:.4%}")
    v2.metric("Annualized Volatility", f"{vol_stats.annualized_volatility:.2%}")
    v3.metric("Mean Log Return", f"{vol_stats.mean_log_return:.6f}")
    v4.metric("Observations", vol_stats.num_observations)

  # Use computed vol as input suggestion
    if st.button("Apply Historical Vol to Pricing"):
      st.session_state["suggested_vol"] = vol_stats.annualized_volatility
      st.success(f"Suggested volatility: {vol_stats.annualized_volatility:.2%}")

    log_rets = VolatilityCalculator.log_returns(prices)
    rolling_vol = VolatilityCalculator.rolling_volatility(prices, roll_window)

    fig_data = make_subplots(rows=2, cols=1, subplot_titles=("Price History", "Rolling Volatility"), vertical_spacing=0.12)
    fig_data.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Price"), row=1, col=1)
    fig_data.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="Rolling Vol", line=dict(color="#dc2626")), row=2, col=1)
    fig_data.update_layout(template=CHART_TEMPLATE, height=500, showlegend=False)
    fig_data.update_yaxes(title_text="Price", row=1, col=1)
    fig_data.update_yaxes(title_text="Annualized σ", row=2, col=1)
    st.plotly_chart(fig_data, use_container_width=True)

    fig_rets = go.Figure()
    fig_rets.add_trace(go.Histogram(x=log_rets.values, nbinsx=50, name="Log Returns"))
    fig_rets.update_layout(template=CHART_TEMPLATE, title="Log Return Distribution", height=300)
    st.plotly_chart(fig_rets, use_container_width=True)

  except (ValueError, KeyError) as e:
    st.error(f"Data validation error: {e}")
else:
  st.info("Upload a CSV file with date and price columns to compute historical volatility.")
  sample_path = PROJECT_ROOT / "data" / "sample_prices.csv"
  if sample_path.exists():
    st.caption(f"Sample data available at: `{sample_path}`")

# ---------------------------------------------------------------------------
# Section 9: ML Return Prediction (XGBoost)
# ---------------------------------------------------------------------------
st.header("ML Return Prediction (XGBoost)")
st.caption(
  "Predict next-day log returns using technical features. "
  "Chronological train/test split and walk-forward backtest — no look-ahead bias."
)

if not ML_AVAILABLE:
  st.warning(
    f"ML module unavailable. {ML_IMPORT_ERROR or 'Install dependencies:'}  \n"
    "```bash\npip install xgboost scikit-learn joblib\nbrew install libomp  # macOS only\n```"
  )
else:
  # Clean up legacy session key that collided with button widget state
  if "ml_backtest" in st.session_state and isinstance(st.session_state["ml_backtest"], bool):
    del st.session_state["ml_backtest"]

  ml_c1, ml_c2, ml_c3 = st.columns(3)
  with ml_c1:
    ml_ticker = st.text_input("ML Ticker", value=st.session_state.get("alpaca_symbol", "AAPL"), key="ml_ticker")
  with ml_c2:
    ml_hist_days = st.number_input("Training history (days)", 180, 3650, 504, key="ml_hist_days")
  with ml_c3:
    ml_test_pct = st.slider("Test set (%)", 10, 40, 20, key="ml_test_pct")

  with st.expander("Model hyperparameters"):
    hp1, hp2, hp3 = st.columns(3)
    with hp1:
      ml_estimators = st.number_input("Trees", 50, 1000, 300, step=50, key="ml_estimators")
      ml_depth = st.number_input("Max depth", 2, 10, 4, key="ml_depth")
    with hp2:
      ml_lr = st.number_input("Learning rate", 0.01, 0.3, 0.05, step=0.01, key="ml_lr")
      ml_cv_splits = st.number_input("CV folds", 3, 10, 5, key="ml_cv_splits")
    with hp3:
      ml_bt_step = st.number_input("Backtest step (days)", 5, 63, 21, key="ml_bt_step")
      ml_min_train = st.number_input("Min train rows", 60, 500, 120, key="ml_min_train")

  ml_config = XGBoostConfig(
    n_estimators=int(ml_estimators),
    max_depth=int(ml_depth),
    learning_rate=float(ml_lr),
    n_cv_splits=int(ml_cv_splits),
    test_size=float(ml_test_pct) / 100.0,
  )

  train_col, backtest_col, predict_col = st.columns(3)
  train_clicked = train_col.button("Train Model", type="primary", key="ml_train_btn")
  backtest_clicked = backtest_col.button("Run Walk-Forward Backtest", key="ml_backtest_btn")
  predict_clicked = predict_col.button("Generate Predictions", key="ml_predict_btn")

  def _load_ohlcv_for_ml(ticker_sym: str, days: int) -> pd.DataFrame:
    """Use cached Alpaca history or fetch fresh OHLCV."""
    if (
      "alpaca_hist" in st.session_state
      and st.session_state.get("alpaca_symbol", "").upper() == ticker_sym.upper()
      and len(st.session_state["alpaca_hist"]) >= 100
    ):
      return st.session_state["alpaca_hist"]
    if not MARKET_DATA_AVAILABLE:
      raise RuntimeError("Alpaca market data required. Configure API keys in .env")
    client = AlpacaMarketData.from_env()
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return client.get_historical_prices(client.validate_symbol(ticker_sym), start=start)

  if train_clicked:
    with st.spinner(f"Training XGBoost on {ml_ticker}..."):
      try:
        ohlcv_ml = _load_ohlcv_for_ml(ml_ticker, int(ml_hist_days))
        predictor = ReturnPredictor(config=ml_config, model_dir=PROJECT_ROOT / "models")
        result = predictor.train(ohlcv_ml, save=True, model_name=ml_ticker.upper())
        st.session_state["ml_result"] = result
        st.session_state["ml_predictor_path"] = str(result.model_path)
        st.session_state["ml_ohlcv"] = ohlcv_ml
        st.session_state["ml_ticker_saved"] = ml_ticker.upper()
        st.success(f"Model trained and saved to `{result.model_path.name}`")
      except Exception as exc:
        st.error(f"Training failed: {exc}")

  if backtest_clicked:
    with st.spinner("Running walk-forward backtest..."):
      try:
        ohlcv_ml = _load_ohlcv_for_ml(ml_ticker, int(ml_hist_days))
        bt = WalkForwardBacktester(
          config=ml_config,
          min_train_rows=int(ml_min_train),
          test_step=int(ml_bt_step),
        )
        bt_result = bt.run(ohlcv_ml, retrain_each_window=True)
        st.session_state["ml_backtest_result"] = bt_result
        st.success("Backtest complete.")
      except Exception as exc:
        st.error(f"Backtest failed: {exc}")

  if predict_clicked and "ml_predictor_path" in st.session_state:
    try:
      if "ml_ohlcv" in st.session_state and len(st.session_state["ml_ohlcv"]) > 0:
        ohlcv_ml = st.session_state["ml_ohlcv"]
      else:
        ohlcv_ml = _load_ohlcv_for_ml(ml_ticker, int(ml_hist_days))
      predictor = ReturnPredictor(config=ml_config, model_dir=PROJECT_ROOT / "models")
      predictor.load(st.session_state["ml_predictor_path"])
      preds = predictor.predict(ohlcv_ml)
      st.session_state["ml_predictions"] = preds
    except Exception as exc:
      st.error(f"Prediction failed: {exc}")

  if "ml_result" in st.session_state:
    result = st.session_state["ml_result"]
    st.subheader("Training Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{result.metrics.rmse:.6f}")
    m2.metric("MAE", f"{result.metrics.mae:.6f}")
    m3.metric("Directional Accuracy", f"{result.metrics.directional_accuracy:.1%}")
    m4.metric("Test samples", result.metrics.n_samples)

    st.caption(f"Train size: {result.train_size} | Test size: {result.test_size}")

    # CV summary
    cv_rmse = np.mean([m.rmse for m in result.cv_metrics])
    cv_dir = np.mean([m.directional_accuracy for m in result.cv_metrics])
    st.info(f"Cross-validation avg RMSE: {cv_rmse:.6f} | avg directional accuracy: {cv_dir:.1%}")

    # Feature importance chart
    imp = result.feature_importance.head(15)
    fig_imp = go.Figure(
      go.Bar(
        x=imp["importance"],
        y=imp["feature"],
        orientation="h",
        marker_color="#7c3aed",
      )
    )
    fig_imp.update_layout(
      template=CHART_TEMPLATE,
      title="Feature Importance (Top 15)",
      xaxis_title="Importance (gain)",
      height=450,
      yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

  if "ml_backtest_result" in st.session_state:
    bt_result = st.session_state["ml_backtest_result"]
    if not hasattr(bt_result, "metrics"):
      st.warning("Backtest data invalid — click **Run Walk-Forward Backtest** again.")
    else:
      st.subheader("Walk-Forward Backtest")

      b1, b2, b3, b4 = st.columns(4)
      b1.metric("RMSE", f"{bt_result.metrics.rmse:.6f}")
      b2.metric("Directional Accuracy", f"{bt_result.metrics.directional_accuracy:.1%}")
      b3.metric("Strategy Return", f"{bt_result.cumulative_strategy_return:.2%}")
      b4.metric("Buy & Hold Return", f"{bt_result.cumulative_buy_hold_return:.2%}")

      pred_df = bt_result.predictions
      fig_bt = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Predicted vs Actual Returns", "Cumulative Strategy vs Buy & Hold"),
        vertical_spacing=0.12,
      )
      fig_bt.add_trace(
        go.Scatter(x=pred_df.index, y=pred_df["actual_return"], name="Actual", line=dict(color="#2563eb")),
        row=1, col=1,
      )
      fig_bt.add_trace(
        go.Scatter(x=pred_df.index, y=pred_df["predicted_return"], name="Predicted", line=dict(color="#ea580c")),
        row=1, col=1,
      )
      positions = (pred_df["predicted_return"] > 0).astype(float)
      strat_cum = (1 + positions * pred_df["actual_return"]).cumprod() - 1
      bh_cum = (1 + pred_df["actual_return"]).cumprod() - 1
      fig_bt.add_trace(go.Scatter(x=pred_df.index, y=strat_cum, name="Long/Flat Strategy"), row=2, col=1)
      fig_bt.add_trace(go.Scatter(x=pred_df.index, y=bh_cum, name="Buy & Hold"), row=2, col=1)
      fig_bt.update_layout(template=CHART_TEMPLATE, height=550, legend=dict(orientation="h", y=1.06))
      st.plotly_chart(fig_bt, use_container_width=True)

  if "ml_predictions" in st.session_state:
    preds = st.session_state["ml_predictions"]
    st.subheader("Latest Return Predictions")
    st.dataframe(
      preds.tail(20).to_frame("predicted_log_return").style.format("{:.6f}"),
      use_container_width=True,
    )
    direction = "UP" if preds.iloc[-1] > 0 else "DOWN"
    st.metric("Latest signal", direction, f"{preds.iloc[-1]:.4%} predicted log return")

# Footer
st.divider()
st.caption(
  "Black-Scholes-Merton model | Continuous compounding | European exercise | "
  "For educational and analytical purposes only — not investment advice."
)
