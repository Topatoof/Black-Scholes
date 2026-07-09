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
# Section 8: Data Import
# ---------------------------------------------------------------------------
st.header("Data Import & Historical Volatility")

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

# Footer
st.divider()
st.caption(
  "Black-Scholes-Merton model | Continuous compounding | European exercise | "
  "For educational and analytical purposes only — not investment advice."
)
