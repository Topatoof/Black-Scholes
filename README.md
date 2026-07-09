# Black-Scholes Options Valuation Platform

A production-quality Python platform for European option pricing, Greeks analysis, implied volatility solving, Monte Carlo benchmarking, and scenario testing. Built with modular architecture and a Streamlit interactive dashboard.

## Features

- **Black-Scholes-Merton pricing** for European calls and puts with dividend yield support
- **Full Greeks suite**: Delta, Gamma, Vega, Theta, Rho (analytical)
- **Implied volatility** solvers: Newton-Raphson and Brent's method
- **Monte Carlo simulation** with antithetic variates for variance reduction
- **Binomial tree** (CRR) model with convergence analysis
- **Historical volatility** from CSV price data
- **Scenario engine** with stress testing and sensitivity analysis
- **Interactive Plotly charts**: payoff diagrams, volatility surface, Greeks curves, MC distributions

## Project Structure

```
black-scholes-platform/
├── src/
│   ├── black_scholes.py      # Core BS-Merton pricing
│   ├── greeks.py             # Analytical Greeks
│   ├── implied_vol.py        # IV solvers (NR + Brent)
│   ├── monte_carlo.py        # GBM Monte Carlo pricer
│   ├── binomial_tree.py      # CRR binomial tree
│   ├── volatility.py         # Historical vol analytics
│   └── scenario_analysis.py  # Sensitivity & scenarios
├── ui/
│   └── app.py                # Streamlit dashboard
├── ml/
│   ├── features.py           # Technical feature engineering
│   ├── xgboost_model.py      # XGBoost return predictor
│   └── backtest.py           # Walk-forward backtesting
├── data/
│   ├── market_data.py        # Alpaca API market data layer
│   └── sample_prices.csv     # Sample price data
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
cd black-scholes-platform
make install-dev    # creates venv + installs package in editable mode
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Streamlit Dashboard

```bash
make run
# or: streamlit run ui/app.py
```

### Python API

```python
from src.black_scholes import BlackScholesModel, OptionType
from src.greeks import GreeksCalculator

price = BlackScholesModel.price(
    spot=100, strike=100, time_to_expiry=1.0,
    risk_free_rate=0.05, volatility=0.20,
    option_type=OptionType.CALL, dividend_yield=0.02
)

greeks = GreeksCalculator.compute(
    spot=100, strike=100, time_to_expiry=1.0,
    risk_free_rate=0.05, volatility=0.20,
    option_type=OptionType.CALL
)
print(greeks.delta, greeks.gamma)
```

## Market Data (Alpaca)

Configure API credentials:

```bash
cp .env.example .env
# Edit .env with your Alpaca API keys from https://alpaca.markets
```

```python
from data.market_data import AlpacaMarketData
from src.black_scholes import BlackScholesModel

client = AlpacaMarketData.from_env()

# Historical OHLCV bars
history = client.get_historical_prices("AAPL", start="2024-01-01")

# Live spot price
spot = client.get_current_price("AAPL")

# Options chain
chain = client.get_options_chain("AAPL")

# Bridge to Black-Scholes engine
bs_inputs = client.prepare_black_scholes_inputs("AAPL", chain.iloc[0], risk_free_rate=0.05)
params = bs_inputs["option_params"]
price = BlackScholesModel.price(
    params.spot, params.strike, params.time_to_expiry,
    params.risk_free_rate, params.volatility,
    bs_inputs["option_type"], params.dividend_yield,
)
```

The Streamlit dashboard includes an **Alpaca Market Data** panel to fetch prices, chains, and price selected contracts directly.

## Machine Learning (XGBoost Return Prediction)

```python
from data.market_data import AlpacaMarketData
from ml import ReturnPredictor, WalkForwardBacktester

# Fetch OHLCV from Alpaca
md = AlpacaMarketData.from_env()
ohlcv = md.get_historical_prices("AAPL", start="2023-01-01")

# Train XGBoost return predictor
predictor = ReturnPredictor()
result = predictor.train(ohlcv)
print(result.metrics.to_dict())          # RMSE, MAE, directional accuracy
print(result.feature_importance.head())  # Feature importance

# Walk-forward backtest
bt = WalkForwardBacktester()
backtest = bt.run(ohlcv)
print(backtest.summary())
```

**macOS note:** XGBoost requires OpenMP: `brew install libomp`

## Running Tests

```bash
make test
# or: pytest tests/ -v
```

## Mathematical Assumptions

| Assumption | Description |
|---|---|
| GBM dynamics | dS = (r − q)S dt + σS dW under risk-neutral measure |
| European exercise | No early exercise (American options not supported) |
| Constant parameters | r, σ, q constant over option life |
| Continuous compounding | All rates are continuously compounded |
| No arbitrage | Enables risk-neutral pricing |
| Frictionless markets | No transaction costs, taxes, or bid-ask spread |
| Log-normal returns | Terminal price is log-normally distributed |

### Key Formulas

**d₁ and d₂:**
```
d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
```

**Call / Put:**
```
C = S·e^(−qT)·N(d₁) − K·e^(−rT)·N(d₂)
P = K·e^(−rT)·N(−d₂) − S·e^(−qT)·N(−d₁)
```

**Put-Call Parity:**
```
C − P = S·e^(−qT) − K·e^(−rT)
```

## Disclaimer

This platform is for educational and analytical purposes only. It does not constitute investment advice.
