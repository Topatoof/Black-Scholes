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
├── tests/                    # Unit test suite
├── data/
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
