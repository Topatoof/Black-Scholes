"""
Scenario analysis and sensitivity engine for options.

Provides systematic stress testing across key risk factors:
    - Stock price shocks
    - Volatility shifts
    - Time decay (theta path)
    - Interest rate changes

Also generates multi-dimensional scenario grids for portfolio-level analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .black_scholes import BlackScholesModel, OptionType
from .greeks import GreeksCalculator


@dataclass
class ScenarioResult:
  """Single scenario outcome."""

  scenario_name: str
  spot: float
  volatility: float
  time_to_expiry: float
  risk_free_rate: float
  option_price: float
  delta: float
  pnl: float  # Change from base case


@dataclass
class SensitivityResult:
  """Sensitivity analysis result for one risk factor."""

  factor_name: str
  factor_values: np.ndarray
  option_prices: np.ndarray
  deltas: np.ndarray
  gammas: np.ndarray


class ScenarioEngine:
  """
  Scenario analysis engine for option risk management.

  Supports one-factor and two-factor sensitivity grids, plus
  predefined stress scenarios (market crash, vol spike, etc.).
  """

  @staticmethod
  def base_case(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
  ) -> Dict[str, float]:
    """Compute base case price and Greeks."""
    price = BlackScholesModel.price(
      spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
    )
    greeks = GreeksCalculator.compute(
      spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
    )
    return {"price": price, **greeks.to_dict()}

  @classmethod
  def spot_sensitivity(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    spot_range_pct: float = 0.30,
    num_points: int = 50,
    dividend_yield: float = 0.0,
  ) -> SensitivityResult:
    """
    Sensitivity of option value to stock price changes.

    Args:
        spot_range_pct: Range as +/- percentage of current spot
        num_points: Number of grid points

    Returns:
        SensitivityResult with prices and Greeks across spot range
    """
    spot_min = spot * (1 - spot_range_pct)
    spot_max = spot * (1 + spot_range_pct)
    spots = np.linspace(spot_min, spot_max, num_points)

    prices = np.empty(num_points)
    deltas = np.empty(num_points)
    gammas = np.empty(num_points)

    for i, s in enumerate(spots):
      prices[i] = BlackScholesModel.price(
        s, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
      )
      g = GreeksCalculator.compute(
        s, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
      )
      deltas[i] = g.delta
      gammas[i] = g.gamma

    return SensitivityResult(
      factor_name="spot",
      factor_values=spots,
      option_prices=prices,
      deltas=deltas,
      gammas=gammas,
    )

  @classmethod
  def volatility_sensitivity(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    vol_range: float = 0.20,
    num_points: int = 50,
    dividend_yield: float = 0.0,
  ) -> SensitivityResult:
    """Sensitivity of option value to volatility changes (+/- vol_range)."""
    vol_min = max(volatility - vol_range, 0.01)
    vol_max = volatility + vol_range
    vols = np.linspace(vol_min, vol_max, num_points)

    prices = np.empty(num_points)
    deltas = np.empty(num_points)
    gammas = np.empty(num_points)

    for i, v in enumerate(vols):
      prices[i] = BlackScholesModel.price(
        spot, strike, time_to_expiry, risk_free_rate, v, option_type, dividend_yield
      )
      g = GreeksCalculator.compute(
        spot, strike, time_to_expiry, risk_free_rate, v, option_type, dividend_yield
      )
      deltas[i] = g.delta
      gammas[i] = g.gamma

    return SensitivityResult(
      factor_name="volatility",
      factor_values=vols,
      option_prices=prices,
      deltas=deltas,
      gammas=gammas,
    )

  @classmethod
  def time_sensitivity(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    num_points: int = 50,
    dividend_yield: float = 0.0,
  ) -> SensitivityResult:
    """Time decay profile: option value vs time to expiration."""
    times = np.linspace(max(time_to_expiry * 0.01, 1 / 365), time_to_expiry, num_points)

    prices = np.empty(num_points)
    deltas = np.empty(num_points)
    gammas = np.empty(num_points)

    for i, t in enumerate(times):
      prices[i] = BlackScholesModel.price(
        spot, strike, t, risk_free_rate, volatility, option_type, dividend_yield
      )
      g = GreeksCalculator.compute(
        spot, strike, t, risk_free_rate, volatility, option_type, dividend_yield
      )
      deltas[i] = g.delta
      gammas[i] = g.gamma

    return SensitivityResult(
      factor_name="time",
      factor_values=times,
      option_prices=prices,
      deltas=deltas,
      gammas=gammas,
    )

  @classmethod
  def rate_sensitivity(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    rate_range: float = 0.03,
    num_points: int = 50,
    dividend_yield: float = 0.0,
  ) -> SensitivityResult:
    """Sensitivity of option value to interest rate changes."""
    rate_min = max(risk_free_rate - rate_range, 0.0)
    rate_max = risk_free_rate + rate_range
    rates = np.linspace(rate_min, rate_max, num_points)

    prices = np.empty(num_points)
    deltas = np.empty(num_points)
    gammas = np.empty(num_points)

    for i, r in enumerate(rates):
      prices[i] = BlackScholesModel.price(
        spot, strike, time_to_expiry, r, volatility, option_type, dividend_yield
      )
      g = GreeksCalculator.compute(
        spot, strike, time_to_expiry, r, volatility, option_type, dividend_yield
      )
      deltas[i] = g.delta
      gammas[i] = g.gamma

    return SensitivityResult(
      factor_name="rate",
      factor_values=rates,
      option_prices=prices,
      deltas=deltas,
      gammas=gammas,
    )

  @classmethod
  def run_scenarios(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    scenarios: Optional[List[Dict]] = None,
    dividend_yield: float = 0.0,
  ) -> List[ScenarioResult]:
    """
    Run predefined or custom stress scenarios.

    Default scenarios:
        - Base case
        - Spot -10%, -20%
        - Spot +10%, +20%
        - Vol +5%, +10%
        - Vol -5%
        - Rate +1%, -1%
        - 30 days elapsed
    """
    if scenarios is None:
      scenarios = [
        {"name": "Base Case", "spot_mult": 1.0, "vol_shift": 0.0, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Spot -10%", "spot_mult": 0.90, "vol_shift": 0.0, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Spot -20%", "spot_mult": 0.80, "vol_shift": 0.0, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Spot +10%", "spot_mult": 1.10, "vol_shift": 0.0, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Spot +20%", "spot_mult": 1.20, "vol_shift": 0.0, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Vol +5%", "spot_mult": 1.0, "vol_shift": 0.05, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Vol +10%", "spot_mult": 1.0, "vol_shift": 0.10, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Vol -5%", "spot_mult": 1.0, "vol_shift": -0.05, "rate_shift": 0.0, "time_shift": 0.0},
        {"name": "Rate +1%", "spot_mult": 1.0, "vol_shift": 0.0, "rate_shift": 0.01, "time_shift": 0.0},
        {"name": "Rate -1%", "spot_mult": 1.0, "vol_shift": 0.0, "rate_shift": -0.01, "time_shift": 0.0},
        {"name": "30 Days Elapsed", "spot_mult": 1.0, "vol_shift": 0.0, "rate_shift": 0.0, "time_shift": 30 / 365},
      ]

    base_price = BlackScholesModel.price(
      spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
    )

    results: List[ScenarioResult] = []
    for sc in scenarios:
      s_spot = spot * sc.get("spot_mult", 1.0)
      s_vol = max(volatility + sc.get("vol_shift", 0.0), 0.01)
      s_rate = max(risk_free_rate + sc.get("rate_shift", 0.0), 0.0)
      s_time = max(time_to_expiry - sc.get("time_shift", 0.0), 1 / 365)

      price = BlackScholesModel.price(
        s_spot, strike, s_time, s_rate, s_vol, option_type, dividend_yield
      )
      greeks = GreeksCalculator.compute(
        s_spot, strike, s_time, s_rate, s_vol, option_type, dividend_yield
      )

      results.append(
        ScenarioResult(
          scenario_name=sc["name"],
          spot=s_spot,
          volatility=s_vol,
          time_to_expiry=s_time,
          risk_free_rate=s_rate,
          option_price=price,
          delta=greeks.delta,
          pnl=price - base_price,
        )
      )

    return results

  @classmethod
  def scenario_grid(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    spot_shocks: np.ndarray,
    vol_shocks: np.ndarray,
    dividend_yield: float = 0.0,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-factor scenario grid: spot shock vs volatility shock.

    Returns:
        (spot_mesh, vol_mesh, price_surface) for heatmap visualization
    """
    spot_mesh, vol_mesh = np.meshgrid(spot_shocks, vol_shocks)
    price_surface = np.zeros_like(spot_mesh)

    for i in range(spot_mesh.shape[0]):
      for j in range(spot_mesh.shape[1]):
        s = spot * (1.0 + spot_mesh[i, j])
        v = max(volatility + vol_mesh[i, j], 0.01)
        price_surface[i, j] = BlackScholesModel.price(
          s, strike, time_to_expiry, risk_free_rate, v, option_type, dividend_yield
        )

    return spot_mesh, vol_mesh, price_surface

  @staticmethod
  def scenarios_to_dataframe(results: List[ScenarioResult]) -> pd.DataFrame:
    """Convert scenario results to a pandas DataFrame for display."""
    return pd.DataFrame(
      [
        {
          "Scenario": r.scenario_name,
          "Spot": r.spot,
          "Volatility": r.volatility,
          "Time (Yrs)": r.time_to_expiry,
          "Rate": r.risk_free_rate,
          "Price": r.option_price,
          "Delta": r.delta,
          "P&L": r.pnl,
        }
        for r in results
      ]
    )
