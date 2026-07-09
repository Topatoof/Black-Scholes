"""
Option Greeks calculator for the Black-Scholes-Merton model.

All sensitivities are computed analytically from closed-form derivatives.
Theta is reported per calendar day (divide annual theta by 365).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import norm

from .black_scholes import BlackScholesModel, OptionParams, OptionType


@dataclass(frozen=True)
class Greeks:
  """Container for all first-order and second-order option sensitivities."""

  delta: float
  gamma: float
  vega: float
  theta: float  # Per calendar day
  rho: float

  def to_dict(self) -> Dict[str, float]:
    """Convert Greeks to a dictionary for display/export."""
    return {
      "delta": self.delta,
      "gamma": self.gamma,
      "vega": self.vega,
      "theta": self.theta,
      "rho": self.rho,
    }


class GreeksCalculator:
  """
  Analytical Greeks for European options under Black-Scholes-Merton.

  Formulas (with continuous dividend yield q):

  Delta:
      Call: e^(-qT) * N(d1)
      Put:  e^(-qT) * [N(d1) - 1]

  Gamma (same for call and put):
      e^(-qT) * n(d1) / (S * sigma * sqrt(T))

  Vega (per 1% vol move, divide raw vega by 100):
      S * e^(-qT) * n(d1) * sqrt(T) / 100

  Theta (annual, then /365 for daily):
      Call: -[S*sigma*e^(-qT)*n(d1)]/(2*sqrt(T)) - r*K*e^(-rT)*N(d2) + q*S*e^(-qT)*N(d1)
      Put:  -[S*sigma*e^(-qT)*n(d1)]/(2*sqrt(T)) + r*K*e^(-rT)*N(-d2) - q*S*e^(-qT)*N(-d1)

  Rho (per 1% rate move):
      Call: K * T * e^(-rT) * N(d2) / 100
      Put:  -K * T * e^(-rT) * N(-d2) / 100

  where n(x) is the standard normal PDF and N(x) is the CDF.
  """

  @staticmethod
  def compute(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
  ) -> Greeks:
    """
    Compute all Greeks for a single option contract.

    Returns:
        Greeks dataclass with delta, gamma, vega, theta (daily), rho
    """
    params = OptionParams(
      spot=spot,
      strike=strike,
      time_to_expiry=time_to_expiry,
      risk_free_rate=risk_free_rate,
      volatility=volatility,
      dividend_yield=dividend_yield,
    )
    params.validate()

    if time_to_expiry <= 0 or volatility <= 0:
      # Degenerate case at expiry or zero vol
      delta = 1.0 if (option_type == OptionType.CALL and spot > strike) else 0.0
      if option_type == OptionType.PUT and spot < strike:
        delta = -1.0
      return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = BlackScholesModel._d1_d2(
      spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )

    sqrt_t = math.sqrt(time_to_expiry)
    discount_asset = math.exp(-dividend_yield * time_to_expiry)
    discount_bond = math.exp(-risk_free_rate * time_to_expiry)
    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type == OptionType.CALL:
      delta = discount_asset * norm.cdf(d1)
    else:
      delta = discount_asset * (norm.cdf(d1) - 1.0)

    # Gamma (identical for calls and puts)
    gamma = discount_asset * pdf_d1 / (spot * volatility * sqrt_t)

    # Vega: sensitivity to 1% change in volatility
    vega = spot * discount_asset * pdf_d1 * sqrt_t / 100.0

    # Theta: annualized time decay, converted to per-day
    common_theta = -(spot * volatility * discount_asset * pdf_d1) / (2.0 * sqrt_t)
    if option_type == OptionType.CALL:
      theta_annual = (
        common_theta
        - risk_free_rate * strike * discount_bond * norm.cdf(d2)
        + dividend_yield * spot * discount_asset * norm.cdf(d1)
      )
    else:
      theta_annual = (
        common_theta
        + risk_free_rate * strike * discount_bond * norm.cdf(-d2)
        - dividend_yield * spot * discount_asset * norm.cdf(-d1)
      )
    theta = theta_annual / 365.0

    # Rho: sensitivity to 1% change in interest rate
    if option_type == OptionType.CALL:
      rho = strike * time_to_expiry * discount_bond * norm.cdf(d2) / 100.0
    else:
      rho = -strike * time_to_expiry * discount_bond * norm.cdf(-d2) / 100.0

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

  @staticmethod
  def compute_curve(
    spots: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    greek: str = "delta",
  ) -> np.ndarray:
    """
    Compute a Greek sensitivity curve over a range of spot prices.

    Args:
        spots: Array of spot prices
        greek: One of 'delta', 'gamma', 'vega', 'theta', 'rho'

    Returns:
        Array of Greek values
    """
    spots = np.asarray(spots, dtype=float)
    values = np.empty_like(spots)

    for i, s in enumerate(spots.flat):
      g = GreeksCalculator.compute(
        s, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
      )
      values.flat[i] = getattr(g, greek)

    return values
