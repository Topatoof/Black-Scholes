"""
Black-Scholes-Merton European option pricing model.

Implements closed-form solutions for European call and put options with
continuous dividend yield and continuous compounding of interest rates.

Mathematical assumptions:
    - Geometric Brownian Motion for the underlying asset price
    - Constant risk-free rate r and dividend yield q over [0, T]
    - Constant volatility sigma over [0, T]
    - No transaction costs, taxes, or short-selling restrictions
    - Continuous trading and no arbitrage
    - European exercise only (no early exercise)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    """Supported European option types."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class OptionParams:
  """
  Container for Black-Scholes input parameters.

  Attributes:
      spot: Current underlying price S_0
      strike: Strike price K
      time_to_expiry: Time to expiration T in years
      risk_free_rate: Continuously compounded risk-free rate r
      volatility: Annualized volatility sigma
      dividend_yield: Continuous dividend yield q (default 0)
  """

  spot: float
  strike: float
  time_to_expiry: float
  risk_free_rate: float
  volatility: float
  dividend_yield: float = 0.0

  def validate(self) -> None:
    """Validate parameter ranges and raise ValueError on invalid inputs."""
    if self.spot <= 0:
      raise ValueError(f"Spot price must be positive, got {self.spot}")
    if self.strike <= 0:
      raise ValueError(f"Strike price must be positive, got {self.strike}")
    if self.time_to_expiry < 0:
      raise ValueError(f"Time to expiry must be non-negative, got {self.time_to_expiry}")
    if self.volatility < 0:
      raise ValueError(f"Volatility must be non-negative, got {self.volatility}")
    if self.dividend_yield < 0:
      raise ValueError(f"Dividend yield must be non-negative, got {self.dividend_yield}")


class BlackScholesModel:
  """
  Black-Scholes-Merton model for European option valuation.

  Uses the standard formulas with continuous compounding:
      d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
      d2 = d1 - sigma * sqrt(T)

      Call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
      Put  = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
  """

  @staticmethod
  def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
  ) -> Tuple[float, float]:
    """
    Compute d1 and d2 terms used throughout Black-Scholes formulas.

    At expiry (T=0), returns limiting values for numerical stability.
    """
    if time_to_expiry <= 0:
      # At expiration, d1/d2 are not used; price equals intrinsic value
      return (0.0, 0.0)

    if volatility <= 0:
      # Zero-volatility limit: deterministic forward price
      forward_moneyness = math.log(spot / strike) + (risk_free_rate - dividend_yield) * time_to_expiry
      sign = 1.0 if forward_moneyness >= 0 else -1.0
      return (sign * 1e10, sign * 1e10)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
      math.log(spot / strike)
      + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2

  @staticmethod
  def price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
  ) -> float:
    """
    Calculate European option price using Black-Scholes-Merton formula.

    Args:
        spot: Current asset price S
        strike: Strike price K
        time_to_expiry: Time to expiration in years T
        risk_free_rate: Continuously compounded risk-free rate r
        volatility: Annualized volatility sigma
        option_type: CALL or PUT
        dividend_yield: Continuous dividend yield q

    Returns:
        Option fair value
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

    # At expiration, option value equals intrinsic value
    if time_to_expiry <= 0:
      if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
      return max(strike - spot, 0.0)

    d1, d2 = BlackScholesModel._d1_d2(
      spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )

    discount_asset = math.exp(-dividend_yield * time_to_expiry)
    discount_bond = math.exp(-risk_free_rate * time_to_expiry)

    if option_type == OptionType.CALL:
      return spot * discount_asset * norm.cdf(d1) - strike * discount_bond * norm.cdf(d2)

    return strike * discount_bond * norm.cdf(-d2) - spot * discount_asset * norm.cdf(-d1)

  @staticmethod
  def price_vectorized(
    spots: np.ndarray,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
  ) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing for sensitivity analysis and charts.

    Args:
        spots: Array of spot prices
        Other args: Same as price()

    Returns:
        Array of option prices corresponding to each spot price
    """
    spots = np.asarray(spots, dtype=float)
    prices = np.empty_like(spots)

    for i, s in enumerate(spots.flat):
      prices.flat[i] = BlackScholesModel.price(
        s, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
      )
    return prices

  @staticmethod
  def intrinsic_value(
    spot: float,
    strike: float,
    option_type: OptionType,
  ) -> float:
    """Calculate intrinsic (exercise) value of the option."""
    if option_type == OptionType.CALL:
      return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)

  @staticmethod
  def payoff_at_expiry(
    spot_prices: np.ndarray,
    strike: float,
    option_type: OptionType,
    premium: float = 0.0,
  ) -> np.ndarray:
    """
    Compute option payoff diagram at expiration (including premium paid).

    Args:
        spot_prices: Range of terminal spot prices
        strike: Strike price K
        option_type: CALL or PUT
        premium: Option premium subtracted from payoff (for P&L diagram)

    Returns:
        Net payoff array (intrinsic - premium)
    """
    spot_prices = np.asarray(spot_prices, dtype=float)
    if option_type == OptionType.CALL:
      intrinsic = np.maximum(spot_prices - strike, 0.0)
    else:
      intrinsic = np.maximum(strike - spot_prices, 0.0)
    return intrinsic - premium
