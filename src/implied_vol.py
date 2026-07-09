"""
Implied volatility solver for European options.

Inverts the Black-Scholes pricing function to find the volatility sigma
such that BS(S, K, T, r, q, sigma) = market_price.

Methods:
    - Newton-Raphson: Uses vega as the derivative for fast quadratic convergence
    - Brent: Robust bracketing method, guaranteed convergence if bracket is valid
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from scipy.optimize import brentq
from scipy.stats import norm

from .black_scholes import BlackScholesModel, OptionType


class SolverMethod(Enum):
  """Available implied volatility solvers."""

  NEWTON_RAPHSON = "newton_raphson"
  BRENT = "brent"


@dataclass
class ImpliedVolResult:
  """Result container for implied volatility computation."""

  implied_vol: float
  iterations: int
  converged: bool
  method: SolverMethod
  final_error: float


class ImpliedVolatilitySolver:
  """
  Solve for implied volatility given market option price.

  Newton-Raphson update:
      sigma_{n+1} = sigma_n - [BS(sigma_n) - price_market] / vega_raw(sigma_n)

  where vega_raw = S * e^(-qT) * n(d1) * sqrt(T)  (per unit vol, not per 1%)

  Brent's method:
      Finds root of f(sigma) = BS(sigma) - market_price on [sigma_min, sigma_max]
      using bisection, secant, and inverse quadratic interpolation.
  """

  DEFAULT_INITIAL_GUESS = 0.20
  MIN_VOL = 1e-6
  MAX_VOL = 5.0
  TOLERANCE = 1e-8
  MAX_ITERATIONS = 100

  @staticmethod
  def _bs_price_and_vega_raw(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    sigma: float,
    option_type: OptionType,
    dividend_yield: float,
  ) -> Tuple[float, float]:
    """Return Black-Scholes price and raw vega (dPrice/dSigma)."""
    if time_to_expiry <= 0:
      return BlackScholesModel.price(
        spot, strike, time_to_expiry, risk_free_rate, sigma, option_type, dividend_yield
      ), 0.0

    d1, d2 = BlackScholesModel._d1_d2(
      spot, strike, time_to_expiry, risk_free_rate, sigma, dividend_yield
    )
    sqrt_t = math.sqrt(time_to_expiry)
    discount_asset = math.exp(-dividend_yield * time_to_expiry)
    discount_bond = math.exp(-risk_free_rate * time_to_expiry)

    if option_type == OptionType.CALL:
      price = spot * discount_asset * norm.cdf(d1) - strike * discount_bond * norm.cdf(d2)
    else:
      price = strike * discount_bond * norm.cdf(-d2) - spot * discount_asset * norm.cdf(-d1)

    vega_raw = spot * discount_asset * norm.pdf(d1) * sqrt_t
    return price, vega_raw

  @staticmethod
  def _price_difference(
    sigma: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    market_price: float,
    option_type: OptionType,
    dividend_yield: float,
  ) -> float:
    """Objective function: BS price minus market price."""
    price, _ = ImpliedVolatilitySolver._bs_price_and_vega_raw(
      spot, strike, time_to_expiry, risk_free_rate, sigma, option_type, dividend_yield
    )
    return price - market_price

  @classmethod
  def newton_raphson(
    cls,
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    initial_guess: float = DEFAULT_INITIAL_GUESS,
    tolerance: float = TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
  ) -> ImpliedVolResult:
    """
    Solve implied volatility using Newton-Raphson iteration.

    Args:
        market_price: Observed market option price
        initial_guess: Starting volatility estimate
        tolerance: Convergence threshold on price error
        max_iterations: Maximum NR iterations

    Returns:
        ImpliedVolResult with solved volatility and convergence info
    """
    if market_price <= 0:
      raise ValueError("Market price must be positive for implied vol solver")
    if time_to_expiry <= 0:
      raise ValueError("Cannot solve implied vol at expiration")

    sigma = initial_guess
    for iteration in range(1, max_iterations + 1):
      price, vega_raw = cls._bs_price_and_vega_raw(
        spot, strike, time_to_expiry, risk_free_rate, sigma, option_type, dividend_yield
      )
      error = price - market_price

      if abs(error) < tolerance:
        return ImpliedVolResult(
          implied_vol=sigma,
          iterations=iteration,
          converged=True,
          method=SolverMethod.NEWTON_RAPHSON,
          final_error=error,
        )

      if vega_raw < 1e-12:
        # Vega too small; NR cannot proceed
        break

      sigma -= error / vega_raw
      sigma = max(cls.MIN_VOL, min(sigma, cls.MAX_VOL))

    # Fallback: did not converge
    final_price, _ = cls._bs_price_and_vega_raw(
      spot, strike, time_to_expiry, risk_free_rate, sigma, option_type, dividend_yield
    )
    return ImpliedVolResult(
      implied_vol=sigma,
      iterations=max_iterations,
      converged=False,
      method=SolverMethod.NEWTON_RAPHSON,
      final_error=final_price - market_price,
    )

  @classmethod
  def brent(
    cls,
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    vol_low: float = MIN_VOL,
    vol_high: float = MAX_VOL,
    tolerance: float = TOLERANCE,
  ) -> ImpliedVolResult:
    """
    Solve implied volatility using Brent's root-finding method.

    Brent's method combines bisection, secant, and inverse quadratic
    interpolation for robust convergence without requiring derivatives.

    Args:
        vol_low, vol_high: Bracket for the root search

    Returns:
        ImpliedVolResult with solved volatility
    """
    if market_price <= 0:
      raise ValueError("Market price must be positive for implied vol solver")
    if time_to_expiry <= 0:
      raise ValueError("Cannot solve implied vol at expiration")

    f_low = cls._price_difference(
      vol_low, spot, strike, time_to_expiry, risk_free_rate, market_price, option_type, dividend_yield
    )
    f_high = cls._price_difference(
      vol_high, spot, strike, time_to_expiry, risk_free_rate, market_price, option_type, dividend_yield
    )

    if f_low * f_high > 0:
      raise ValueError(
        f"No root in bracket [{vol_low}, {vol_high}]. "
        f"BS({vol_low})-market={f_low:.6f}, BS({vol_high})-market={f_high:.6f}"
      )

    implied_vol = brentq(
      cls._price_difference,
      vol_low,
      vol_high,
      args=(spot, strike, time_to_expiry, risk_free_rate, market_price, option_type, dividend_yield),
      xtol=tolerance,
      rtol=tolerance,
      maxiter=cls.MAX_ITERATIONS,
    )

    final_error = cls._price_difference(
      implied_vol, spot, strike, time_to_expiry, risk_free_rate, market_price, option_type, dividend_yield
    )

    return ImpliedVolResult(
      implied_vol=implied_vol,
      iterations=0,  # scipy brentq does not expose iteration count easily
      converged=True,
      method=SolverMethod.BRENT,
      final_error=final_error,
    )

  @classmethod
  def solve(
    cls,
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    method: SolverMethod = SolverMethod.NEWTON_RAPHSON,
    initial_guess: float = DEFAULT_INITIAL_GUESS,
  ) -> ImpliedVolResult:
    """
    Unified entry point for implied volatility solving.

    Falls back to Brent if Newton-Raphson fails to converge.
    """
    if method == SolverMethod.BRENT:
      return cls.brent(
        market_price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield
      )

    result = cls.newton_raphson(
      market_price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield, initial_guess
    )
    if not result.converged:
      try:
        result = cls.brent(
          market_price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield
        )
      except ValueError:
        pass
    return result

  @staticmethod
  def volatility_smile(
    strikes: list[float],
    market_prices: list[float],
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    method: SolverMethod = SolverMethod.NEWTON_RAPHSON,
  ) -> list[Optional[float]]:
    """
    Compute implied volatilities across multiple strikes (volatility smile).

    Returns:
        List of implied vols; None for strikes where solver fails
    """
    implied_vols: list[Optional[float]] = []
    for strike, price in zip(strikes, market_prices):
      try:
        result = ImpliedVolatilitySolver.solve(
          price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield, method
        )
        implied_vols.append(result.implied_vol if result.converged else None)
      except (ValueError, RuntimeError):
        implied_vols.append(None)
    return implied_vols
