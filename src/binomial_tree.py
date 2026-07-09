"""
Binomial tree model for European (and American) option pricing.

Uses the Cox-Ross-Rubinstein (CRR) parameterization:
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)

At each node, option value is discounted expected payoff:
    V = exp(-r * dt) * [p * V_up + (1-p) * V_down]

Converges to Black-Scholes as num_steps -> infinity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .black_scholes import OptionType


@dataclass
class BinomialResult:
  """Container for binomial tree pricing results."""

  price: float
  num_steps: int
  stock_tree: Optional[np.ndarray] = None
  option_tree: Optional[np.ndarray] = None


class BinomialTreePricer:
  """
  CRR binomial tree pricer for European options.

  The tree is built forward for stock prices and backward for option values.
  Supports continuous dividend yield and continuous compounding.
  """

  @staticmethod
  def _crr_parameters(
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    num_steps: int,
    dividend_yield: float = 0.0,
  ) -> tuple[float, float, float, float]:
    """
    Compute CRR tree parameters.

    Returns:
        (dt, u, d, p) where dt = T/n, u = up factor, d = down factor, p = risk-neutral prob
    """
    dt = time_to_expiry / num_steps
    u = np.exp(volatility * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((risk_free_rate - dividend_yield) * dt)
    p = (growth - d) / (u - d)

    if p < 0 or p > 1:
      raise ValueError(
        f"Risk-neutral probability {p:.4f} out of [0,1]. "
        "Check parameters or increase num_steps."
      )

    return dt, u, d, p

  @classmethod
  def price(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    num_steps: int = 100,
    dividend_yield: float = 0.0,
    return_tree: bool = False,
  ) -> BinomialResult:
    """
    Price a European option using the binomial tree model.

    Args:
        num_steps: Number of time steps in the tree
        return_tree: If True, return full stock and option trees

    Returns:
        BinomialResult with price and optionally the trees
    """
    if spot <= 0 or strike <= 0:
      raise ValueError("Spot and strike must be positive")
    if time_to_expiry <= 0:
      if option_type == OptionType.CALL:
        return BinomialResult(price=max(spot - strike, 0.0), num_steps=0)
      return BinomialResult(price=max(strike - spot, 0.0), num_steps=0)
    if num_steps <= 0:
      raise ValueError("Number of steps must be positive")

    dt, u, d, p = cls._crr_parameters(
      time_to_expiry, risk_free_rate, volatility, num_steps, dividend_yield
    )
    discount = np.exp(-risk_free_rate * dt)

    # Build stock price tree: S[i,j] = S_0 * u^j * d^(i-j)
    stock_tree = np.zeros((num_steps + 1, num_steps + 1))
    for i in range(num_steps + 1):
      for j in range(i + 1):
        stock_tree[i, j] = spot * (u**j) * (d ** (i - j))

    # Terminal option payoffs
    option_tree = np.zeros((num_steps + 1, num_steps + 1))
    for j in range(num_steps + 1):
      terminal_s = stock_tree[num_steps, j]
      if option_type == OptionType.CALL:
        option_tree[num_steps, j] = max(terminal_s - strike, 0.0)
      else:
        option_tree[num_steps, j] = max(strike - terminal_s, 0.0)

    # Backward induction
    for i in range(num_steps - 1, -1, -1):
      for j in range(i + 1):
        option_tree[i, j] = discount * (
          p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j]
        )

    return BinomialResult(
      price=float(option_tree[0, 0]),
      num_steps=num_steps,
      stock_tree=stock_tree if return_tree else None,
      option_tree=option_tree if return_tree else None,
    )

  @classmethod
  def convergence_analysis(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    step_counts: list[int],
    dividend_yield: float = 0.0,
    bs_reference: Optional[float] = None,
  ) -> tuple[list[int], list[float], list[float]]:
    """
    Price option at increasing tree depths to show convergence to BS.

    Returns:
        Tuple of (step_counts, prices, errors_vs_bs)
    """
    prices: list[float] = []
    errors: list[float] = []

    for n in step_counts:
      result = cls.price(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, n, dividend_yield
      )
      prices.append(result.price)
      if bs_reference is not None:
        errors.append(abs(result.price - bs_reference))
      else:
        errors.append(0.0)

    return step_counts, prices, errors
