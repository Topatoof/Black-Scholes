"""
Monte Carlo simulation for European option pricing.

Uses Geometric Brownian Motion (GBM) to simulate terminal asset prices
and prices options via risk-neutral expectation with continuous compounding.

GBM dynamics under risk-neutral measure Q:
    dS = (r - q) S dt + sigma S dW

Exact discretization for terminal price:
    S_T = S_0 * exp[(r - q - sigma^2/2) * T + sigma * sqrt(T) * Z]
    where Z ~ N(0, 1)

Option price:
    V = e^(-rT) * E^Q[max(S_T - K, 0)]  (call)
    V = e^(-rT) * E^Q[max(K - S_T, 0)]  (put)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .black_scholes import BlackScholesModel, OptionType


@dataclass
class MonteCarloResult:
  """Container for Monte Carlo simulation results."""

  price: float
  standard_error: float
  confidence_interval_95: Tuple[float, float]
  num_simulations: int
  terminal_prices: Optional[np.ndarray] = None
  payoffs: Optional[np.ndarray] = None


class MonteCarloPricer:
  """
  Monte Carlo pricer for European options under GBM.

  Antithetic variates are optionally used to reduce variance by pairing
  each random draw Z with -Z.
  """

  @staticmethod
  def simulate_terminal_prices(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    num_simulations: int,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    seed: Optional[int] = None,
  ) -> np.ndarray:
    """
    Simulate terminal asset prices under GBM.

    Args:
        spot: Initial asset price S_0
        time_to_expiry: Time horizon T in years
        risk_free_rate: Risk-free rate r
        volatility: Annual volatility sigma
        num_simulations: Number of Monte Carlo paths
        dividend_yield: Continuous dividend yield q
        antithetic: Use antithetic variates for variance reduction
        seed: Random seed for reproducibility

    Returns:
        Array of simulated terminal prices S_T
    """
    if spot <= 0:
      raise ValueError("Spot price must be positive")
    if time_to_expiry < 0:
      raise ValueError("Time to expiry must be non-negative")
    if volatility < 0:
      raise ValueError("Volatility must be non-negative")
    if num_simulations <= 0:
      raise ValueError("Number of simulations must be positive")

    rng = np.random.default_rng(seed)

    n = num_simulations // 2 if antithetic else num_simulations
    z = rng.standard_normal(n)

    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    diffusion = volatility * np.sqrt(time_to_expiry)

    if antithetic:
      z_full = np.concatenate([z, -z])
    else:
      z_full = z

    terminal_prices = spot * np.exp(drift + diffusion * z_full)
    return terminal_prices

  @classmethod
  def price(
    cls,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    num_simulations: int = 100_000,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    seed: Optional[int] = None,
    return_paths: bool = False,
  ) -> MonteCarloResult:
    """
    Price a European option via Monte Carlo simulation.

    Args:
        return_paths: If True, include terminal prices and payoffs in result

    Returns:
        MonteCarloResult with price estimate, standard error, and CI
    """
    if time_to_expiry <= 0:
      price = BlackScholesModel.price(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, dividend_yield
      )
      return MonteCarloResult(
        price=price,
        standard_error=0.0,
        confidence_interval_95=(price, price),
        num_simulations=0,
      )

    terminal_prices = cls.simulate_terminal_prices(
      spot, time_to_expiry, risk_free_rate, volatility, num_simulations, dividend_yield, antithetic, seed
    )

    if option_type == OptionType.CALL:
      payoffs = np.maximum(terminal_prices - strike, 0.0)
    else:
      payoffs = np.maximum(strike - terminal_prices, 0.0)

    discount = np.exp(-risk_free_rate * time_to_expiry)
    discounted_payoffs = discount * payoffs

    mc_price = float(np.mean(discounted_payoffs))
    std_err = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs)))
    ci_low = mc_price - 1.96 * std_err
    ci_high = mc_price + 1.96 * std_err

    return MonteCarloResult(
      price=mc_price,
      standard_error=std_err,
      confidence_interval_95=(ci_low, ci_high),
      num_simulations=len(terminal_prices),
      terminal_prices=terminal_prices if return_paths else None,
      payoffs=discounted_payoffs if return_paths else None,
    )

  @staticmethod
  def convergence_analysis(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    simulation_counts: list[int],
    dividend_yield: float = 0.0,
    seed: Optional[int] = 42,
  ) -> Tuple[list[int], list[float], list[float]]:
    """
    Run MC at increasing simulation counts to demonstrate convergence.

    Returns:
        Tuple of (counts, prices, standard_errors)
    """
    counts: list[int] = []
    prices: list[float] = []
    errors: list[float] = []

    for n in simulation_counts:
      result = MonteCarloPricer.price(
        spot, strike, time_to_expiry, risk_free_rate, volatility, option_type, n, dividend_yield, True, seed
      )
      counts.append(n)
      prices.append(result.price)
      errors.append(result.standard_error)

    return counts, prices, errors
