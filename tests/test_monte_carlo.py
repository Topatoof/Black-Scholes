"""Unit tests for Monte Carlo pricer."""

import pytest

from src.black_scholes import BlackScholesModel, OptionType
from src.monte_carlo import MonteCarloPricer


SPOT = 100.0
STRIKE = 100.0
T = 1.0
R = 0.05
SIGMA = 0.20
Q = 0.0


class TestMonteCarloPricer:
  """Tests for MC simulation."""

  def test_mc_price_near_bs(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    mc = MonteCarloPricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 100_000, Q, True, 42
    )
    assert abs(mc.price - bs) < 0.15

  def test_mc_put_price(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    mc = MonteCarloPricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, 100_000, Q, True, 42
    )
    assert abs(mc.price - bs) < 0.15

  def test_standard_error_positive(self):
    mc = MonteCarloPricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 50_000, Q, True, 42
    )
    assert mc.standard_error > 0

  def test_confidence_interval_contains_bs(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    mc = MonteCarloPricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 200_000, Q, True, 42
    )
    assert mc.confidence_interval_95[0] <= bs <= mc.confidence_interval_95[1]

  def test_antithetic_reduces_variance(self):
    mc_no_anti = MonteCarloPricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 10_000, Q, False, 42
    )
    mc_anti = MonteCarloPricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 10_000, Q, True, 42
    )
    # Antithetic should generally have lower or similar std error
    assert mc_anti.standard_error <= mc_no_anti.standard_error * 1.5

  def test_terminal_prices_shape(self):
    prices = MonteCarloPricer.simulate_terminal_prices(SPOT, T, R, SIGMA, 1000, Q, True, 42)
    assert len(prices) == 1000

  def test_invalid_spot_raises(self):
    with pytest.raises(ValueError):
      MonteCarloPricer.simulate_terminal_prices(-1.0, T, R, SIGMA, 1000)

  def test_convergence_analysis(self):
    counts, prices, errors = MonteCarloPricer.convergence_analysis(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, [1000, 5000, 10000]
    )
    assert len(counts) == 3
    assert errors[-1] <= errors[0]  # More sims -> lower error
