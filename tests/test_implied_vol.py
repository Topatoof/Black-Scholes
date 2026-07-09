"""Unit tests for implied volatility solver."""

import pytest

from src.black_scholes import BlackScholesModel, OptionType
from src.implied_vol import ImpliedVolatilitySolver, SolverMethod


SPOT = 100.0
STRIKE = 100.0
T = 1.0
R = 0.05
TRUE_SIGMA = 0.25
Q = 0.0


class TestImpliedVolatilitySolver:
  """Tests for IV inversion."""

  @pytest.fixture
  def market_price(self):
    return BlackScholesModel.price(SPOT, STRIKE, T, R, TRUE_SIGMA, OptionType.CALL, Q)

  def test_newton_raphson_recovers_vol(self, market_price):
    result = ImpliedVolatilitySolver.newton_raphson(
      market_price, SPOT, STRIKE, T, R, OptionType.CALL, Q
    )
    assert result.converged
    assert result.implied_vol == pytest.approx(TRUE_SIGMA, rel=1e-6)

  def test_brent_recovers_vol(self, market_price):
    result = ImpliedVolatilitySolver.brent(
      market_price, SPOT, STRIKE, T, R, OptionType.CALL, Q
    )
    assert result.converged
    assert result.implied_vol == pytest.approx(TRUE_SIGMA, rel=1e-6)

  def test_solve_unified(self, market_price):
    result = ImpliedVolatilitySolver.solve(
      market_price, SPOT, STRIKE, T, R, OptionType.CALL, Q
    )
    assert result.converged
    assert result.implied_vol == pytest.approx(TRUE_SIGMA, rel=1e-5)

  def test_put_implied_vol(self):
    price = BlackScholesModel.price(SPOT, STRIKE, T, R, TRUE_SIGMA, OptionType.PUT, Q)
    result = ImpliedVolatilitySolver.solve(price, SPOT, STRIKE, T, R, OptionType.PUT, Q)
    assert result.implied_vol == pytest.approx(TRUE_SIGMA, rel=1e-5)

  def test_negative_price_raises(self):
    with pytest.raises(ValueError):
      ImpliedVolatilitySolver.solve(-1.0, SPOT, STRIKE, T, R, OptionType.CALL)

  def test_zero_expiry_raises(self):
    with pytest.raises(ValueError):
      ImpliedVolatilitySolver.solve(5.0, SPOT, STRIKE, 0.0, R, OptionType.CALL)

  def test_volatility_smile(self):
    strikes = [90.0, 100.0, 110.0]
    prices = [
      BlackScholesModel.price(SPOT, k, T, R, TRUE_SIGMA + 0.02 * (k - SPOT) / SPOT, OptionType.CALL)
      for k in strikes
    ]
    ivs = ImpliedVolatilitySolver.volatility_smile(strikes, prices, SPOT, T, R, OptionType.CALL)
    assert all(iv is not None for iv in ivs)
