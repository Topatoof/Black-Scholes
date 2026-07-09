"""Unit tests for Black-Scholes pricing model."""

import math

import numpy as np
import pytest

from src.black_scholes import BlackScholesModel, OptionType


# Standard test parameters (ATM, 1-year)
SPOT = 100.0
STRIKE = 100.0
T = 1.0
R = 0.05
SIGMA = 0.20
Q = 0.02


class TestBlackScholesModel:
  """Tests for European option pricing."""

  def test_call_price_positive(self):
    price = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    assert price > 0

  def test_put_price_positive(self):
    price = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    assert price > 0

  def test_put_call_parity(self):
    """Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)"""
    call = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    put = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    lhs = call - put
    rhs = SPOT * math.exp(-Q * T) - STRIKE * math.exp(-R * T)
    assert abs(lhs - rhs) < 1e-10

  def test_at_expiry_call_itm(self):
    price = BlackScholesModel.price(110.0, 100.0, 0.0, R, SIGMA, OptionType.CALL)
    assert price == pytest.approx(10.0)

  def test_at_expiry_call_otm(self):
    price = BlackScholesModel.price(90.0, 100.0, 0.0, R, SIGMA, OptionType.CALL)
    assert price == 0.0

  def test_at_expiry_put_itm(self):
    price = BlackScholesModel.price(90.0, 100.0, 0.0, R, SIGMA, OptionType.PUT)
    assert price == pytest.approx(10.0)

  def test_call_increases_with_spot(self):
    p1 = BlackScholesModel.price(90.0, STRIKE, T, R, SIGMA, OptionType.CALL)
    p2 = BlackScholesModel.price(110.0, STRIKE, T, R, SIGMA, OptionType.CALL)
    assert p2 > p1

  def test_put_decreases_with_spot(self):
    p1 = BlackScholesModel.price(90.0, STRIKE, T, R, SIGMA, OptionType.PUT)
    p2 = BlackScholesModel.price(110.0, STRIKE, T, R, SIGMA, OptionType.PUT)
    assert p2 < p1

  def test_call_decreases_with_dividend(self):
    c0 = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 0.0)
    c1 = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 0.05)
    assert c1 < c0

  def test_price_vectorized(self):
    spots = np.array([80.0, 100.0, 120.0])
    prices = BlackScholesModel.price_vectorized(spots, STRIKE, T, R, SIGMA, OptionType.CALL)
    assert len(prices) == 3
    assert prices[2] > prices[1] > prices[0]

  def test_intrinsic_value(self):
    assert BlackScholesModel.intrinsic_value(110.0, 100.0, OptionType.CALL) == 10.0
    assert BlackScholesModel.intrinsic_value(90.0, 100.0, OptionType.PUT) == 10.0

  def test_payoff_at_expiry(self):
    spots = np.array([80.0, 100.0, 120.0])
    payoff = BlackScholesModel.payoff_at_expiry(spots, 100.0, OptionType.CALL, premium=5.0)
    assert payoff[0] == -5.0
    assert payoff[2] == pytest.approx(15.0)

  def test_invalid_spot_raises(self):
    with pytest.raises(ValueError):
      BlackScholesModel.price(-1.0, STRIKE, T, R, SIGMA, OptionType.CALL)

  def test_known_call_value(self):
    """Regression test against known BS call value."""
  # S=100, K=100, T=1, r=0.05, sigma=0.2, q=0
    price = BlackScholesModel.price(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
    assert price == pytest.approx(10.4506, rel=1e-3)
