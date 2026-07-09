"""Unit tests for binomial tree pricer."""

import pytest

from src.black_scholes import BlackScholesModel, OptionType
from src.binomial_tree import BinomialTreePricer


SPOT = 100.0
STRIKE = 100.0
T = 1.0
R = 0.05
SIGMA = 0.20
Q = 0.0


class TestBinomialTreePricer:
  """Tests for CRR binomial tree."""

  def test_converges_to_bs(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    bin_result = BinomialTreePricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 500, Q
    )
    assert abs(bin_result.price - bs) < 0.05

  def test_put_converges_to_bs(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    bin_result = BinomialTreePricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, 500, Q
    )
    assert abs(bin_result.price - bs) < 0.05

  def test_more_steps_better_accuracy(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    r50 = BinomialTreePricer.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 50, Q)
    r500 = BinomialTreePricer.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 500, Q)
    assert abs(r500.price - bs) <= abs(r50.price - bs)

  def test_at_expiry_intrinsic(self):
    result = BinomialTreePricer.price(110.0, 100.0, 0.0, R, SIGMA, OptionType.CALL)
    assert result.price == 10.0

  def test_return_tree(self):
    result = BinomialTreePricer.price(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 10, Q, return_tree=True
    )
    assert result.stock_tree is not None
    assert result.option_tree is not None
    assert result.stock_tree.shape == (11, 11)

  def test_invalid_steps_raises(self):
    with pytest.raises(ValueError):
      BinomialTreePricer.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, 0)

  def test_convergence_analysis(self):
    bs = BlackScholesModel.price(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    steps, prices, errors = BinomialTreePricer.convergence_analysis(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, [50, 100, 200], Q, bs
    )
    assert errors[-1] <= errors[0]
