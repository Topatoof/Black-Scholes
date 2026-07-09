"""Unit tests for option Greeks."""

import pytest

from src.black_scholes import OptionType
from src.greeks import GreeksCalculator


SPOT = 100.0
STRIKE = 100.0
T = 1.0
R = 0.05
SIGMA = 0.20
Q = 0.0


class TestGreeksCalculator:
  """Tests for analytical Greeks."""

  def test_call_delta_range(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    assert 0 < g.delta < 1

  def test_put_delta_range(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    assert -1 < g.delta < 0

  def test_gamma_positive(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    assert g.gamma > 0

  def test_gamma_same_for_call_put(self):
    gc = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    gp = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    assert gc.gamma == pytest.approx(gp.gamma)

  def test_vega_positive(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    assert g.vega > 0

  def test_call_rho_positive(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    assert g.rho > 0

  def test_put_rho_negative(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.PUT, Q)
    assert g.rho < 0

  def test_delta_numerical_check(self):
    """Verify delta via finite difference."""
    eps = 0.01
    from src.black_scholes import BlackScholesModel

    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    p_up = BlackScholesModel.price(SPOT + eps, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    p_dn = BlackScholesModel.price(SPOT - eps, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    numerical_delta = (p_up - p_dn) / (2 * eps)
    assert g.delta == pytest.approx(numerical_delta, rel=1e-3)

  def test_greeks_to_dict(self):
    g = GreeksCalculator.compute(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, Q)
    d = g.to_dict()
    assert set(d.keys()) == {"delta", "gamma", "vega", "theta", "rho"}

  def test_compute_curve(self):
    import numpy as np

    spots = np.linspace(80, 120, 10)
    curve = GreeksCalculator.compute_curve(
      spots, STRIKE, T, R, SIGMA, OptionType.CALL, Q, "delta"
    )
    assert len(curve) == 10
