"""Unit tests for scenario analysis engine."""

import numpy as np
import pytest

from src.black_scholes import OptionType
from src.scenario_analysis import ScenarioEngine


SPOT = 100.0
STRIKE = 100.0
T = 1.0
R = 0.05
SIGMA = 0.20


class TestScenarioEngine:
  """Tests for scenario and sensitivity analysis."""

  def test_base_case(self):
    result = ScenarioEngine.base_case(SPOT, STRIKE, T, R, SIGMA, OptionType.CALL)
    assert result["price"] > 0
    assert "delta" in result

  def test_spot_sensitivity(self):
    sens = ScenarioEngine.spot_sensitivity(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL
    )
    assert len(sens.option_prices) == 50
    assert sens.option_prices[-1] > sens.option_prices[0]

  def test_volatility_sensitivity(self):
    sens = ScenarioEngine.volatility_sensitivity(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL
    )
    assert sens.option_prices[-1] > sens.option_prices[0]

  def test_time_sensitivity(self):
    sens = ScenarioEngine.time_sensitivity(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL
    )
    assert sens.option_prices[0] < sens.option_prices[-1]

  def test_rate_sensitivity_call(self):
    sens = ScenarioEngine.rate_sensitivity(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL
    )
    assert sens.option_prices[-1] > sens.option_prices[0]

  def test_run_scenarios(self):
    results = ScenarioEngine.run_scenarios(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL
    )
    assert len(results) > 0
    base = [r for r in results if r.scenario_name == "Base Case"][0]
    assert base.pnl == pytest.approx(0.0)

  def test_scenario_grid(self):
    spot_shocks = np.linspace(-0.2, 0.2, 5)
    vol_shocks = np.linspace(-0.05, 0.05, 5)
    s_mesh, v_mesh, prices = ScenarioEngine.scenario_grid(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL, spot_shocks, vol_shocks
    )
    assert prices.shape == (5, 5)

  def test_scenarios_to_dataframe(self):
    results = ScenarioEngine.run_scenarios(
      SPOT, STRIKE, T, R, SIGMA, OptionType.CALL
    )
    df = ScenarioEngine.scenarios_to_dataframe(results)
    assert "P&L" in df.columns
    assert len(df) == len(results)
