"""Unit tests for volatility calculator."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.volatility import VolatilityCalculator


class TestVolatilityCalculator:
  """Tests for historical volatility analytics."""

  @pytest.fixture
  def price_series(self):
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    return pd.Series(prices, index=dates)

  def test_log_returns_length(self, price_series):
    returns = VolatilityCalculator.log_returns(price_series)
    assert len(returns) == len(price_series) - 1

  def test_historical_volatility(self, price_series):
    stats = VolatilityCalculator.historical_volatility(price_series)
    assert stats.daily_volatility > 0
    assert stats.annualized_volatility > stats.daily_volatility
    assert stats.num_observations == 251

  def test_rolling_volatility(self, price_series):
    rolling = VolatilityCalculator.rolling_volatility(price_series, window=21)
    assert rolling.dropna().iloc[-1] > 0

  def test_validate_empty_raises(self):
    with pytest.raises(ValueError):
      VolatilityCalculator.validate_price_data(pd.Series(dtype=float))

  def test_validate_negative_prices_raises(self, price_series):
    bad = price_series.copy()
    bad.iloc[5] = -1.0
    with pytest.raises(ValueError):
      VolatilityCalculator.validate_price_data(bad)

  def test_load_csv(self):
    sample = Path(__file__).parent.parent / "data" / "sample_prices.csv"
    prices = VolatilityCalculator.load_csv(str(sample))
    assert len(prices) > 0

  def test_load_csv_missing_column(self, tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("A,B\n1,2\n")
    with pytest.raises(ValueError):
      VolatilityCalculator.load_csv(str(f))

  def test_synthetic_smile(self):
    smile_fn = VolatilityCalculator.generate_synthetic_smile(100.0, 0.20, -0.1, 0.05)
    atm = smile_fn(100.0)
    otm = smile_fn(110.0)
    assert atm == pytest.approx(0.20)
    assert otm != atm

  def test_volatility_surface_grid(self):
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([0.5, 1.0])
    vols = np.ones((2, 3)) * 0.20
    k_mesh, t_mesh, v_surf = VolatilityCalculator.volatility_surface_grid(100.0, strikes, expiries, vols)
    assert k_mesh.shape == (2, 3)

  def test_invalid_surface_shape_raises(self):
    with pytest.raises(ValueError):
      VolatilityCalculator.volatility_surface_grid(
        100.0, np.array([90, 100]), np.array([0.5, 1.0]), np.ones((3, 3))
      )
