"""Unit tests for walk-forward backtesting."""

import numpy as np
import pandas as pd
import pytest

from ml.backtest import WalkForwardBacktester
from ml.xgboost_model import XGBoostConfig


@pytest.fixture
def ohlcv_data():
  np.random.seed(7)
  n = 300
  dates = pd.date_range("2021-01-01", periods=n, freq="B")
  close = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n)))
  return pd.DataFrame(
    {
      "open": close,
      "high": close * 1.005,
      "low": close * 0.995,
      "close": close,
      "volume": np.random.randint(1e6, 3e6, n).astype(float),
    },
    index=dates,
  )


class TestWalkForwardBacktester:
  def test_run_backtest(self, ohlcv_data):
    config = XGBoostConfig(n_estimators=30, n_cv_splits=2)
    bt = WalkForwardBacktester(config=config, min_train_rows=100, test_step=30)
    result = bt.run(ohlcv_data, retrain_each_window=True)
    assert result.n_periods > 0
    assert "predicted_return" in result.predictions.columns
    assert "actual_return" in result.predictions.columns
    assert result.metrics.rmse >= 0

  def test_insufficient_data_raises(self, ohlcv_data):
    bt = WalkForwardBacktester(min_train_rows=500)
    with pytest.raises(ValueError):
      bt.run(ohlcv_data.iloc[:50])
