"""Unit tests for ML feature engineering."""

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, FeatureEngineer


@pytest.fixture
def ohlcv_data():
  """Synthetic OHLCV data with realistic structure."""
  np.random.seed(42)
  n = 200
  dates = pd.date_range("2023-01-01", periods=n, freq="B")
  close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
  high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
  low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
  open_ = close * (1 + np.random.normal(0, 0.003, n))
  volume = np.random.randint(500_000, 2_000_000, n).astype(float)
  return pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
    index=dates,
  )


class TestFeatureEngineer:
  def test_build_features_shape(self, ohlcv_data):
    engineer = FeatureEngineer()
    features = engineer.build_features(ohlcv_data)
    assert len(features) > 0
    for col in FEATURE_COLUMNS:
      assert col in features.columns
    assert TARGET_COLUMN in features.columns

  def test_no_lookahead_in_features(self, ohlcv_data):
    """Features at t must not change when future prices change."""
    engineer = FeatureEngineer()
    f1 = engineer.build_features(ohlcv_data)

    modified = ohlcv_data.copy()
    modified.iloc[-5:, modified.columns.get_loc("close")] *= 2.0
    f2 = engineer.build_features(modified)

    # Shared history (excluding last rows affected by target shift) should match
    common_idx = f1.index.intersection(f2.index)
    # Exclude last horizon rows where target differs
    stable_idx = common_idx[:-5]
    for col in FEATURE_COLUMNS:
      pd.testing.assert_series_equal(
        f1.loc[stable_idx, col],
        f2.loc[stable_idx, col],
        check_names=False,
        rtol=1e-10,
      )

  def test_rsi_bounds(self, ohlcv_data):
    engineer = FeatureEngineer()
    features = engineer.build_features(ohlcv_data)
    rsi = features["rsi_14"].dropna()
    assert rsi.min() >= 0
    assert rsi.max() <= 100

  def test_validate_ohlcv_missing_column(self):
    with pytest.raises(ValueError):
      FeatureEngineer.validate_ohlcv(pd.DataFrame({"close": [1, 2, 3]}))

  def test_split_features_target(self, ohlcv_data):
    engineer = FeatureEngineer()
    features = engineer.build_features(ohlcv_data)
    X, y = engineer.split_features_target(features)
    assert len(X) == len(y)
    assert list(X.columns) == FEATURE_COLUMNS
