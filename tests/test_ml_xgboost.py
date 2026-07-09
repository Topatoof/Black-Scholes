"""Unit tests for XGBoost return predictor."""

import numpy as np
import pandas as pd
import pytest

from ml.xgboost_model import ReturnPredictor, XGBoostConfig


@pytest.fixture
def ohlcv_data():
  np.random.seed(123)
  n = 250
  dates = pd.date_range("2022-01-01", periods=n, freq="B")
  close = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, n)))
  return pd.DataFrame(
    {
      "open": close * 0.999,
      "high": close * 1.01,
      "low": close * 0.99,
      "close": close,
      "volume": np.random.randint(1e6, 5e6, n).astype(float),
    },
    index=dates,
  )


@pytest.fixture
def small_config():
  return XGBoostConfig(n_estimators=50, n_cv_splits=3, test_size=0.2)


class TestReturnPredictor:
  def test_train_and_metrics(self, ohlcv_data, small_config):
    predictor = ReturnPredictor(config=small_config)
    result = predictor.train(ohlcv_data, save=False)
    assert result.metrics.n_samples > 0
    assert result.metrics.rmse >= 0
    assert result.metrics.mae >= 0
    assert 0 <= result.metrics.directional_accuracy <= 1
    assert len(result.cv_metrics) == small_config.n_cv_splits
    assert len(result.feature_importance) > 0

  def test_chronological_split_no_shuffle(self, ohlcv_data, small_config):
    from ml.features import FeatureEngineer

    engineer = FeatureEngineer()
    features = engineer.build_features(ohlcv_data)
    X, y = engineer.split_features_target(features)
    X_train, X_test, _, _ = ReturnPredictor.chronological_split(X, y, test_size=0.2)
    assert X_train.index.max() < X_test.index.min()

  def test_save_and_load(self, ohlcv_data, small_config, tmp_path):
    predictor = ReturnPredictor(config=small_config, model_dir=tmp_path)
    result = predictor.train(ohlcv_data, save=True, model_name="test_model")
    assert result.model_path.exists()

    loaded = ReturnPredictor(config=small_config, model_dir=tmp_path)
    loaded.load(result.model_path)
    assert loaded.is_fitted

    preds_orig = predictor.predict(ohlcv_data)
    preds_loaded = loaded.predict(ohlcv_data)
    pd.testing.assert_series_equal(preds_orig, preds_loaded)

  def test_directional_accuracy(self):
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.005, -0.01, -0.01, 0.02])
    metrics = ReturnPredictor.compute_metrics(y_true, y_pred)
    assert metrics.directional_accuracy == 0.5
