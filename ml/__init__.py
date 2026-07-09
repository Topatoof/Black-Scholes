"""Machine learning module for stock return prediction."""

from .backtest import BacktestResult, WalkForwardBacktester
from .features import FEATURE_COLUMNS, TARGET_COLUMN, FeatureConfig, FeatureEngineer
from .xgboost_model import ModelMetrics, ReturnPredictor, TrainResult, XGBoostConfig

__all__ = [
  "FeatureEngineer",
  "FeatureConfig",
  "FEATURE_COLUMNS",
  "TARGET_COLUMN",
  "ReturnPredictor",
  "XGBoostConfig",
  "ModelMetrics",
  "TrainResult",
  "WalkForwardBacktester",
  "BacktestResult",
]
