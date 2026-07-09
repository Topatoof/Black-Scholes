"""
Walk-forward backtesting for return prediction models.

Simulates realistic out-of-sample evaluation by retraining (or using
a fixed model) on rolling windows and measuring prediction quality
and simple trading performance over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ml.features import TARGET_COLUMN, FeatureEngineer
from ml.xgboost_model import ModelMetrics, ReturnPredictor, XGBoostConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
  """Results from a walk-forward backtest."""

  predictions: pd.DataFrame
  metrics: ModelMetrics
  cumulative_strategy_return: float
  cumulative_buy_hold_return: float
  n_periods: int

  def summary(self) -> dict:
    """Return a summary dictionary of backtest performance."""
    return {
      "rmse": self.metrics.rmse,
      "mae": self.metrics.mae,
      "directional_accuracy": self.metrics.directional_accuracy,
      "cumulative_strategy_return": self.cumulative_strategy_return,
      "cumulative_buy_hold_return": self.cumulative_buy_hold_return,
      "n_periods": self.n_periods,
    }


class WalkForwardBacktester:
  """
  Walk-forward backtest for return prediction models.

  Methodology (no look-ahead):
      For each test window [t, t + step):
          1. Train on all data strictly before t
          2. Predict returns for [t, t + step)
          3. Record predictions vs actuals

  Strategy simulation (simple long/flat):
      Go long when predicted return > 0, flat otherwise.
      P&L = position * actual_return (no transaction costs).
  """

  def __init__(
    self,
    config: Optional[XGBoostConfig] = None,
    min_train_rows: int = 120,
    test_step: int = 21,
  ) -> None:
    """
    Args:
        config: XGBoost hyperparameters
        min_train_rows: Minimum training rows before first prediction
        test_step: Number of days per walk-forward test window
    """
    self.config = config or XGBoostConfig()
    self.min_train_rows = min_train_rows
    self.test_step = test_step
    self._engineer = FeatureEngineer()

  def run(
    self,
    ohlcv: pd.DataFrame,
    retrain_each_window: bool = True,
  ) -> BacktestResult:
    """
    Execute walk-forward backtest.

    Args:
        ohlcv: Full OHLCV history
        retrain_each_window: Retrain model each window (slower, more realistic)

    Returns:
        BacktestResult with predictions, metrics, and strategy returns
    """
    feature_df = self._engineer.build_features(ohlcv)
    X, y = self._engineer.split_features_target(feature_df)

    if len(X) < self.min_train_rows + self.test_step:
      raise ValueError(
        f"Need at least {self.min_train_rows + self.test_step} rows, got {len(X)}"
      )

    all_preds: List[float] = []
    all_actuals: List[float] = []
    all_dates: List[pd.Timestamp] = []

    start = self.min_train_rows
    predictor = ReturnPredictor(config=self.config)

    while start < len(X):
      end = min(start + self.test_step, len(X))
      X_train, y_train = X.iloc[:start], y.iloc[:start]
      X_test, y_test = X.iloc[start:end], y.iloc[start:end]

      if retrain_each_window:
        predictor = ReturnPredictor(config=self.config)
        # Inline train on chronological split (use all train data)
        val_size = max(int(len(X_train) * 0.15), 5)
        model = predictor._build_model()
        model.fit(
          X_train.iloc[:-val_size].values,
          y_train.iloc[:-val_size].values,
          eval_set=[(X_train.iloc[-val_size:].values, y_train.iloc[-val_size:].values)],
          verbose=False,
        )
        predictor._model = model
      elif not predictor.is_fitted:
        val_size = max(int(len(X_train) * 0.15), 5)
        model = predictor._build_model()
        model.fit(
          X_train.iloc[:-val_size].values,
          y_train.iloc[:-val_size].values,
          eval_set=[(X_train.iloc[-val_size:].values, y_train.iloc[-val_size:].values)],
          verbose=False,
        )
        predictor._model = model

      preds = predictor._model.predict(X_test.values)
      all_preds.extend(preds.tolist())
      all_actuals.extend(y_test.values.tolist())
      all_dates.extend(X_test.index.tolist())
      start = end

    pred_df = pd.DataFrame(
      {
        "predicted_return": all_preds,
        "actual_return": all_actuals,
      },
      index=pd.DatetimeIndex(all_dates),
    )

    metrics = ReturnPredictor.compute_metrics(
      pred_df["actual_return"].values,
      pred_df["predicted_return"].values,
    )

    # Simple long/flat strategy vs buy-and-hold
    positions = (pred_df["predicted_return"] > 0).astype(float)
    strategy_returns = positions * pred_df["actual_return"]
    cumulative_strategy = float((1 + strategy_returns).prod() - 1)
    cumulative_bh = float((1 + pred_df["actual_return"]).prod() - 1)

    return BacktestResult(
      predictions=pred_df,
      metrics=metrics,
      cumulative_strategy_return=cumulative_strategy,
      cumulative_buy_hold_return=cumulative_bh,
      n_periods=len(pred_df),
    )

  @staticmethod
  def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
  ) -> ModelMetrics:
    """Evaluate pre-computed predictions against actual returns."""
    aligned = pd.DataFrame({"actual": y_true, "predicted": y_pred}).dropna()
    return ReturnPredictor.compute_metrics(
      aligned["actual"].values,
      aligned["predicted"].values,
    )
