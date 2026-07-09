"""
XGBoost return prediction model.

Trains a gradient-boosted regressor to predict forward log returns
from engineered technical features. Uses chronological train/test
splitting and time-series cross-validation to prevent look-ahead bias.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, FeatureEngineer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models")


@dataclass
class ModelMetrics:
  """Evaluation metrics for return prediction."""

  rmse: float
  mae: float
  directional_accuracy: float
  n_samples: int

  def to_dict(self) -> Dict[str, float]:
    return {
      "rmse": self.rmse,
      "mae": self.mae,
      "directional_accuracy": self.directional_accuracy,
      "n_samples": self.n_samples,
    }


@dataclass
class TrainResult:
  """Container for training output."""

  metrics: ModelMetrics
  cv_metrics: List[ModelMetrics]
  feature_importance: pd.DataFrame
  model_path: Optional[Path] = None
  train_size: int = 0
  test_size: int = 0


@dataclass
class XGBoostConfig:
  """Hyperparameters for XGBoost regressor."""

  n_estimators: int = 300
  max_depth: int = 4
  learning_rate: float = 0.05
  subsample: float = 0.8
  colsample_bytree: float = 0.8
  min_child_weight: int = 5
  reg_alpha: float = 0.1
  reg_lambda: float = 1.0
  random_state: int = 42
  test_size: float = 0.2
  n_cv_splits: int = 5
  early_stopping_rounds: int = 20


class ReturnPredictor:
  """
  XGBoost-based forward return predictor.

  Pipeline:
      1. Engineer features from OHLCV (no look-ahead)
      2. Chronological train/test split
      3. TimeSeriesSplit cross-validation on training set
      4. Fit XGBoost regressor with early stopping
      5. Evaluate RMSE, MAE, directional accuracy
      6. Persist model + metadata to disk

  Example:
      >>> predictor = ReturnPredictor()
      >>> result = predictor.train(ohlcv_df)
      >>> print(result.metrics.directional_accuracy)
      >>> preds = predictor.predict(result.feature_importance)  # needs trained model
  """

  def __init__(
    self,
    config: Optional[XGBoostConfig] = None,
    model_dir: Optional[Path] = None,
  ) -> None:
    self.config = config or XGBoostConfig()
    self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    self.model_dir.mkdir(parents=True, exist_ok=True)
    self._model: Any = None
    self._feature_engineer = FeatureEngineer()
    self._feature_names: List[str] = FEATURE_COLUMNS.copy()

  @property
  def is_fitted(self) -> bool:
    return self._model is not None

  def _build_model(self) -> Any:
    """Instantiate XGBoost regressor."""
    try:
      from xgboost import XGBRegressor
    except ImportError as exc:
      raise ImportError("xgboost is required. Install with: pip install xgboost") from exc

    cfg = self.config
    return XGBRegressor(
      n_estimators=cfg.n_estimators,
      max_depth=cfg.max_depth,
      learning_rate=cfg.learning_rate,
      subsample=cfg.subsample,
      colsample_bytree=cfg.colsample_bytree,
      min_child_weight=cfg.min_child_weight,
      reg_alpha=cfg.reg_alpha,
      reg_lambda=cfg.reg_lambda,
      random_state=cfg.random_state,
      objective="reg:squarederror",
      n_jobs=-1,
    )

  @staticmethod
  def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically (no shuffling) to prevent leakage.

    Args:
        X: Feature matrix (time-ordered index)
        y: Target vector
        test_size: Fraction of most recent data held out for testing

    Returns:
        X_train, X_test, y_train, y_test
    """
    if not 0 < test_size < 1:
      raise ValueError("test_size must be between 0 and 1")
    n = len(X)
    split_idx = int(n * (1 - test_size))
    if split_idx < 10 or (n - split_idx) < 5:
      raise ValueError(f"Insufficient data for split: {n} rows")
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

  @staticmethod
  def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """
    Compute RMSE, MAE, and directional accuracy.

    Directional accuracy = fraction of times sign(predicted) == sign(actual).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    direction_correct = np.sign(y_true) == np.sign(y_pred)
    # Exclude near-zero actual returns from direction tally
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
      dir_acc = float(direction_correct[mask].mean())
    else:
      dir_acc = float(direction_correct.mean())
    return ModelMetrics(rmse=rmse, mae=mae, directional_accuracy=dir_acc, n_samples=len(y_true))

  def cross_validate(
    self,
    X_train: pd.DataFrame,
    y_train: pd.Series,
  ) -> List[ModelMetrics]:
    """
  Run time-series cross-validation on the training set.

  Uses ``TimeSeriesSplit`` to respect temporal ordering: each fold
  trains on past data and validates on future data within the train set.

    Args:
        X_train: Training features (chronological)
        y_train: Training targets

    Returns:
        List of ModelMetrics, one per CV fold
    """
    cfg = self.config
    tscv = TimeSeriesSplit(n_splits=cfg.n_cv_splits)
    fold_metrics: List[ModelMetrics] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
      X_tr = X_train.iloc[train_idx]
      y_tr = y_train.iloc[train_idx]
      X_val = X_train.iloc[val_idx]
      y_val = y_train.iloc[val_idx]

      model = self._build_model()
      model.fit(
        X_tr.values,
        y_tr.values,
        eval_set=[(X_val.values, y_val.values)],
        verbose=False,
      )
      preds = model.predict(X_val.values)
      metrics = self.compute_metrics(y_val.values, preds)
      fold_metrics.append(metrics)
      logger.info(
        "CV fold %d: RMSE=%.6f MAE=%.6f DirAcc=%.2f%%",
        fold + 1,
        metrics.rmse,
        metrics.mae,
        metrics.directional_accuracy * 100,
      )

    return fold_metrics

  def get_feature_importance(self) -> pd.DataFrame:
    """
    Extract feature importance from the fitted model.

    Returns:
        DataFrame with columns: feature, importance (gain-based)

    Raises:
        RuntimeError: If model has not been trained
    """
    if not self.is_fitted:
      raise RuntimeError("Model not trained. Call train() first.")
    importances = self._model.feature_importances_
    df = pd.DataFrame({"feature": self._feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)

  def train(
    self,
    ohlcv: pd.DataFrame,
    save: bool = True,
    model_name: Optional[str] = None,
  ) -> TrainResult:
    """
    Full training pipeline: features → split → CV → fit → evaluate → save.

    Args:
        ohlcv: OHLCV price data
        save: Persist model to disk
        model_name: Filename stem (default: timestamp)

    Returns:
        TrainResult with metrics, CV results, and feature importance
    """
    feature_df = self._feature_engineer.build_features(ohlcv)
    X, y = self._feature_engineer.split_features_target(feature_df)

    X_train, X_test, y_train, y_test = self.chronological_split(
      X, y, test_size=self.config.test_size
    )

    # Cross-validation on training portion only
    cv_metrics = self.cross_validate(X_train, y_train)

    # Final fit on full training set with early stopping on tail
    val_size = max(int(len(X_train) * 0.15), 5)
    X_fit = X_train.iloc[:-val_size]
    y_fit = y_train.iloc[:-val_size]
    X_es = X_train.iloc[-val_size:]
    y_es = y_train.iloc[-val_size:]

    self._model = self._build_model()
    self._model.fit(
      X_fit.values,
      y_fit.values,
      eval_set=[(X_es.values, y_es.values)],
      verbose=False,
    )

    # Test set evaluation (truly unseen future data)
    test_preds = self._model.predict(X_test.values)
    test_metrics = self.compute_metrics(y_test.values, test_preds)
    importance = self.get_feature_importance()

    model_path = None
    if save:
      model_path = self.save(model_name=model_name)

    return TrainResult(
      metrics=test_metrics,
      cv_metrics=cv_metrics,
      feature_importance=importance,
      model_path=model_path,
      train_size=len(X_train),
      test_size=len(X_test),
    )

  def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
    """
    Predict forward returns for the most recent rows in OHLCV data.

    Args:
        ohlcv: OHLCV bars (must have enough history for feature warm-up)

    Returns:
        Series of predicted forward log returns, indexed by date

    Raises:
        RuntimeError: If model is not fitted
    """
    if not self.is_fitted:
      raise RuntimeError("Model not trained. Call train() or load().")
    feature_df = self._feature_engineer.build_features(ohlcv)
    X, _ = self._feature_engineer.split_features_target(feature_df)
    preds = self._model.predict(X.values)
    return pd.Series(preds, index=X.index, name="predicted_return")

  def save(self, model_name: Optional[str] = None) -> Path:
    """
    Save trained model and metadata to disk.

    Saves:
        models/{name}.joblib  — serialized XGBoost model
        models/{name}_meta.json — feature names, config, timestamp

    Returns:
        Path to the saved model file
    """
    if not self.is_fitted:
      raise RuntimeError("No model to save.")

    name = model_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = self.model_dir / f"{name}.joblib"
    meta_path = self.model_dir / f"{name}_meta.json"

    joblib.dump(self._model, model_path)

    meta = {
      "feature_names": self._feature_names,
      "target": TARGET_COLUMN,
      "config": asdict(self.config),
      "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Model saved to %s", model_path)
    return model_path

  def load(self, model_path: Union[str, Path]) -> None:
    """
    Load a previously saved model from disk.

    Args:
        model_path: Path to .joblib model file
    """
    model_path = Path(model_path)
    if not model_path.exists():
      raise FileNotFoundError(f"Model not found: {model_path}")

    self._model = joblib.load(model_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    if meta_path.exists():
      meta = json.loads(meta_path.read_text())
      self._feature_names = meta.get("feature_names", FEATURE_COLUMNS)
    logger.info("Model loaded from %s", model_path)
