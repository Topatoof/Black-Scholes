"""
Feature engineering for stock return prediction.

All features are computed using only information available at or before
time *t* to avoid look-ahead bias. The prediction target is the
forward log return from *t* to *t + horizon*.

Financial rationale for each feature
------------------------------------

Daily returns
    Recent return history captures short-term momentum (trend persistence)
    and mean-reversion dynamics documented in equity anomalies literature.

Moving averages (SMA/EMA ratios)
    Price relative to its moving average indicates trend strength.
    Price above SMA → bullish trend; below → bearish. Used by technical
    traders as dynamic support/resistance levels.

RSI (Relative Strength Index)
    Measures speed and magnitude of recent price changes on a 0–100 scale.
    RSI > 70 suggests overbought (potential reversal down); RSI < 30
    suggests oversold (potential bounce). Captures mean-reversion signals.

MACD (Moving Average Convergence Divergence)
    Difference between fast and slow EMAs of price. Positive MACD signals
    upward momentum; negative signals downward. The signal line crossover
    is a classic trend-following indicator.

ATR (Average True Range)
    Measures market volatility using the full high-low-close range.
    High ATR → turbulent regime; low ATR → calm regime. Volatility
    clustering (GARCH effect) means ATR helps predict future risk.

Historical volatility
    Rolling standard deviation of log returns. Volatility tends to cluster:
    high-vol periods follow high-vol periods. Useful for sizing risk and
  as a regime indicator for return predictability.

Volume changes
    Rising volume on price moves confirms conviction; falling volume
    suggests weak trends. Volume spikes often precede breakouts or
    reversals. Log volume change normalizes scale across tickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Feature column names used by the model pipeline
FEATURE_COLUMNS: List[str] = [
  "return_1d",
  "return_5d",
  "return_21d",
  "sma_ratio_10",
  "sma_ratio_21",
  "sma_ratio_50",
  "ema_ratio_12",
  "rsi_14",
  "macd",
  "macd_signal",
  "macd_hist",
  "atr_14",
  "atr_ratio",
  "hist_vol_21",
  "volume_change_1d",
  "volume_change_5d",
  "volume_sma_ratio",
]

TARGET_COLUMN = "target_forward_return"


@dataclass(frozen=True)
class FeatureConfig:
  """Hyperparameters for feature engineering."""

  rsi_window: int = 14
  atr_window: int = 14
  macd_fast: int = 12
  macd_slow: int = 26
  macd_signal: int = 9
  vol_window: int = 21
  forward_horizon: int = 1  # days ahead to predict
  min_history: int = 60  # minimum rows before first valid sample


class FeatureEngineer:
  """
  Build leak-free feature matrix from OHLCV price data.

  Every feature at row *t* uses data from t and earlier only.
  The target is the log return from t to t + horizon (unknown at t,
  used only for supervised training).
  """

  def __init__(self, config: Optional[FeatureConfig] = None) -> None:
    self.config = config or FeatureConfig()

  @staticmethod
  def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV input DataFrame.

    Args:
        df: DataFrame with open, high, low, close, volume columns

    Returns:
        Cleaned, sorted DataFrame

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower() if hasattr(df.columns, "str") else df.columns)
    # Normalize column names to lowercase
    frame = df.copy()
    frame.columns = [c.lower() for c in frame.columns]
    missing = required - set(frame.columns)
    if missing:
      raise ValueError(f"OHLCV data missing columns: {sorted(missing)}")
    if frame.empty:
      raise ValueError("OHLCV data is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
      frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    for col in required:
      frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["close"])
    if (frame["close"] <= 0).any():
      raise ValueError("Close prices must be positive")
    return frame

  def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features and forward return target from OHLCV bars.

    Args:
        ohlcv: OHLCV DataFrame indexed by date

    Returns:
        DataFrame with feature columns and ``target_forward_return``.
        Rows with insufficient history or missing target are dropped.

    Note:
        No feature uses future data. Target uses shift(-horizon) on
        close prices, which is correct for supervised learning labels.
    """
    frame = self.validate_ohlcv(ohlcv)
    cfg = self.config
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    # --- Daily returns (log) ---
    log_ret = np.log(close / close.shift(1))
    frame["return_1d"] = log_ret
    frame["return_5d"] = np.log(close / close.shift(5))
    frame["return_21d"] = np.log(close / close.shift(21))

    # --- Moving average ratios (price / SMA - 1) ---
    for window, col in [(10, "sma_ratio_10"), (21, "sma_ratio_21"), (50, "sma_ratio_50")]:
      sma = close.rolling(window, min_periods=window).mean()
      frame[col] = close / sma - 1.0

    # --- EMA ratio ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    frame["ema_ratio_12"] = close / ema_12 - 1.0

    # --- RSI ---
    frame["rsi_14"] = self._rsi(close, cfg.rsi_window)

    # --- MACD ---
    macd_line, signal_line, hist = self._macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    frame["macd"] = macd_line / close  # scale by price for cross-ticker stability
    frame["macd_signal"] = signal_line / close
    frame["macd_hist"] = hist / close

    # --- ATR ---
    atr = self._atr(high, low, close, cfg.atr_window)
    frame["atr_14"] = atr / close  # normalized ATR
    frame["atr_ratio"] = atr / atr.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean() - 1.0

    # --- Historical volatility (rolling std of log returns, annualized) ---
    daily_vol = log_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)
    frame["hist_vol_21"] = daily_vol * np.sqrt(252)

    # --- Volume changes ---
    log_vol = np.log(volume.replace(0, np.nan))
    frame["volume_change_1d"] = log_vol - log_vol.shift(1)
    frame["volume_change_5d"] = log_vol - log_vol.shift(5)
    vol_sma = volume.rolling(21, min_periods=21).mean()
    frame["volume_sma_ratio"] = volume / vol_sma - 1.0

    # --- Target: forward log return (label, not a feature) ---
    frame[TARGET_COLUMN] = np.log(close.shift(-cfg.forward_horizon) / close)

    # Drop warm-up rows and rows without target
    feature_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    result = frame[feature_cols].iloc[cfg.min_history :].copy()
    result = result.dropna()
    return result

  @staticmethod
  def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

  @staticmethod
  def _macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

  @staticmethod
  def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
      [
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
      ],
      axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

  def split_features_target(
    self,
    feature_df: pd.DataFrame,
  ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y.

    Args:
        feature_df: Output from ``build_features``

    Returns:
        (X, y) tuple ready for model training
    """
    if TARGET_COLUMN not in feature_df.columns:
      raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
    X = feature_df[FEATURE_COLUMNS].copy()
    y = feature_df[TARGET_COLUMN].copy()
    return X, y
