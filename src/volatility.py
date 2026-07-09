"""
Historical and implied volatility analytics.

Computes log returns, rolling volatility, and annualized volatility
from price time series data with validation and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolatilityStats:
  """Summary statistics from historical volatility analysis."""

  daily_volatility: float
  annualized_volatility: float
  mean_log_return: float
  num_observations: int
  trading_days_per_year: int


class VolatilityCalculator:
  """
  Historical volatility calculator from price data.

  Log returns are computed as:
      r_t = ln(P_t / P_{t-1})

  Historical volatility (sample std of log returns):
      sigma_daily = std(r_t)
      sigma_annual = sigma_daily * sqrt(trading_days_per_year)

  Annualization assumes 252 trading days per year by default.
  """

  TRADING_DAYS_PER_YEAR = 252

  @staticmethod
  def validate_price_data(prices: pd.Series) -> pd.Series:
    """
    Validate and clean price data for volatility calculation.

    Checks:
        - Non-empty series
        - Numeric values
        - All prices positive
        - No excessive missing data

    Args:
        prices: Series of asset prices indexed by date

    Returns:
        Cleaned price series sorted by index

    Raises:
        ValueError: If data fails validation
    """
    if prices is None or len(prices) == 0:
      raise ValueError("Price data is empty")

    if not pd.api.types.is_numeric_dtype(prices):
      raise ValueError("Price data must be numeric")

    prices = prices.dropna()
    if len(prices) < 2:
      raise ValueError("Need at least 2 price observations")

    if (prices <= 0).any():
      raise ValueError("All prices must be positive")

    missing_pct = prices.isna().mean()
    if missing_pct > 0.1:
      raise ValueError(f"Too many missing values: {missing_pct:.1%}")

    return prices.sort_index()

  @staticmethod
  def load_csv(
    file_path: str,
    date_column: str = "Date",
    price_column: str = "Close",
    date_format: Optional[str] = None,
  ) -> pd.Series:
    """
    Load price data from CSV file.

    Args:
        file_path: Path to CSV file
        date_column: Column name for dates
        price_column: Column name for prices
        date_format: Optional date format string

    Returns:
        Price series indexed by date
    """
    try:
      df = pd.read_csv(file_path)
    except FileNotFoundError:
      raise FileNotFoundError(f"CSV file not found: {file_path}")
    except pd.errors.EmptyDataError:
      raise ValueError("CSV file is empty")

    if date_column not in df.columns:
      raise ValueError(f"Date column '{date_column}' not found. Available: {list(df.columns)}")
    if price_column not in df.columns:
      raise ValueError(f"Price column '{price_column}' not found. Available: {list(df.columns)}")

    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    df = df.dropna(subset=[date_column, price_column])
    df = df.set_index(date_column)

    return VolatilityCalculator.validate_price_data(df[price_column])

  @staticmethod
  def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from price series.

    r_t = ln(P_t / P_{t-1})
    """
    prices = VolatilityCalculator.validate_price_data(prices)
    return np.log(prices / prices.shift(1)).dropna()

  @staticmethod
  def historical_volatility(
    prices: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
  ) -> VolatilityStats:
    """
    Calculate historical (realized) volatility from prices.

    Args:
        prices: Price time series
        trading_days_per_year: Days for annualization (default 252)

    Returns:
        VolatilityStats with daily and annualized vol
    """
    returns = VolatilityCalculator.log_returns(prices)
    daily_vol = float(returns.std(ddof=1))
    annual_vol = daily_vol * np.sqrt(trading_days_per_year)

    return VolatilityStats(
      daily_volatility=daily_vol,
      annualized_volatility=annual_vol,
      mean_log_return=float(returns.mean()),
      num_observations=len(returns),
      trading_days_per_year=trading_days_per_year,
    )

  @staticmethod
  def rolling_volatility(
    prices: pd.Series,
    window: int = 21,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
  ) -> pd.Series:
    """
    Compute rolling annualized volatility.

    Args:
        prices: Price time series
        window: Rolling window size in trading days
        trading_days_per_year: Annualization factor

    Returns:
        Series of rolling annualized volatility
    """
    if window < 2:
      raise ValueError("Rolling window must be at least 2")

    returns = VolatilityCalculator.log_returns(prices)
    rolling_daily = returns.rolling(window=window).std(ddof=1)
    return rolling_daily * np.sqrt(trading_days_per_year)

  @staticmethod
  def volatility_surface_grid(
    spot: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    volatilities: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a volatility surface grid for 3D visualization.

    Args:
        spot: Current spot price (for moneyness reference)
        strikes: 1D array of strike prices
        expiries: 1D array of expiries in years
        volatilities: 2D array of shape (len(expiries), len(strikes))

    Returns:
        Tuple of (strike_mesh, expiry_mesh, vol_surface) for plotting
    """
    strike_mesh, expiry_mesh = np.meshgrid(strikes, expiries)
    if volatilities.shape != strike_mesh.shape:
      raise ValueError(
        f"Volatility grid shape {volatilities.shape} does not match "
        f"expected ({len(expiries)}, {len(strikes)})"
      )
    return strike_mesh, expiry_mesh, volatilities

  @staticmethod
  def generate_synthetic_smile(
    spot: float,
    atm_vol: float,
    skew: float = -0.1,
    smile: float = 0.05,
  ) -> callable:
    """
    Generate a synthetic volatility smile function for demonstration.

    sigma(K) = atm_vol + skew * (K/S - 1) + smile * (K/S - 1)^2

    Args:
        spot: Current spot price
        atm_vol: At-the-money volatility
        skew: Linear skew coefficient
        smile: Quadratic smile coefficient

    Returns:
        Callable mapping strike -> implied volatility
    """

    def smile_fn(strike: float) -> float:
      moneyness = strike / spot - 1.0
      return max(atm_vol + skew * moneyness + smile * moneyness**2, 0.01)

    return smile_fn
