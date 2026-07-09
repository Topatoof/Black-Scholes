"""
Alpaca market data integration for the Black-Scholes platform.

Provides authenticated access to historical stock prices, live quotes,
and options chains via the Alpaca Market Data API. Output is normalized
into pandas DataFrames that feed directly into ``OptionParams`` and the
pricing engine.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.black_scholes import OptionParams, OptionType
from src.volatility import VolatilityCalculator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MarketDataError(Exception):
  """Base exception for market data operations."""


class AuthenticationError(MarketDataError):
  """Raised when Alpaca API credentials are missing or invalid."""


class DataValidationError(MarketDataError):
  """Raised when API response data fails validation."""


class APIRequestError(MarketDataError):
  """Raised when an Alpaca API request fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")

HISTORICAL_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OPTIONS_CHAIN_COLUMNS = [
  "symbol",
  "underlying_symbol",
  "strike",
  "expiration",
  "option_type",
  "bid",
  "ask",
  "last",
  "mid",
  "volume",
  "open_interest",
  "implied_volatility",
  "delta",
  "gamma",
  "vega",
  "theta",
  "rho",
]


@dataclass(frozen=True)
class AlpacaCredentials:
  """
  Alpaca API credentials.

  Attributes:
      api_key: Alpaca API key ID
      secret_key: Alpaca API secret key
      paper: Use paper trading endpoints when applicable
  """

  api_key: str
  secret_key: str
  paper: bool = True

  def validate(self) -> None:
    """Ensure credentials are non-empty."""
    if not self.api_key or not self.api_key.strip():
      raise AuthenticationError("Alpaca API key is missing. Set ALPACA_API_KEY.")
    if not self.secret_key or not self.secret_key.strip():
      raise AuthenticationError("Alpaca secret key is missing. Set ALPACA_SECRET_KEY.")

  @classmethod
  def from_env(cls, paper: Optional[bool] = None) -> "AlpacaCredentials":
    """
    Load credentials from environment variables.

    Environment variables:
        ALPACA_API_KEY: API key ID
        ALPACA_SECRET_KEY: API secret key
        ALPACA_PAPER: 'true'/'false' (default true)

    Automatically loads a ``.env`` file when python-dotenv is installed.

    Returns:
        AlpacaCredentials instance

    Raises:
        AuthenticationError: If required variables are not set
    """
    try:
      from dotenv import load_dotenv

      load_dotenv()
    except ImportError:
      pass
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if paper is None:
      paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes")
    creds = cls(api_key=api_key, secret_key=secret_key, paper=paper)
    creds.validate()
    return creds


# ---------------------------------------------------------------------------
# Market data client
# ---------------------------------------------------------------------------


class AlpacaMarketData:
  """
  Alpaca Market Data API client for stocks and options.

  Wraps ``alpaca-py`` with validation, error handling, and pandas output
  formatted for the Black-Scholes analytics engine.

  Example:
      >>> client = AlpacaMarketData.from_env()
      >>> prices = client.get_historical_prices("AAPL", start="2024-01-01")
      >>> spot = client.get_current_price("AAPL")
      >>> chain = client.get_options_chain("AAPL")
      >>> params = client.to_option_params(chain.iloc[0], spot=spot)
  """

  def __init__(self, credentials: AlpacaCredentials) -> None:
    """
    Initialize the market data client.

    Args:
        credentials: Validated Alpaca API credentials
    """
    credentials.validate()
    self.credentials = credentials
    self._stock_client = None
    self._option_client = None

  @classmethod
  def from_env(cls, paper: Optional[bool] = None) -> "AlpacaMarketData":
    """Create client using ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` env vars."""
    return cls(AlpacaCredentials.from_env(paper=paper))

  def _import_alpaca(self) -> Any:
    """Import alpaca-py with a helpful error if not installed."""
    try:
      import alpaca  # noqa: F401
    except ImportError as exc:
      raise MarketDataError(
        "alpaca-py is required for market data. Install with: pip install alpaca-py"
      ) from exc

  @property
  def stock_client(self) -> Any:
    """Lazy-initialize the Alpaca stock historical data client."""
    if self._stock_client is None:
      self._import_alpaca()
      from alpaca.data.historical import StockHistoricalDataClient

      self._stock_client = StockHistoricalDataClient(
        api_key=self.credentials.api_key,
        secret_key=self.credentials.secret_key,
      )
    return self._stock_client

  @property
  def option_client(self) -> Any:
    """Lazy-initialize the Alpaca option historical data client."""
    if self._option_client is None:
      self._import_alpaca()
      from alpaca.data.historical import OptionHistoricalDataClient

      self._option_client = OptionHistoricalDataClient(
        api_key=self.credentials.api_key,
        secret_key=self.credentials.secret_key,
      )
    return self._option_client

  # -------------------------------------------------------------------------
  # Validation helpers
  # -------------------------------------------------------------------------

  @staticmethod
  def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize a ticker symbol.

    Args:
        symbol: Raw ticker string (e.g. 'aapl')

    Returns:
        Uppercase normalized symbol

    Raises:
        DataValidationError: If symbol format is invalid
    """
    if not symbol or not str(symbol).strip():
      raise DataValidationError("Symbol must be a non-empty string")
    normalized = str(symbol).strip().upper()
    if not SYMBOL_PATTERN.match(normalized):
      raise DataValidationError(f"Invalid symbol format: {symbol!r}")
    return normalized

  @staticmethod
  def validate_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a historical OHLCV DataFrame.

    Args:
        df: Raw price DataFrame

    Returns:
        Cleaned DataFrame with standard columns and DatetimeIndex

    Raises:
        DataValidationError: If required columns are missing or data is empty
    """
    if df is None or df.empty:
      raise DataValidationError("Historical price data is empty")

    frame = df.copy()

    # Flatten multi-index from alpaca-py if present
    if isinstance(frame.index, pd.MultiIndex):
      frame = frame.reset_index()

    if "timestamp" in frame.columns:
      frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
      frame = frame.set_index("timestamp")
    elif not isinstance(frame.index, pd.DatetimeIndex):
      frame.index = pd.to_datetime(frame.index, utc=True)

    rename_map = {
      "open": "open",
      "high": "high",
      "low": "low",
      "close": "close",
      "volume": "volume",
    }
    missing = [c for c in rename_map if c not in frame.columns]
    if missing:
      raise DataValidationError(f"Missing OHLCV columns: {missing}")

    frame = frame[list(rename_map.keys())].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = frame.sort_index()

    if frame.empty:
      raise DataValidationError("No valid OHLCV rows after cleaning")

    if (frame["close"] <= 0).any():
      raise DataValidationError("Historical prices must be positive")

    return frame

  @staticmethod
  def validate_options_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize an options chain DataFrame.

    Args:
        df: Raw options chain DataFrame

    Returns:
        Cleaned chain with standard columns

    Raises:
        DataValidationError: If chain is empty or missing required fields
    """
    if df is None or df.empty:
      raise DataValidationError("Options chain is empty")

    frame = df.copy()
    required = {"symbol", "strike", "expiration", "option_type"}
    missing = required - set(frame.columns)
    if missing:
      raise DataValidationError(f"Options chain missing columns: {sorted(missing)}")

    frame["symbol"] = frame["symbol"].astype(str)
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["expiration"] = pd.to_datetime(frame["expiration"], utc=True)
    frame["option_type"] = frame["option_type"].str.lower()
    frame = frame.dropna(subset=["symbol", "strike", "expiration"])

    if frame.empty:
      raise DataValidationError("No valid option contracts after cleaning")

    return frame.reset_index(drop=True)

  # -------------------------------------------------------------------------
  # Stock data
  # -------------------------------------------------------------------------

  def get_historical_prices(
    self,
    symbol: str,
    start: Union[str, date, datetime, pd.Timestamp],
    end: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
    timeframe: str = "1Day",
    feed: str = "iex",
  ) -> pd.DataFrame:
    """
    Download historical OHLCV bars for a stock.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL')
        start: Start date (inclusive)
        end: End date (exclusive); defaults to now
        timeframe: Bar size ('1Day', '1Hour', '1Min', etc.)
        feed: Data feed ('iex' for free tier, 'sip' for paid)

    Returns:
        DataFrame indexed by timestamp with columns:
        open, high, low, close, volume

    Raises:
        DataValidationError: Invalid symbol or empty response
        APIRequestError: Alpaca API failure
    """
    symbol = self.validate_symbol(symbol)
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
      start_ts = start_ts.tz_localize("UTC")
    else:
      start_ts = start_ts.tz_convert("UTC")
    if end is not None:
      end_ts = pd.Timestamp(end)
      if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
      else:
        end_ts = end_ts.tz_convert("UTC")
    else:
      end_ts = pd.Timestamp.now(tz="UTC")

    if start_ts >= end_ts:
      raise DataValidationError("start must be before end")

    try:
      from alpaca.data.enums import DataFeed
      from alpaca.data.requests import StockBarsRequest
      from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

      tf = self._parse_timeframe(timeframe)
      feed_enum = DataFeed.SIP if feed.lower() == "sip" else DataFeed.IEX

      request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        feed=feed_enum,
      )
      bars = self.stock_client.get_stock_bars(request)
      raw_df = bars.df if hasattr(bars, "df") else pd.DataFrame()

    except DataValidationError:
      raise
    except Exception as exc:
      raise APIRequestError(f"Failed to fetch historical prices for {symbol}: {exc}") from exc

    if raw_df is None or raw_df.empty:
      raise DataValidationError(f"No historical data returned for {symbol}")

    return self.validate_ohlcv_frame(raw_df)

  def get_current_price(self, symbol: str, use_mid: bool = True) -> float:
    """
    Retrieve the latest stock price.

    Uses the latest quote mid-price by default (bid+ask)/2, falling back
    to the latest trade price if quotes are unavailable.

    Args:
        symbol: Ticker symbol
        use_mid: Prefer quote mid over last trade

    Returns:
        Latest price as float

    Raises:
        DataValidationError: Invalid or non-positive price
        APIRequestError: Alpaca API failure
    """
    symbol = self.validate_symbol(symbol)

    try:
      from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

      price: Optional[float] = None

      if use_mid:
        quotes = self.stock_client.get_stock_latest_quote(
          StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )
        quote = quotes.get(symbol) if isinstance(quotes, dict) else None
        if quote is not None:
          bid = float(getattr(quote, "bid_price", 0) or 0)
          ask = float(getattr(quote, "ask_price", 0) or 0)
          if bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
          elif ask > 0:
            price = ask
          elif bid > 0:
            price = bid

      if price is None:
        trades = self.stock_client.get_stock_latest_trade(
          StockLatestTradeRequest(symbol_or_symbols=symbol)
        )
        trade = trades.get(symbol) if isinstance(trades, dict) else None
        if trade is not None:
          price = float(getattr(trade, "price", 0) or 0)

    except DataValidationError:
      raise
    except Exception as exc:
      raise APIRequestError(f"Failed to fetch current price for {symbol}: {exc}") from exc

    if price is None or price <= 0:
      raise DataValidationError(f"No valid current price for {symbol}")

    return price

  # -------------------------------------------------------------------------
  # Options data
  # -------------------------------------------------------------------------

  def get_options_chain(
    self,
    underlying_symbol: str,
    expiration_date_gte: Optional[Union[str, date]] = None,
    expiration_date_lte: Optional[Union[str, date]] = None,
    strike_price_gte: Optional[float] = None,
    strike_price_lte: Optional[float] = None,
    option_type: Optional[str] = None,
    feed: str = "indicative",
  ) -> pd.DataFrame:
    """
    Retrieve the options chain for an underlying symbol.

    Args:
        underlying_symbol: Underlying ticker (e.g. 'AAPL')
        expiration_date_gte: Minimum expiration date filter
        expiration_date_lte: Maximum expiration date filter
        strike_price_gte: Minimum strike filter
        strike_price_lte: Maximum strike filter
        option_type: 'call' or 'put' filter
        feed: Options feed ('indicative' or 'opra')

    Returns:
        DataFrame with one row per contract including greeks and IV

    Raises:
        DataValidationError: Empty chain or invalid data
        APIRequestError: Alpaca API failure
    """
    underlying_symbol = self.validate_symbol(underlying_symbol)

    try:
      from alpaca.data.enums import OptionsFeed
      from alpaca.data.requests import OptionChainRequest

      feed_enum = OptionsFeed.OPRA if feed.lower() == "opra" else OptionsFeed.INDICATIVE

      request_kwargs: Dict[str, Any] = {
        "underlying_symbol": underlying_symbol,
        "feed": feed_enum,
      }
      if expiration_date_gte is not None:
        request_kwargs["expiration_date_gte"] = str(expiration_date_gte)
      if expiration_date_lte is not None:
        request_kwargs["expiration_date_lte"] = str(expiration_date_lte)
      if strike_price_gte is not None:
        request_kwargs["strike_price_gte"] = float(strike_price_gte)
      if strike_price_lte is not None:
        request_kwargs["strike_price_lte"] = float(strike_price_lte)
      if option_type is not None:
        from alpaca.trading.enums import ContractType

        request_kwargs["type"] = (
          ContractType.CALL if option_type.lower() == "call" else ContractType.PUT
        )

      chain_data = self.option_client.get_option_chain(OptionChainRequest(**request_kwargs))
      frame = self._chain_to_dataframe(chain_data, underlying_symbol)

    except DataValidationError:
      raise
    except Exception as exc:
      raise APIRequestError(
        f"Failed to fetch options chain for {underlying_symbol}: {exc}"
      ) from exc

    return self.validate_options_chain(frame)

  def _chain_to_dataframe(self, chain_data: Any, underlying_symbol: str) -> pd.DataFrame:
    """Convert Alpaca option chain response to a normalized DataFrame."""
    if not chain_data:
      return pd.DataFrame(columns=OPTIONS_CHAIN_COLUMNS)

    rows: List[Dict[str, Any]] = []
    items = chain_data.items() if isinstance(chain_data, dict) else []

    for contract_symbol, snapshot in items:
      row = self._snapshot_to_row(contract_symbol, snapshot, underlying_symbol)
      if row is not None:
        rows.append(row)

    return pd.DataFrame(rows, columns=OPTIONS_CHAIN_COLUMNS)

  @staticmethod
  def _snapshot_to_row(
    contract_symbol: str,
    snapshot: Any,
    underlying_symbol: str,
  ) -> Optional[Dict[str, Any]]:
    """Map a single OptionsSnapshot to a flat dictionary."""
    strike, expiration, opt_type = AlpacaMarketData._parse_occ_symbol(contract_symbol)

    quote = getattr(snapshot, "latest_quote", None)
    trade = getattr(snapshot, "latest_trade", None)
    greeks = getattr(snapshot, "greeks", None)

    bid = float(getattr(quote, "bid_price", 0) or 0) if quote else 0.0
    ask = float(getattr(quote, "ask_price", 0) or 0) if quote else 0.0
    last = float(getattr(trade, "price", 0) or 0) if trade else 0.0
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last

    iv = getattr(snapshot, "implied_volatility", None)
    if iv is not None:
      iv = float(iv)

    return {
      "symbol": contract_symbol,
      "underlying_symbol": underlying_symbol,
      "strike": strike,
      "expiration": expiration,
      "option_type": opt_type,
      "bid": bid,
      "ask": ask,
      "last": last,
      "mid": mid,
      "volume": float(getattr(trade, "size", 0) or 0) if trade else 0.0,
      "open_interest": None,
      "implied_volatility": iv,
      "delta": float(getattr(greeks, "delta", 0) or 0) if greeks else None,
      "gamma": float(getattr(greeks, "gamma", 0) or 0) if greeks else None,
      "vega": float(getattr(greeks, "vega", 0) or 0) if greeks else None,
      "theta": float(getattr(greeks, "theta", 0) or 0) if greeks else None,
      "rho": float(getattr(greeks, "rho", 0) or 0) if greeks else None,
    }

  @staticmethod
  def _parse_occ_symbol(symbol: str) -> tuple[Optional[float], Optional[pd.Timestamp], str]:
    """
    Parse OCC option symbol into strike, expiration, and type.

    Format: ROOT + YYMMDD + C/P + strike*1000 (8 digits)
    Example: AAPL250117C00150000
    """
    match = re.match(r"^[A-Z]+(\d{6})([CP])(\d{8})$", symbol)
    if not match:
      return None, None, "call"

    date_part, cp_flag, strike_part = match.groups()
    expiration = pd.Timestamp(f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:6]}", tz="UTC")
    strike = int(strike_part) / 1000.0
    opt_type = "call" if cp_flag == "C" else "put"
    return strike, expiration, opt_type

  @staticmethod
  def _parse_timeframe(timeframe: str) -> Any:
    """Map string timeframe to alpaca TimeFrame object."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mapping = {
      "1min": TimeFrame(1, TimeFrameUnit.Minute),
      "1hour": TimeFrame(1, TimeFrameUnit.Hour),
      "1day": TimeFrame(1, TimeFrameUnit.Day),
    }
    key = timeframe.replace(" ", "").lower()
    if key not in mapping:
      raise DataValidationError(
        f"Unsupported timeframe {timeframe!r}. Use one of: {list(mapping)}"
      )
    return mapping[key]

  # -------------------------------------------------------------------------
  # Black-Scholes bridge
  # -------------------------------------------------------------------------

  def get_close_series(self, historical: pd.DataFrame) -> pd.Series:
    """
    Extract a validated close-price series from historical OHLCV data.

    Args:
        historical: Output from ``get_historical_prices``

    Returns:
        Close price series for volatility calculations
    """
    frame = self.validate_ohlcv_frame(historical)
    return VolatilityCalculator.validate_price_data(frame["close"])

  def compute_historical_volatility(
    self,
    historical: pd.DataFrame,
    trading_days_per_year: int = 252,
  ) -> float:
    """
    Compute annualized historical volatility from OHLCV bars.

    Args:
        historical: Output from ``get_historical_prices``
        trading_days_per_year: Annualization factor

    Returns:
        Annualized volatility (sigma) for Black-Scholes input
    """
    closes = self.get_close_series(historical)
    stats = VolatilityCalculator.historical_volatility(closes, trading_days_per_year)
    return stats.annualized_volatility

  @staticmethod
  def years_to_expiry(expiration: Union[str, date, datetime, pd.Timestamp]) -> float:
    """Convert an expiration timestamp to time-to-expiry in years."""
    expiry = pd.Timestamp(expiration)
    if expiry.tzinfo is None:
      expiry = expiry.tz_localize("UTC")
    else:
      expiry = expiry.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    delta_days = (expiry - now).total_seconds() / 86400.0
    return max(delta_days / 365.0, 1.0 / 365.0)

  def to_option_params(
    self,
    contract_row: Union[pd.Series, Dict[str, Any]],
    spot: float,
    risk_free_rate: float,
    volatility: Optional[float] = None,
    dividend_yield: float = 0.0,
  ) -> OptionParams:
    """
    Build ``OptionParams`` from an options chain row for Black-Scholes pricing.

    Args:
        contract_row: Row from ``get_options_chain`` DataFrame
        spot: Current underlying price
        risk_free_rate: Continuously compounded risk-free rate
        volatility: Override volatility; uses chain IV if None
        dividend_yield: Continuous dividend yield

    Returns:
        Validated OptionParams ready for BlackScholesModel.price()

    Raises:
        DataValidationError: Missing required fields
    """
    row = contract_row if isinstance(contract_row, dict) else contract_row.to_dict()

    strike = float(row.get("strike", 0))
    expiration = row.get("expiration")
    if expiration is None:
      raise DataValidationError("Contract row missing expiration")

    sigma = volatility
    if sigma is None:
      iv = row.get("implied_volatility")
      if iv is None or (isinstance(iv, float) and pd.isna(iv)):
        raise DataValidationError(
          "Volatility not provided and contract has no implied_volatility"
        )
      sigma = float(iv)

    params = OptionParams(
      spot=float(spot),
      strike=strike,
      time_to_expiry=self.years_to_expiry(expiration),
      risk_free_rate=float(risk_free_rate),
      volatility=float(sigma),
      dividend_yield=float(dividend_yield),
    )
    params.validate()
    return params

  def prepare_black_scholes_inputs(
    self,
    underlying_symbol: str,
    contract_row: Union[pd.Series, Dict[str, Any]],
    risk_free_rate: float,
    volatility: Optional[float] = None,
    dividend_yield: float = 0.0,
    historical_lookback_days: int = 252,
  ) -> Dict[str, Any]:
    """
    Fetch live data and produce a complete Black-Scholes input package.

    Combines current spot, optional historical vol, and contract details
    into a dictionary consumed by the pricing engine and UI.

    Args:
        underlying_symbol: Stock ticker
        contract_row: Selected option contract from the chain
        risk_free_rate: Risk-free rate for pricing
        volatility: Explicit vol; if None uses historical then chain IV
        dividend_yield: Dividend yield assumption
        historical_lookback_days: Days of history for realized vol fallback

    Returns:
        Dictionary with keys:
            option_params, option_type, market_price, symbol,
            historical_volatility, spot
    """
    symbol = self.validate_symbol(underlying_symbol)
    spot = self.get_current_price(symbol)

    hist_vol: Optional[float] = None
    try:
      start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=historical_lookback_days)
      historical = self.get_historical_prices(symbol, start=start)
      hist_vol = self.compute_historical_volatility(historical)
    except MarketDataError as exc:
      logger.warning("Could not compute historical volatility for %s: %s", symbol, exc)

    sigma = volatility or hist_vol
    params = self.to_option_params(
      contract_row,
      spot=spot,
      risk_free_rate=risk_free_rate,
      volatility=sigma,
      dividend_yield=dividend_yield,
    )

    row = contract_row if isinstance(contract_row, dict) else contract_row.to_dict()
    opt_type_str = str(row.get("option_type", "call")).lower()
    option_type = OptionType.CALL if opt_type_str == "call" else OptionType.PUT
    market_price = float(row.get("mid") or row.get("last") or 0)

    return {
      "symbol": symbol,
      "spot": spot,
      "option_params": params,
      "option_type": option_type,
      "market_price": market_price,
      "historical_volatility": hist_vol,
      "contract_symbol": row.get("symbol"),
    }
