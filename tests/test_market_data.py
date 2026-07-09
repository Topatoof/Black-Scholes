"""Unit tests for Alpaca market data layer."""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from data.market_data import (
  AlpacaCredentials,
  AlpacaMarketData,
  AuthenticationError,
  DataValidationError,
)
from src.black_scholes import OptionType


@pytest.fixture
def credentials():
  return AlpacaCredentials(api_key="test_key", secret_key="test_secret")


@pytest.fixture
def client(credentials):
  return AlpacaMarketData(credentials)


class TestAlpacaCredentials:
  def test_from_env_missing_raises(self, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    try:
      from dotenv import load_dotenv
      monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    except ImportError:
      pass
    with pytest.raises(AuthenticationError):
      AlpacaCredentials.from_env()

  def test_from_env_success(self, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key123")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret456")
    creds = AlpacaCredentials.from_env()
    assert creds.api_key == "key123"
    assert creds.secret_key == "secret456"


class TestValidation:
  def test_validate_symbol(self, client):
    assert client.validate_symbol("aapl") == "AAPL"

  def test_validate_symbol_empty_raises(self, client):
    with pytest.raises(DataValidationError):
      client.validate_symbol("")

  def test_validate_ohlcv_frame(self, client):
    df = pd.DataFrame(
      {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]},
      index=pd.DatetimeIndex([datetime(2024, 1, 1, tzinfo=timezone.utc)]),
    )
    result = client.validate_ohlcv_frame(df)
    assert len(result) == 1

  def test_validate_ohlcv_empty_raises(self, client):
    with pytest.raises(DataValidationError):
      client.validate_ohlcv_frame(pd.DataFrame())

  def test_validate_options_chain(self, client):
    df = pd.DataFrame(
      [{
        "symbol": "AAPL250117C00150000",
        "underlying_symbol": "AAPL",
        "strike": 150.0,
        "expiration": "2025-01-17",
        "option_type": "call",
        "bid": 1.0,
        "ask": 1.2,
        "last": 1.1,
        "mid": 1.1,
      }]
    )
    result = client.validate_options_chain(df)
    assert len(result) == 1

  def test_parse_occ_symbol(self):
    strike, expiry, opt_type = AlpacaMarketData._parse_occ_symbol("AAPL250117C00150000")
    assert strike == 150.0
    assert opt_type == "call"
    assert expiry.year == 2025


class TestBlackScholesBridge:
  def test_years_to_expiry(self):
    future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=180)
    t = AlpacaMarketData.years_to_expiry(future)
    assert 0.4 < t < 0.6

  def test_to_option_params(self, client):
    row = {
      "symbol": "AAPL250117C00150000",
      "strike": 150.0,
      "expiration": pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=90),
      "option_type": "call",
      "implied_volatility": 0.25,
      "mid": 5.0,
    }
    params = client.to_option_params(row, spot=155.0, risk_free_rate=0.05)
    assert params.spot == 155.0
    assert params.strike == 150.0
    assert params.volatility == 0.25

  def test_to_option_params_missing_vol_raises(self, client):
    row = {
      "strike": 150.0,
      "expiration": pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=90),
      "option_type": "call",
    }
    with pytest.raises(DataValidationError):
      client.to_option_params(row, spot=155.0, risk_free_rate=0.05)

  @patch.object(AlpacaMarketData, "get_current_price", return_value=155.0)
  @patch.object(AlpacaMarketData, "get_historical_prices")
  def test_prepare_black_scholes_inputs(self, mock_hist, mock_price, client):
    dates = pd.date_range("2024-01-01", periods=30, freq="B", tz="UTC")
    prices = pd.Series(range(150, 180), index=dates, dtype=float)
    mock_hist.return_value = pd.DataFrame(
      {
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": 1_000_000,
      },
      index=dates,
    )
    row = {
      "symbol": "AAPL250117C00150000",
      "strike": 150.0,
      "expiration": pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=90),
      "option_type": "call",
      "implied_volatility": 0.25,
      "mid": 5.0,
    }
    result = client.prepare_black_scholes_inputs("AAPL", row, risk_free_rate=0.05, volatility=0.25)
    assert result["spot"] == 155.0
    assert result["option_type"] == OptionType.CALL
    assert result["option_params"].strike == 150.0


class TestHistoricalVolatility:
  def test_compute_historical_volatility(self, client):
    dates = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    prices = 100 * (1 + pd.Series(range(60), index=dates) * 0.001)
    df = pd.DataFrame(
      {
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": 1_000_000,
      },
      index=dates,
    )
    vol = client.compute_historical_volatility(df)
    assert vol > 0
