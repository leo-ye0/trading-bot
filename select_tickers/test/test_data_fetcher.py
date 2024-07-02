import pytest
import pandas as pd
from select_tickers_fundamental.data_fetcher import fetch_tickers, get_unique_tickers

def test_fetch_tickers():
    # Test fetching tickers from URLs
    amex_tickers = fetch_tickers("https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt")
    assert isinstance(amex_tickers, list)
    assert len(amex_tickers) > 0

@pytest.mark.parametrize("ticker", ["AAPL", "MSFT", "GOOGL"])
def test_fetch_financials(ticker):
    # Test fetching financials for a specific ticker
    financials = get_unique_tickers(ticker)
    assert financials is not None
    assert isinstance(financials, pd.DataFrame)

# Add more test cases as needed
