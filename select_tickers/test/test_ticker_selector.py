import pytest
from select_tickers_fundamental.ticker_selector import select_tickers

def test_select_tickers_with_lowest_growth():
    # Test selecting tickers based on growth criteria
    selected_tickers = select_tickers()
    assert isinstance(selected_tickers, list)
    assert len(selected_tickers) == 30