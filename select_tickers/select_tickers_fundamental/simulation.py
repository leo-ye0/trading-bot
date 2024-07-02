from datetime import datetime, timedelta
import yfinance as yf
from ticker_selector import select_tickers
from data_fetcher import get_unique_tickers
import pandas as pd

def fetch_historical_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval='1d')
        return data
    except Exception as e:
        print(f"Failed to fetch historical data for {ticker}: {e}")
        return None

def simulate_buying(tickers, start_date, investment_per_ticker):
    portfolio = {}
    for ticker in tickers:
        data = fetch_historical_data(ticker, start_date, start_date + timedelta(days=1))
        if data is not None and not data.empty:
            close_price = data.iloc[-1]['Close']
            num_shares = investment_per_ticker // close_price
            portfolio[ticker] = {'num_shares': num_shares, 'purchase_price': close_price}
            print(f"Bought {num_shares} shares of {ticker} at {close_price}")
        else:
            print(f"No data available for {ticker} on {start_date}")
    return portfolio

def calculate_end_date(start_date):
    # Calculate end date as start date + 63 trading days
    end_date = start_date + timedelta(days=63)
    return end_date

def simulate_selling(portfolio, end_date):
    portfolio_value = 0
    for ticker, info in portfolio.items():
        data = fetch_historical_data(ticker, end_date, end_date + timedelta(days=1))
        if data is not None and not data.empty:
            close_price = data.iloc[-1]['Close']
            num_shares = info['num_shares']
            portfolio_value += num_shares * close_price
            print(f"Sold {num_shares} shares of {ticker} at {close_price}")
        else:
            print(f"No data available for {ticker} on {end_date}")
    return portfolio_value

def calculate_profit_loss(initial_investment, portfolio_value):
    return portfolio_value - initial_investment

# if __name__ == "__main__":
#     # Example usage
#     selected_tickers = pd.read_csv('selected_top30tickers.txt', header=None)[0].tolist()  # Read tickers from file
#     start_date = datetime(2024, 1, 5).date()  # Replace with your desired start date
#     investment_per_ticker = 1000  # Example initial investment per ticker

#     # Simulate buying
#     portfolio = simulate_buying(selected_tickers, start_date, investment_per_ticker)
#     print(f"Portfolio after buying:\n{portfolio}")

#     # Hold for three months
#     end_date = calculate_end_date(start_date)

#     # Simulate selling
#     portfolio_value = simulate_selling(portfolio, end_date)
#     print(f"Portfolio value after three months: ${portfolio_value:.2f}")

#     # Calculate profit or loss
#     initial_investment = len(selected_tickers) * investment_per_ticker
#     profit_loss = calculate_profit_loss(initial_investment, portfolio_value)
#     print(f"Profit/Loss after three months: ${profit_loss:.2f}")

