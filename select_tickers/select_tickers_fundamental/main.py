from datetime import datetime, timedelta
from scheduler import run_periodically
from simulation import simulate_buying, simulate_selling, calculate_profit_loss, calculate_end_date
from data_fetcher import get_unique_tickers
from ticker_selector import select_tickers
import sys
import pandas as pd

def main_task():
    # Fetch unique tickers
    unique_tickers = get_unique_tickers()
    print("Total unique tickers:", len(unique_tickers))

    # Write unique tickers to 'tickers.txt'
    try:
        with open('tickers.txt', 'w') as f:
            for ticker in unique_tickers:
                f.write(f"{ticker}\n")
        print("Unique tickers written to tickers.txt")
    except IOError:
        print("Error writing to tickers.txt")
        return

    # Load tickers from file
    try:
        with open('tickers.txt', 'r') as f:
            tickers = [line.strip() for line in f]
    except FileNotFoundError:
        print("Error: tickers.txt not found.")
        tickers = []

    if tickers:
        print("Processing tickers...")
        selected_tickers = select_tickers(tickers)
        #selected_tickers =  pd.read_csv('selected_top30tickers.txt', header=None)[0].tolist()
        print("Selected Tickers with the Highest Asset Growth:", selected_tickers)

        # Write selected tickers to a file
        try:
            with open('selected_top30tickers.txt', 'w') as f:
                for ticker in selected_tickers:
                    f.write(f"{ticker}\n")
            print("Selected tickers written to selected_top30tickers.txt")
        except IOError:
            print("Error writing to selected_top30tickers.txt")

        # Simulation parameters
        start_date = datetime(2024, 1, 5).date()  # Replace with your desired start date
        investment_per_ticker = 1000  # Example initial investment per ticker

        # Simulate buying
        portfolio = simulate_buying(selected_tickers, start_date, investment_per_ticker)
        print(f"Portfolio after buying:\n{portfolio}")

        # Hold for three months
        end_date = calculate_end_date(start_date)

        # Simulate selling
        portfolio_value = simulate_selling(portfolio, end_date)
        print(f"Portfolio value after three months: ${portfolio_value:.2f}")

        # Calculate profit or loss
        initial_investment = len(selected_tickers) * investment_per_ticker
        profit_loss = calculate_profit_loss(initial_investment, portfolio_value)
        print(f"Profit/Loss after three months: ${profit_loss:.2f}")
    else:
        print("No tickers to process.")

if __name__ == "__main__":
    # Define your trading days interval
    trading_days_interval = 63

    # Check if the script should run periodically
    if len(sys.argv) > 1 and sys.argv[1] == "--scheduler":
        # Run the periodic task scheduler
        run_periodically(trading_days_interval, main_task)
    else:
        # Run the main task immediately
        main_task()
