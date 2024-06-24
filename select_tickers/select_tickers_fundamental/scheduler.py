import time
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from ticker_selector import select_tickers
from data_fetcher import get_unique_tickers


def calculate_next_trading_day(current_date, exchange_code, num_days):
    try:
        # Fetch the trading calendar for the given exchange code
        trading_calendar = get_calendar(exchange_code)

        # Fetch the trading schedule
        schedule = trading_calendar.schedule(start_date=current_date, end_date=current_date + timedelta(days=num_days))

        # Iterate through the schedule to find the next trading day after current_date
        for trading_date in schedule.index:
            if trading_date > current_date:
                return trading_date
        
        return None  # No trading day found within the specified range

    except Exception as e:
        print(f"Error fetching trading calendar: {e}")
        return None
def run_periodically(trading_days_interval):
    # Initialize trading calendar (example using NYSE calendar)
    trading_calendar = get_calendar('XNYS')

    while True:
        # Calculate next run date
        current_date = datetime.now()
        next_run_date = calculate_next_trading_day(current_date, trading_calendar, trading_days_interval)
        
        # Execute your script here
        print(f"Running script at {datetime.now()}...")
        tickers = get_unique_tickers()
        selected_tickers = select_tickers(tickers)
        print("Selected tickers:", selected_tickers)
        
        # Sleep until next run
        sleep_duration = (next_run_date - datetime.now()).total_seconds()
        print(f"Next run scheduled at {next_run_date}. Sleeping for {sleep_duration} seconds...")
        time.sleep(sleep_duration)

if __name__ == "__main__":
    # Set interval in trading days (63 trading days)
    trading_days_interval = 63
    run_periodically(trading_days_interval)