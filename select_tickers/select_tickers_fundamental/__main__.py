from datetime import datetime, timedelta
from scheduler import run_periodically

if __name__ == "__main__":
    # Define your trading days interval
    trading_days_interval = 63
    
    # Run the periodic task scheduler
    run_periodically(trading_days_interval)