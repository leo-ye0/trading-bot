import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from time import sleep
from data_fetcher import get_unique_tickers

def fetch_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        return balance_sheet
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None
    
def calculate_asset_growth(balance_sheet):
    try:
        total_assets = balance_sheet.loc['Total Assets']
        
        # Drop rows with NaN values
        total_assets = total_assets.dropna()
        # print(total_assets)
        
        if len(total_assets) < 2:
            return None
        
        # Calculate annual growth rates
        annual_growth = []
        for i in range(len(total_assets) - 1):
            latest_assets = total_assets.iloc[i]
            previous_assets = total_assets.iloc[i + 1]
            
            # Handle division by zero gracefully
            if previous_assets == 0:
                growth = np.nan  # Set growth to NaN for missing data
            else:
                growth = (latest_assets - previous_assets) / previous_assets
            
            annual_growth.append(growth)
        
        # Calculate the average annual growth rate
        average_growth = np.nanmean(annual_growth)  # Ignore NaN values when calculating mean
        
        return average_growth

    except Exception as e:
        print(f"Error calculating asset growth: {e}")
        return None

    
def select_tickers(tickers, batch_size=100):
    ticker_growth = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for ticker in batch:
            balance_sheet = fetch_financials(ticker)
            if balance_sheet is not None:
                asset_growth = calculate_asset_growth(balance_sheet)
                if asset_growth is not None:
                    ticker_growth.append((ticker, asset_growth))
        print(f"Processed batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}")
        sleep(1)  # To avoid rate limiting

    # Sort by asset growth in ascending order and select the 30 tickers with the lowest growth
    ticker_growth.sort(key=lambda x: x[1], reverse=True)
    selected_tickers = [ticker for ticker, growth in ticker_growth[:30]]

    return selected_tickers

