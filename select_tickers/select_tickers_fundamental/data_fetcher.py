import requests
import pandas as pd

url_amex = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_tickers.txt"
url_nasdaq = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"
url_nyse = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt"


def fetch_tickers(url):
    response = requests.get(url)
    if response.status_code == 200:
    # Get the content of the file
        tickers = response.text
        tickers_list = tickers.splitlines()
        return tickers_list
    else:
        print("Failed to retrieve the file. Status code:", response.status_code)
        return None
    
def get_unique_tickers():
    tickers_amex = fetch_tickers(url_amex)
    tickers_nasdaq = fetch_tickers(url_nasdaq)
    tickers_nyse = fetch_tickers(url_nyse)
    tickers = tickers_amex + tickers_nasdaq + tickers_nyse
    unique_tickers = list(set(tickers))
    return unique_tickers


if __name__ == "__main__":
    unique_tickers = get_unique_tickers()
    print("Total unique tickers:", len(unique_tickers))
    with open('tickers.txt', 'w') as f:
        for ticker in unique_tickers:
            f.write(f"{ticker}\n")
