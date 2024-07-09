from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime as datetime
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest
from alpaca_trade_api.rest import REST, Order

def place_new_order(api_key, secret_key, symbol, qty, side, order_type, time_in_force, limit_price=None, paper=True):
    # Initialize the trading client
    trading_client = TradingClient(api_key, secret_key, paper=paper)
    side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
    
    # Determine the time in force
    tif_enum = getattr(TimeInForce, time_in_force.upper())
    
    # Create the appropriate order request
    if order_type.lower() == 'market':
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif_enum
        )
    elif order_type.lower() == 'limit':
        if limit_price is None:
            raise ValueError("Limit price must be provided for limit orders")
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif_enum,
            limit_price=limit_price
        )
    else:
        raise ValueError("Invalid order type. Must be 'market' or 'limit'")
    
    # Submit the order
    response = trading_client.submit_order(request)
    return response


if __name__ == "__main__":
    # API keys
    api_key = 'PKNC2Y52PK84SV0AJ5G0'
    secret_key = 'VsL2M0iivKdfEEbM6QvdCQsuqnpvaz9s91utOvhI'

    # Submit a market order to buy 1 share of Apple at market price
    response = place_new_order(api_key, secret_key, 'AAPL', 1, 'buy', 'market', 'gtc')
    print("Market order response:", response)

    # Submit a limit order to attempt to sell 1 share of AMD at a particular price ($20.50) when the market opens
    response = place_new_order(api_key, secret_key, 'AMD', 1, 'sell', 'limit', 'opg', limit_price=20.50)
    print("Limit order response:", response)
