import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from alpaca_connect import place_new_order 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from alpaca.trading.client import TradingClient
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
sys.path.append("../FinRL-Library")
import itertools
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
from alpaca_trade_api.rest import REST, TimeFrame
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2021-01-01'
TRADE_START_DATE = '2021-01-01'
TRADE_END_DATE = '2022-01-01'


def fetch_30(tickers_file, train_start_date, train_end_date, test_start_date, test_end_date):
    try:
        check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
        print("==============Start Fetching Data===========")
        
        # Read tickers from file
        selected_30 = pd.read_csv(tickers_file, header=None)
        tickers = selected_30[0].tolist()
        
        if not tickers:
            print("Error: No tickers found in the provided file.")
            return None
        
        # Fetch data from Yahoo Finance
        df = YahooDownloader(start_date=train_start_date,
                             end_date=test_end_date,
                             ticker_list=tickers).fetch_data()
        
        if df.empty:
            print("Error: Failed to fetch data from Yahoo Finance.")
            return None
        
        print("==============Data Loaded===========")
        
        # Sort data by date and ticker
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Feature engineering and processing
        fe = FeatureEngineer(use_technical_indicator=True,
                             tech_indicator_list=INDICATORS,
                             use_turbulence=True,
                             user_defined_feature=False)
        
        processed = fe.preprocess_data(df)
        processed.fillna(0, inplace=True)
        processed.replace(np.inf, 0, inplace=True)
        
        # Create combinations of all dates and tickers
        list_ticker = processed["tic"].unique().tolist()
        list_date = pd.date_range(processed['date'].min(), processed['date'].max()).astype(str)
        combination = list(itertools.product(list_date, list_ticker))
        
        # Merge processed data with all date-ticker combinations
        df1 = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
        df1 = df1[df1['date'].isin(processed['date'])]
        df1 = df1.sort_values(['date', 'tic']).reset_index(drop=True)
        df1.fillna(0, inplace=True)
        
        print(f"Number of unique tickers fetched: {len(df.tic.unique())}")
        
        return df1
    
    except Exception as e:
        print(f"Error in fetch_30: {str(e)}")
        return None


# def algorithm_build(df):
#     stock_dimension = len(df.tic.unique())
#     state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
#     print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
#     env_kwargs = {
#     "hmax": 100, 
#     "initial_amount": 1000000, 
#     "buy_cost_pct": 0.001, 
#     "sell_cost_pct": 0.001, 
#     "state_space": state_space, 
#     "stock_dim": stock_dimension, 
#     "tech_indicator_list": INDICATORS,
#     "action_space": stock_dimension, 
#     "reward_scaling": 1e-4,
#     "print_verbosity":5}
#     rebalance_window = 63 #63 # rebalance_window is the number of days to retrain the model
#     validation_window = 63 #63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

#     ensemble_agent = DRLEnsembleAgent(df=df,
#                     train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
#                     val_test_period=(TEST_START_DATE,TEST_END_DATE),
#                     rebalance_window=rebalance_window, 
#                     validation_window=validation_window, 
#                     **env_kwargs)
#     A2C_model_kwargs = {
#                     'n_steps': 5,
#                     'ent_coef': 0.005,
#                     'learning_rate': 0.0007
#                     }

#     PPO_model_kwargs = {
#                         "ent_coef":0.01,
#                         "n_steps": 2, #2048
#                         "learning_rate": 0.00025,
#                         "batch_size": 128
#                         }

#     DDPG_model_kwargs = {
#                         #"action_noise":"ornstein_uhlenbeck",
#                         "buffer_size": 10000, #10_000
#                         "learning_rate": 0.0005,
#                         "batch_size": 64
#                         }

#     TD3_model_kwargs = {
#                         "action_noise":"ornstein_uhlenbeck",
#                         "buffer_size": 10000, #10_000
#                         "learning_rate": 0.0005,
#                         "batch_size": 64
#                         }

#     SAC_model_kwargs = {
#                         "buffer_size": 10000, #10_000
#                         "learning_rate": 0.0005,
#                         "batch_size": 64
#                         }

#     timesteps_dict = {'a2c' : 1000, #10_000 each
#                     'ppo' : 1000, 
#                     'ddpg' : 1000,
#                     'td3' : 1000,
#                     'sac' : 1000}
#     df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs=A2C_model_kwargs,
#                                                  PPO_model_kwargs=PPO_model_kwargs,
#                                                  DDPG_model_kwargs=DDPG_model_kwargs,
#                                                  TD3_model_kwargs=TD3_model_kwargs,
#                                                  SAC_model_kwargs=SAC_model_kwargs,
#                                                  timesteps_dict=timesteps_dict)
#     return df_summary
    


# df = fetch_30('../select_tickers/select_tickers_fundamental/selected_tickers.txt',
#          train_start_date='2009-04-01',
#          train_end_date='2022-01-01',
#          test_start_date='2022-01-01',
#          test_end_date='2024-06-01')
# df_summary = algorithm_build(df)
# print(df_summary)
def data_split(df, start_date, end_date):
    return df[(df['date'] >= start_date) & (df['date'] < end_date)].reset_index(drop=True)

def design_strategy(tickers_file):
    try:
        # Fetch data and preprocess
        df1 = fetch_30(tickers_file, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)
        #print(df1)
        
        
        if df1 is None:
            print("Error fetching data. Exiting strategy design.")
            return None, None
        print("Shape of df1:", df1.shape)
        
        # Split data into train and trade periods
        train = data_split(df1, TRAIN_START_DATE, TRAIN_END_DATE)
        trade = data_split(df1, TRADE_START_DATE, TRADE_END_DATE)

        print("Train DataFrame shape:", train.shape)
        print("Trade DataFrame shape:", trade.shape)
        
        # Calculate state space dynamically based on INDICATORS
        stock_dim = len(train['tic'].unique())
        print(f"Stock Dimension: {stock_dim}")
        # state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
        state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
        print(f"Calculated state space: {state_space}")     
        
        env_kwargs = {
            "num_stock_shares": [stock_dim],  # List of zeros
            "stock_dim": stock_dim,
            "hmax": 100,
            "initial_amount": 1000000,
            "buy_cost_pct": [0.001] * stock_dim,
            "sell_cost_pct": [0.001] * stock_dim,
            "reward_scaling": 1e-4,
            "state_space": state_space,
            "action_space": stock_dim,
            "tech_indicator_list": INDICATORS,
            "print_verbosity": 10
        }
        print("Creating training environment...")
        env_train = StockTradingEnv(train, **env_kwargs)
        print("Training environment created.")
        
        print("Creating trading environment...")
        env_trade = StockTradingEnv(trade, **env_kwargs)
        print("Trading environment created.")
        print(f"Observation space shape: {env_train.observation_space.shape}")
        
        #Train model
        agent = DRLAgent(env=env_train)
        model = agent.get_model("ddpg")
        print("Training model...")

        # print(env_train.initial) #=true
        obs = env_train.reset()
        print(len(env_train.state))
        print(len(obs[0]))
        #print(env_train.df.day)

        trained_model = agent.train_model(model=model, tb_log_name='DDPG', total_timesteps=5000)
        print("Model training completed.")

        #Evaluate model
 #       df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=env_trade)
 #       print("Evaluation results:", df_account_value, df_actions)
        return df_account_value, df_actions
    
    except Exception as e:
        print(f"Error in design_strategy: {str(e)}")
        return None, None


    
def execute_trades(alpaca_api, df_actions):
    try:
        # Connect to Alpaca
        api_key = alpaca_api['API_KEY']
        secret_key = alpaca_api['API_SECRET']
        paper = alpaca_api.get('PAPER', True)  # Defaults to True if not provided
        trading_client = TradingClient(api_key, secret_key, paper=paper)
        
        # Iterate over each action in df_actions
        for index, action in df_actions.iterrows():
            ticker = action['tic']
            qty = action['qty']  # Adjust based on your strategy output
            side = action['action']  # Assuming 'action' indicates 'buy' or 'sell'
            
            if action['order_type'].lower() == 'market':
                order_type = 'market'
                time_in_force = 'gtc'  # Good till canceled, adjust based on strategy
                limit_price = None  # No limit price for market order
            elif action['order_type'].lower() == 'limit':
                order_type = 'limit'
                time_in_force = 'gtc'  # Good till canceled, adjust based on strategy
                limit_price = action['limit_price']  # Retrieve limit price from df_actions
            
            # Example: Submit order based on strategy action
            response = place_new_order(api_key, secret_key, ticker, qty, side, order_type, time_in_force, limit_price)
            print(f"Submitted {side} {order_type} order for {qty} shares of {ticker}. Response: {response}")
    
    except Exception as e:
        print(f"Error executing trades: {str(e)}")

if __name__ == "__main__":
    # Alpaca API credentials
    alpaca_api = {
        'API_KEY': 'PKNC2Y52PK84SV0AJ5G0',
        'API_SECRET': 'VsL2M0iivKdfEEbM6QvdCQsuqnpvaz9s91utOvhI',
        'PAPER': True,  # Set to False for live trading
    }
    tickers_file = '../select_tickers/select_tickers_fundamental/selected_top30tickers.txt'  # Replace with your selected tickers file
    df_account_value, df_actions = design_strategy(tickers_file)
    
    if df_actions is not None:
        print("Strategy actions:\n", df_actions)
        
        # Execute trades based on the strategy actions
        execute_trades(alpaca_api, df_actions)
    
    else:
        print("Evaluation results:", df_account_value, df_actions)
        print("Error occurred in strategy execution. No actions to execute.")



# from alpaca_trade_api.rest import REST

# def fetch_selected_tickers(tickers_file):
#     try:
#         with open(tickers_file, 'r') as f:
#             tickers = [line.strip() for line in f.readlines()]
#         return tickers
#     except Exception as e:
#         print(f"Error fetching tickers from file: {str(e)}")
#         return []

# def execute_trades(alpaca_api, tickers):
#     try:
#         # Connect to Alpaca
#         api = REST(alpaca_api['API_KEY'], alpaca_api['API_SECRET'], base_url=alpaca_api['BASE_URL'])
        
#         # Example: Buy 10 shares of each selected ticker
#         for ticker in tickers:
#             try:
#                 api.submit_order(
#                     symbol=ticker,
#                     qty=10,
#                     side='buy',
#                     type='market',
#                     time_in_force='gtc'
#                 )
#                 print(f"Submitted buy order for {ticker}")
#             except Exception as e:
#                 print(f"Error executing trade for {ticker}: {str(e)}")
#                 continue  # Skip to the next ticker if there's an error
    
#     except Exception as e:
#         print(f"Error executing trades: {str(e)}")

# if __name__ == "__main__":
#     # Alpaca API credentials
#     alpaca_api = {
#         'API_KEY': 'PKNC2Y52PK84SV0AJ5G0',
#         'API_SECRET': 'VsL2M0iivKdfEEbM6QvdCQsuqnpvaz9s91utOvhI',
#         'BASE_URL': 'https://paper-api.alpaca.markets'  # or 'https://api.alpaca.markets' for live trading
#     }
    
#     tickers_file = '../select_tickers/select_tickers_fundamental/selected_tickers.txt'  # Replace with your selected tickers file
#     selected_tickers = fetch_selected_tickers(tickers_file)
    
#     if selected_tickers:
#         print(f"Executing trades for {len(selected_tickers)} tickers.")
#         execute_trades(alpaca_api, selected_tickers)
#     else:
#         print("No tickers found to execute trades.")
