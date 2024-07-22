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
from trade_env import StockEnvTrade
from train_env import StockEnvTrain
from finrl import config
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
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

def add_technical_indicator(df):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        stock = Sdf.retype(df.copy())
        stock['close'] = stock['adjcp']
        unique_ticker = stock.tic.unique()

        macd = pd.DataFrame()
        rsi = pd.DataFrame()

        #temp = stock[stock.tic == unique_ticker[0]]['macd']
        for i in range(len(unique_ticker)):
            ## macd
            temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
            temp_macd = pd.DataFrame(temp_macd)
            macd = macd.append(temp_macd, ignore_index=True)
            ## rsi
            temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
            temp_rsi = pd.DataFrame(temp_rsi)
            rsi = rsi.append(temp_rsi, ignore_index=True)

        df['macd'] = macd
        df['rsi'] = rsi
        return df

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df

def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets

    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)


    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index

# def data_split(df, start_date, end_date):
#     return df[(df['date'] >= start_date) & (df['date'] < end_date)].reset_index(drop=True)
def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def model_train(env_train):
    df = FeatureEngineer(df.copy(),
                      use_technical_indicator=True,
                      tech_indicator_list = config.INDICATORS,
                      use_turbulence=True,
                      user_defined_feature = False).preprocess_data()
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # model settings
    model_ddpg = DDPG('MlpPolicy',
                    env_train,
                    batch_size=64,
                    buffer_size=100000,
                    param_noise=param_noise,
                    action_noise=action_noise,
                    verbose=0,
                    tensorboard_log="./multiple_stock_tensorboard/")

    ## 250k timesteps: took about 20 mins to finish
    model_ddpg.learn(total_timesteps=250000, tb_log_name="DDPG_run_1")
# def design_strategy(tickers_file):
#     try:
#         # Fetch data and preprocess
#         df1 = fetch_30(tickers_file, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)
#         #print(df1)
        
        
#         if df1 is None:
#             print("Error fetching data. Exiting strategy design.")
#             return None, None
#         print("Shape of df1:", df1.shape)
        
#         # Split data into train and trade periods
#         train = data_split(df1, TRAIN_START_DATE, TRAIN_END_DATE)
#         trade = data_split(df1, TRADE_START_DATE, TRADE_END_DATE)

#         print("Train DataFrame shape:", train.shape)
#         print("Trade DataFrame shape:", trade.shape)
        
#         # Calculate state space dynamically based on INDICATORS
#         stock_dim = len(train['tic'].unique())
#         print(f"Stock Dimension: {stock_dim}")
#         # state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
#         state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
#         print(f"Calculated state space: {state_space}")     
        
#         env_kwargs = {
#             "num_stock_shares": [stock_dim],  # List of zeros
#             "stock_dim": stock_dim,
#             "hmax": 100,
#             "initial_amount": 1000000,
#             "buy_cost_pct": [0.001] * stock_dim,
#             "sell_cost_pct": [0.001] * stock_dim,
#             "reward_scaling": 1e-4,
#             "state_space": state_space,
#             "action_space": stock_dim,
#             "tech_indicator_list": INDICATORS,
#             "print_verbosity": 10
#         }
#         print("Creating training environment...")
#         env_train = StockTradingEnv(train, **env_kwargs)
#         print("Training environment created.")
        
#         print("Creating trading environment...")
#         env_trade = StockTradingEnv(trade, **env_kwargs)
#         print("Trading environment created.")
#         print(f"Observation space shape: {env_train.observation_space.shape}")
        
#         #Train model
#         agent = DRLAgent(env=env_train)
#         model = agent.get_model("ddpg")
#         print("Training model...")

#         # print(env_train.initial) #=true
#         obs = env_train.reset()
#         print(len(env_train.state))
#         print(len(obs[0]))
#         #print(env_train.df.day)

#         trained_model = agent.train_model(model=model, tb_log_name='DDPG', total_timesteps=5000)
#         print("Model training completed.")

#         #Evaluate model
#  #       df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=env_trade)
#  #       print("Evaluation results:", df_account_value, df_actions)
#         return df_account_value, df_actions
    
#     except Exception as e:
#         print(f"Error in design_strategy: {str(e)}")
#         return None, None

def DRL_prediction(model, data, env, obs):
    print("==============Model Prediction===========")
    for i in range(len(data.index.unique())):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    
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
