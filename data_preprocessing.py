import numpy as np
import random
import math
import heapq
import multiprocess as mp
import talib as ta
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
import time
from islandga import IslandGGA 
from chromosome import Chromosome 
import itertools
import yfinance as yf


def get_last_date_of_month(year, month):
    """Return the last date of the month.
    
    Args:
        year (int): Year, i.e. 2022
        month (int): Month, i.e. 1 for January

    Returns:
        date (datetime): Last date of the current month
    """
    
    if month == 12:
        last_date = datetime(year, month, 31)
    else:
        last_date = datetime(year, month + 1, 1) + timedelta(days=-1)
    #last_date = last_date.date()
    return last_date





class Data:

    def __init__(self,stock_ticker,start_date,end_date,data_period,test_period):
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(self.stock_ticker, start=self.start_date, end=self.end_date)
        self.test_period = test_period
        self.train_data = []
        self.test_data = []
        self.conditions =[]
        self.strategies =[]
        self.top_strategy_names=[]
        self.data_period = data_period


    def clean1(self):
        self.data =self.data.reset_index()
        self.data = self.data.sort_values('Date')
        # Convert the 'Date' column to datetime format
        self.data['Date'] = pd.to_datetime(self.data['Date'])

    def clean(self):
        # Remove rows with any NaN values
        df = self.data
        df = df.dropna()
        
        # Replace infinite values with NaN and then drop those rows
        #df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Remove columns with zero variance
        #zero_var_cols = df_clean.columns[df_clean.std() == 0]
        #df_clean = df_clean.drop(columns=zero_var_cols)
        # Fix multi-index columns (if applicable)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Keep only the second level (actual column names)

        # Reset index to move 'Date' from index to column
        df.reset_index(inplace=True)
        df = df.rename(columns={
                            'Close': 'close', 'Volume': 'volume',
                            'Open': 'open', 'High': 'high', 'Low': 'low'   
                        })
        self.data = df
    

    def split_data(self):
        self.train_data = self.data[self.data['Date'].dt.year < self.test_period]
        self.test_data= self.data[self.data['Date'].dt.year >= self.test_period]

    def plot_data(self):
        # Plotting the 'close' column for 2010-2019
        plt.plot(self.train_data['Date'], self.train_data['close'], color='blue', label='Train')

        # Plotting the 'close' column for 2020
        plt.plot(self.test_data['Date'], self.test_data['close'], color='red', label='Test')

        plt.xlabel('Date')
        plt.ylabel('Close Price')
        #plt.title('Close Price Over Time')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    def create_signals(self):
        # Rename columns to remove spaces in column names
        #self.data = self.data.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})
        # Change date string into date format and sort the dataframe in ascending order

        # Create signals using Talib library
        self.data['5EMA'] = ta.SMA(np.array(self.data['close']), 5)
        self.data['20EMA'] = ta.EMA(np.array(self.data['close']), timeperiod=20)
        self.data['RSI'] = ta.RSI(np.array(self.data['close']), timeperiod=14)
        self.data['WILLR'] = ta.WILLR(np.array(self.data['high']), np.array(self.data['low']), np.array(self.data['close']), timeperiod=14)
        self.data['MOM'] = ta.MOM(np.array(self.data['close']), timeperiod=5)
        self.data['CCI'] = ta.CCI(np.array(self.data['high']), np.array(self.data['low']), np.array(self.data['close']), timeperiod=14)
        self.data['SLOWK'], self.data['SLOWD'] = ta.STOCH(np.array(self.data['high']), np.array(self.data['low']), np.array(self.data['close']), fastk_period=14, slowk_period=3, slowd_period=3)
        self.data['MACD'], self.data['MACDSIGNAL'], self.data['MACDHIST'] = ta.MACD(np.array(self.data['close']), fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['DMI'] = ta.DX(np.array(self.data['high']), np.array(self.data['low']), np.array(self.data['close']), timeperiod=14)
        self.data['OBV'] = ta.OBV(np.array(self.data['close'], dtype=float), np.array(self.data['volume'], dtype=float))
        self.data['MFI'] = ta.MFI(np.array(self.data['high'], dtype=float), np.array(self.data['low'], dtype=float), np.array(self.data['close'], dtype=float), np.array(self.data['volume'], dtype=float), timeperiod=14)
        self.data['BBANDS_UPPER'], self.data['BBANDS_MIDDLE'], self.data['BBANDS_LOWER'] = ta.BBANDS(np.array(self.data['close'], dtype=float), timeperiod=20, nbdevup=2, nbdevdn=2)
        self.data['TRIMA'] = ta.TRIMA(np.array(self.data['close'], dtype=float), timeperiod=30)
        


    def generate_candidate_trading_signals(self, data):
        """Based on parameter setting adopted in Chen et al (2021) """
        data = data.copy()
        conditions = {
            'TS1': [
                (data['5EMA'].astype(float) > data['20EMA'].astype(float)),
                (data['5EMA'].astype(float) < data['20EMA'].astype(float))
            ],
            'TS2': [
                (data['RSI'].astype(float) > 30),
                (data['RSI'].astype(float) < 70)
            ],
            'TS3': [
                (data['WILLR'].astype(float) < 80),
                (data['WILLR'].astype(float) > 20)
            ],
            'TS4': [
                (data['MOM'].astype(float) > 0),
                (data['MOM'].astype(float) <= 0)
            ],
            'TS5': [
                (data['CCI'].astype(float) > 100),
                (data['CCI'].astype(float) <= 100)
            ],
            'TS6': [
                (data['SLOWK'].astype(float) > data['SLOWD'].astype(float)) & (data['SLOWD'].astype(float) < 20),
                (data['SLOWK'].astype(float) < data['SLOWD'].astype(float)) & (data['SLOWD'].astype(float) > 80)
            ],
            'TS7': [
                (data['MACD'].astype(float) > 0),
                (data['MACD'].astype(float) <= 0)
            ],
            'TS8': [
                (data['CCI'].astype(float) > 100),
                (data['CCI'].astype(float) <= -100)
            ],
            'TS9': [
                (data['OBV'].astype(float) > 0),
                (data['OBV'].astype(float) <= 0)
            ],
            
            'TS10': [
                (data['MFI'].astype(float) > 30),
                (data['MFI'].astype(float) < 70)
            ],
            'TS11': [
                (data['BBANDS_UPPER'].astype(float) > data['BBANDS_LOWER'].astype(float)),
                (data['BBANDS_UPPER'].astype(float) < data['BBANDS_LOWER'].astype(float))
            ],
            'TS12': [
                (data['TRIMA'].astype(float) > data['close'].astype(float)),
                (data['TRIMA'].astype(float) < data['close'].astype(float))
            ],     
        }

        # create a list of the values we want to assign for each condition 1: buy, 0: sell
        values = [1, 0]

        # create a new column and use np.select to assign values to it using our lists as arguments
        for i in conditions:
            data[i] = np.select(conditions[i], values)
        self.strategies = list(conditions.keys())
        return data
    
    def generate_signals(self,data,strategy):
        
        monthly_returns = []
        monthly_return = 0
        trade_freq = 0
        monthly_freq = 0
        market_position = 'out'
        max_loss = 0
        for row in data.itertuples(index=False):    
            #close all trade positions at month end. 
            last_date = get_last_date_of_month(row.Date.year,row.Date.month)
            if row.Date == last_date:
                if market_position =='in':
                    sell_price = row.close
                    trade_return = (sell_price - cost_price)/cost_price
                    market_position = 'out' 
                    trade_freq +=1
                    monthly_freq +=1
                    monthly_return += trade_return
                    avg_monthly_return =monthly_return/monthly_freq
                    monthly_returns.append(avg_monthly_return)
                    monthly_return = 0
                    monthly_freq = 0     
                else:
                    try:
                        avg_monthly_return = monthly_return/monthly_freq
                    except ZeroDivisionError:
                        avg_monthly_return = 0
                    monthly_returns.append(avg_monthly_return)
                
                    monthly_return = 0
                    monthly_freq = 0         
            else:
                if market_position == 'out' :
                    if row[data.columns.get_loc(strategy)] == 1:
                        cost_price = row.close
                        market_position = 'in'
                        
                else:
                    sell_price = row.close
                    trade_return = (sell_price - cost_price)/cost_price
                    if trade_return <= -1 or trade_return >= 1: 
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'   
                        
                    if row[data.columns.get_loc(strategy)] == 0 and trade_return >= 1 :
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'
        return monthly_returns

    def strategy_performance(self):
        if self.data_period =="train":
            raw_data = self.train_data
        else:
            raw_data=self.test_data
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy] =  self.generate_signals(raw_data,strategy)
        strategy_performance = pd.DataFrame.from_dict(strategy_performance)
        return strategy_performance

    def rank_strategies(self):
        # Calculate profits for each trading strategy
        ts_data = self.strategy_performance()
        profits = (1 + ts_data).cumprod().iloc[-1] -1
        top_strategies = profits.nlargest(12)
        # Get the names of the top strategies
        self.top_strategy_names = top_strategies.index.tolist()



    def data_preprocess(self):
        self.clean()
        self.create_signals()
        self.split_data()
        self.train_data = self.generate_candidate_trading_signals(self.train_data)
        self.test_data = self.generate_candidate_trading_signals(self.test_data)
        self.rank_strategies()
        self.plot_data()

    ### for comparison methods

    def comparison_metrics(self):
        # Monthly returns DataFrame for each asset
        monthly_returns = self.strategy_performance()
        cumulative_returns = (1 + monthly_returns).cumprod()        
        # Calculate mean cumulative returns
        mean_cumulative_returns = cumulative_returns.mean(axis=1)
        # Calculate portfolio performance at the end of the period
        portfolio_return = mean_cumulative_returns.iloc[-1]-1
        # MDD
        cumulative_max = cumulative_returns.cummax()
        drawdown = cumulative_max - cumulative_returns
        max_drawdown = drawdown.max()
        # Calculate mean Maximum Drawdown (MDD) for all groups
        portfolio_mdd = np.mean(max_drawdown)
        num_groups = len(self.top_strategy_names) # Number of groups
        # Calculate equal weights for each group
        equal_weights = 1 / num_groups
        # Create an array of equal weights for each group
        weights= np.full(num_groups, equal_weights)
        covariance_matrix = monthly_returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)

        ####### buy and hold

       

        return  print(f"Return: {portfolio_return}\nPortfolio MDD: {portfolio_mdd}\nPortfolio Std Dev: {portfolio_std_dev}")


    def buy_and_hold(self):
         # Load the monthly returns data.
        if self.data_period =="train":
            raw_data = self.train_data.copy()
        else:
            raw_data=self.test_data.copy()
            
        asset_prices = raw_data
        # Set the 'Date' column as the index of the DataFrame if it's not already
        asset_prices.set_index('Date', inplace=True)

        # Resample the asset prices to monthly frequency and select the first and last values of each month
        monthly_prices = asset_prices.resample('M').agg({'close': ['first', 'last']})

        # Calculate the monthly returns as the percentage change in price
        monthly_returns = monthly_prices['close', 'last'].pct_change()

        # Calculate the cumulative returns
        cumulative_returns = (1 + monthly_returns).cumprod()
        # Calculate the profit as the difference between the final cumulative return and 1 (initial investment)
        profit = cumulative_returns.iloc[-1]-1
        cumulative_max = cumulative_returns.cummax()
        drawdown = cumulative_max - cumulative_returns
        max_drawdown = drawdown.max()


        # Calculate the maximum drawdown.
        # Print the the results.
        print(f"Buy and Hold Strategy Profit: {profit:.2f}.")
        print(f"Buy and Hold Strategy MDD   : {max_drawdown:.2f}.")
        
        