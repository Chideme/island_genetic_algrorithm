import numpy as np
import random
import math
import heapq
import multiprocess as mp
import talib as ta
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime,date
import time
from islandga import IslandGGA 
from chromosome import Chromosome 
import itertools
import yfinance as yf
import datetime


class Data:

    def __init__(self,stock_ticker,start_date,end_date,test_period):
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(self.stock_ticker, start=self.start_date, end=self.end_date)
        self.test_period = test_period
        self.train_data = []
        self.test_data = []
        self.conditions =[]
        self.strategies =[]


    def clean(self):
        self.data =self.data.reset_index()
        self.data = self.data.sort_values('Date')
        # Convert the 'Date' column to datetime format
        self.data['Date'] = pd.to_datetime(self.data['Date'])

    def split_data(self):
        self.train_data = self.data[self.data['Date'].dt.year < self.test_period]
        self.test_data= self.data[self.data['Date'].dt.year >= self.test_period]

    def plot_data(self):
        # Plotting the 'close' column for 2010-2019
        plt.plot(self.train_data['Date'], self.train_data['close'], color='blue', label='Train')

        # Plotting the 'close' column for 2020
        plt.plot(self.test_data['Date'], self.test_data['close'], color='red', label='Test')

        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.title('Close Price Over Time')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    def create_signals(self):
        # Rename columns to remove spaces in column names
        self.data = self.data.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})
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
        self.data['ATR'] = ta.ATR(np.array(self.data['high']), np.array(self.data['low']), np.array(self.data['close']), timeperiod=14)
        self.data['OBV'] = ta.OBV(np.array(self.data['close'], dtype=float), np.array(self.data['volume'], dtype=float))
        self.data['ADOSC'] = ta.ADOSC(np.array(self.data['high'], dtype=float), np.array(self.data['low'], dtype=float), np.array(self.data['close'], dtype=float), np.array(self.data['volume'], dtype=float), fastperiod=3, slowperiod=10)
        self.data['MFI'] = ta.MFI(np.array(self.data['high'], dtype=float), np.array(self.data['low'], dtype=float), np.array(self.data['close'], dtype=float), np.array(self.data['volume'], dtype=float), timeperiod=14)
        self.data['ROC'] = ta.ROC(np.array(self.data['close'], dtype=float), timeperiod=10)
        self.data['TRIX'] = ta.TRIX(np.array(self.data['close'], dtype=float), timeperiod=30)
        self.data['AROON_UP'], self.data['AROON_DOWN'] = ta.AROON(np.array(self.data['high'], dtype=float), np.array(self.data['low'], dtype=float), timeperiod=14)
        self.data['ADX'] = ta.ADX(np.array(self.data['high'], dtype=float), np.array(self.data['low'], dtype=float), np.array(self.data['close'], dtype=float), timeperiod=14)
        self.data['BBANDS_UPPER'], self.data['BBANDS_MIDDLE'], self.data['BBANDS_LOWER'] = ta.BBANDS(np.array(self.data['close'], dtype=float), timeperiod=20, nbdevup=2, nbdevdn=2)
        self.data['TRIMA'] = ta.TRIMA(np.array(self.data['close'], dtype=float), timeperiod=30)
        self.data['SAR'] = ta.SAR(np.array(self.data['high'], dtype=float), np.array(self.data['low'],), acceleration=0.02, maximum=0.2)
        self.data['STOCHRSI_FASTK'], self.data['STOCHRSI_FASTD'] = ta.STOCHRSI(np.array(self.data['close'], dtype=float), timeperiod=14, fastk_period=5, fastd_period=3)




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
                (data['ATR'].astype(float) > 0),
                (data['ATR'].astype(float) <= 0)
            ],
            'TS10': [
                (data['OBV'].astype(float) > 0),
                (data['OBV'].astype(float) <= 0)
            ],
            'TS11': [
                (data['ADOSC'].astype(float) > 0),
                (data['ADOSC'].astype(float) <= 0)
            ],
            'TS12': [
                (data['MFI'].astype(float) > 30),
                (data['MFI'].astype(float) < 70)
            ],
            'TS13': [
                (data['ROC'].astype(float) > 0),
                (data['ROC'].astype(float) <= 0)
            ],
            'TS14': [
                (data['TRIX'].astype(float) > 0),
                (data['TRIX'].astype(float) <= 0)
            ],
            'TS15': [
                (data['AROON_UP'].astype(float) > 0) & (data['AROON_DOWN'].astype(float) > 0),
                (data['AROON_UP'].astype(float) <= 0) & (data['AROON_DOWN'].astype(float) <= 0)
            ],
            'TS16': [
                (data['ADX'].astype(float) > 20),
                (data['ADX'].astype(float) <= 20)
            ],
            'TS17': [
                (data['BBANDS_UPPER'].astype(float) > data['BBANDS_LOWER'].astype(float)),
                (data['BBANDS_UPPER'].astype(float) < data['BBANDS_LOWER'].astype(float))
            ],
            'TS18': [
                (data['TRIMA'].astype(float) > data['close'].astype(float)),
                (data['TRIMA'].astype(float) < data['close'].astype(float))
            ],
            'TS19': [
                (data['SAR'].astype(float) > data['close'].astype(float)),
                (data['SAR'].astype(float) < data['close'].astype(float))
            ],
            'TS20': [
                (data['STOCHRSI_FASTK'].astype(float) > data['STOCHRSI_FASTD'].astype(float)),
                (data['STOCHRSI_FASTK'].astype(float) < data['STOCHRSI_FASTD'].astype(float))
            ]
        }

        # create a list of the values we want to assign for each condition 1: buy, 0: sell
        values = [1, 0]

        # create a new column and use np.select to assign values to it using our lists as arguments
        for i in conditions:
            data[i] = np.select(conditions[i], values)
        self.strategies = list(conditions.keys())
        return data


    def data_preprocess(self):
        self.clean()
        self.create_signals()
        self.split_data()
        self.train_data = self.generate_candidate_trading_signals(self.train_data)
        self.test_data = self.generate_candidate_trading_signals(self.test_data)
        self.plot_data()


        