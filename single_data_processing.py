import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime
import matplotlib.pyplot as plt


def clean_financial_data(df):
    # Remove rows with any NaN values
    df_clean = df.dropna()
    
    # Replace infinite values with NaN and then drop those rows
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Remove columns with zero variance
    #zero_var_cols = df_clean.columns[df_clean.std() == 0]
    #df_clean = df_clean.drop(columns=zero_var_cols)
    df = df.rename(columns={
                        'Close': 'close', 'Volume': 'volume',
                        'Open': 'open', 'High': 'high', 'Low': 'low'
                    })
    
    return df_clean

class SingleAssetTI:
    def __init__(self, ticker, start_date, end_date,test_period):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.test_period = test_period
        self.data = self.load_and_prepare_data()
        self.train = self.data[self.data['Date'].dt.year < self.test_period]
        self.test= self.data[self.data['Date'].dt.year >= self.test_period]
        self.strategy_results = {}

    def load_and_prepare_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df = df.rename(columns={
            'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'
        }).dropna()

        return df

        

    def create_indicators(self,data):
        

        # Create signals using Talib library
        data['5EMA'] = ta.SMA(np.array(data['close']), 5)
        data['20EMA'] = ta.EMA(np.array(data['close']), timeperiod=20)
        data['RSI'] = ta.RSI(np.array(data['close']), timeperiod=14)
        data['WILLR'] = ta.WILLR(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['MOM'] = ta.MOM(np.array(data['close']), timeperiod=5)
        data['CCI'] = ta.CCI(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['SLOWK'], data['SLOWD'] = ta.STOCH(np.array(data['high']), np.array(data['low']), np.array(data['close']), fastk_period=14, slowk_period=3, slowd_period=3)
        data['MACD'], data['MACDSIGNAL'], data['MACDHIST'] = ta.MACD(np.array(data['close']), fastperiod=12, slowperiod=26, signalperiod=9)
        data['DMI'] = ta.DX(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['ATR'] = ta.ATR(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['OBV'] = ta.OBV(np.array(data['close'], dtype=float), np.array(data['volume'], dtype=float))
        data['ADOSC'] = ta.ADOSC(np.array(data['high'], dtype=float), np.array(data['low'], dtype=float), np.array(data['close'], dtype=float), np.array(data['volume'], dtype=float), fastperiod=3, slowperiod=10)
        data['MFI'] = ta.MFI(np.array(data['high'], dtype=float), np.array(data['low'], dtype=float), np.array(data['close'], dtype=float), np.array(data['volume'], dtype=float), timeperiod=14)
        data['ROC'] = ta.ROC(np.array(data['close'], dtype=float), timeperiod=10)
        data['TRIX'] = ta.TRIX(np.array(data['close'], dtype=float), timeperiod=30)
        data['AROON_UP'], data['AROON_DOWN'] = ta.AROON(np.array(data['high'], dtype=float), np.array(data['low'], dtype=float), timeperiod=14)
        data['ADX'] = ta.ADX(np.array(data['high'], dtype=float), np.array(data['low'], dtype=float), np.array(data['close'], dtype=float), timeperiod=14)
        data['BBANDS_UPPER'], data['BBANDS_MIDDLE'], data['BBANDS_LOWER'] = ta.BBANDS(np.array(data['close'], dtype=float), timeperiod=20, nbdevup=2, nbdevdn=2)
        data['TRIMA'] = ta.TRIMA(np.array(data['close'], dtype=float), timeperiod=30)
        data['SAR'] = ta.SAR(np.array(data['high'], dtype=float), np.array(data['low'],), acceleration=0.02, maximum=0.2)
        data['STOCHRSI_FASTK'], data['STOCHRSI_FASTD'] = ta.STOCHRSI(np.array(data['close'], dtype=float), timeperiod=14, fastk_period=5, fastd_period=3)

        return data


    def evaluate_strategy(self, data, buy_cond, sell_cond):
        
        data['signal'] = np.select([buy_cond, sell_cond], [1, 0], default=np.nan)
        data['signal'] = data['signal'].ffill().fillna(0)
        data['position'] = data['signal'].shift(1)

        daily_ret = data['close'].pct_change() * data['position']
        daily_ret = daily_ret.dropna()

        if daily_ret.std() == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std()

        return sharpe, daily_ret


    
    def generate_strategies(self, data):
        #data = self.data.copy()
        processed_data = self.create_indicators(data)

        # Define all strategy combinations
        buying_indicators = [
            lambda d: d['5EMA'].astype(float) > d['20EMA'].astype(float),
            lambda d: d['RSI'].astype(float) > 30,
            lambda d: d['WILLR'].astype(float) < -80,
            lambda d: d['MOM'].astype(float) > 0,
            lambda d: d['CCI'].astype(float) > 100
        ]
        selling_indicators = [
            lambda d: d['5EMA'].astype(float) < d['20EMA'].astype(float),
            lambda d: d['RSI'].astype(float) < 70,
            lambda d: d['WILLR'].astype(float) > -20,
            lambda d: d['MOM'].astype(float) <= 0,
            lambda d: d['CCI'].astype(float) <= 100
        ]

        strategy_combinations = list(zip(buying_indicators, selling_indicators))

        # Optionally expand to 50 strategies with more combinations
        # You can uncomment below and add more logic-based combinations if needed
        # while len(strategy_combinations) < 50:
        #     strategy_combinations.append((
        #         lambda d: d['MACD'] > d['MACDSIGNAL'],
        #         lambda d: d['MACD'] < d['MACDSIGNAL']
        #     ))

        strategy_id = 1
        for buy, sell in strategy_combinations:
            strat_name = f"TS{strategy_id}"
            try:
                sharpe, returns = self.evaluate_strategy(processed_data, buy(processed_data), sell(processed_data))
                self.strategy_performance[strat_name] = {
                    'sharpe': sharpe,
                    'returns': returns,
                    'buy_condition': buy,
                    'sell_condition': sell
                }
                self.strategy_returns[strat_name] = returns
            except Exception as e:
                print(f"Error evaluating strategy {strat_name}: {e}")
            strategy_id += 1

        # Combine all strategy returns into a DataFrame
        self.daily_returns = pd.DataFrame(self.strategy_returns)
        return self.daily_returns
    

    def generate_test_returns(self, test_data):
        """
        Applies all trained strategies to the test data and returns their individual return series.
        """
        processed_data = self.create_indicators(test_data)
        test_strategy_returns = {}

        for strat_name, strat_info in self.strategy_performance.items():
            try:
                buy_condition = strat_info['buy_condition']
                sell_condition = strat_info['sell_condition']

                # Evaluate returns on test data
                _, returns = self.evaluate_strategy(
                    processed_data,
                    buy_condition(processed_data),
                    sell_condition(processed_data)
                )

                test_strategy_returns[strat_name] = returns

            except Exception as e:
                print(f"[ERROR] Strategy {strat_name} failed: {e}")

        return pd.DataFrame(test_strategy_returns)




    def select_top_strategies(self, top_n=3):
        df = pd.DataFrame({k: v['returns'] for k, v in self.strategy_results.items()})
        metrics = {}
        for k in df.columns:
            r = df[k]
            sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() != 0 else 0
            metrics[k] = sharpe
        top_strats = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return df[[s[0] for s in top_strats]]
    
   

    def plot_data(self):
        # Plotting the 'close' column for 2010-2019
        plt.plot(self.train['Date'], self.train['close'], color='blue', label='Train')

        # Plotting the 'close' column for 2020
        plt.plot(self.test['Date'], self.test['close'], color='red', label='Test')

        plt.xlabel('Date')
        plt.ylabel('Close Price')
        #plt.title('Close Price Over Time')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    def data_preprocess(self):
        
        self.train_data = self.generate_strategies(self.train_data)
        self.test_data = self.generate_test_returns(self.test_data)
        #self.rank_strategies()
        self.plot_data()
