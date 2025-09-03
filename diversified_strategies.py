import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class DiversifiedTradingStrategies:
    def __init__(self, tickers, start_date, end_date, test_period):
        self.tickers = tickers  # List of multiple tickers
        self.start_date = start_date
        self.end_date = end_date
        self.test_period = test_period
        self.all_data = {}
        self.train_data = {}
        self.test_data = {}
        self.strategy_returns = {}
        self.strategy_performance = {}
        self.load_all_data()
        self.strategy_names = []
        
    def load_all_data(self):
        """Load data for all tickers"""
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.reset_index(inplace=True)
                df = df.rename(columns={
                    'Close': 'close', 'Open': 'open', 'High': 'high', 
                    'Low': 'low', 'Volume': 'volume'
                }).dropna()
                
                self.all_data[ticker] = df.copy()
                self.train_data[ticker] = df[df['Date'].dt.year < self.test_period]
                self.test_data[ticker] = df[df['Date'].dt.year >= self.test_period]
                print(f"Loaded {ticker}: {len(df)} records")
            except Exception as e:
                print(f"Failed to load {ticker}: {e}")

    def create_indicators(self, data):
        """Create technical indicators for a single dataset"""
        # Basic indicators
        data['5EMA'] = ta.SMA(np.array(data['close']), 5)
        data['20EMA'] = ta.EMA(np.array(data['close']), timeperiod=20)
        data['RSI'] = ta.RSI(np.array(data['close']), timeperiod=14)
        data['WILLR'] = ta.WILLR(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['MOM'] = ta.MOM(np.array(data['close']), timeperiod=5)
        data['CCI'] = ta.CCI(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['SLOWK'], data['SLOWD'] = ta.STOCH(np.array(data['high']), np.array(data['low']), np.array(data['close']), fastk_period=14, slowk_period=3, slowd_period=3)
        data['MACD'], data['MACDSIGNAL'], data['MACDHIST'] = ta.MACD(np.array(data['close']), fastperiod=12, slowperiod=26, signalperiod=9)
        data['ADX'] = ta.ADX(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod=14)
        data['OBV'] = ta.OBV(np.array(data['close'], dtype=float), np.array(data['volume'], dtype=float))
        data['MFI'] = ta.MFI(np.array(data['high'], dtype=float), np.array(data['low'], dtype=float), np.array(data['close'], dtype=float), np.array(data['volume'], dtype=float), timeperiod=14)
        data['BBANDS_UPPER'], data['BBANDS_MIDDLE'], data['BBANDS_LOWER'] = ta.BBANDS(np.array(data['close'], dtype=float), timeperiod=20, nbdevup=2, nbdevdn=2)
        data['SAR'] = ta.SAR(np.array(data['high'], dtype=float), np.array(data['low']), acceleration=0.02, maximum=0.2)
        return data

    def evaluate_strategy(self, data, buy_cond, sell_cond):
        """Evaluate a single strategy on given data"""
        data = data.copy()
        data['signal'] = np.select([buy_cond, sell_cond], [1, 0], default=np.nan)
        data['signal'] = data['signal'].ffill().fillna(0)
        data['position'] = data['signal'].shift(1)
        
        daily_ret = data['close'].pct_change() * data['position']
        daily_ret = daily_ret.dropna()
        
        if len(daily_ret) == 0 or daily_ret.std() == 0:
            return 0, pd.Series(dtype=float)
        
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std()
        return sharpe, daily_ret

    def create_diversified_strategies(self):
        """Create genuinely diversified strategies across multiple dimensions"""
        
        # 1. ASSET-SPECIFIC STRATEGIES (20 strategies)
        asset_specific_strategies = []

        # Replace the lambda functions with proper closure handling:
        for i, ticker in enumerate(self.tickers[:5]):  # Use first 5 tickers
            # Each ticker gets 4 specialized strategies
            strategies = [
                {
                    'name': f'Momentum_{ticker}',
                    'ticker': ticker,
                    'buy': (lambda d, t=ticker: (d['RSI'] > 50) & (d['5EMA'] > d['20EMA'])),
                    'sell': (lambda d, t=ticker: (d['RSI'] < 50) | (d['5EMA'] < d['20EMA']))
                },
                {
                    'name': f'MeanReversion_{ticker}',
                    'ticker': ticker,
                    'buy': (lambda d, t=ticker: (d['RSI'] < 30) & (d['close'] < d['BBANDS_LOWER'])),
                    'sell': (lambda d, t=ticker: (d['RSI'] > 70) | (d['close'] > d['BBANDS_UPPER']))
                },
                {
                    'name': f'Breakout_{ticker}',
                    'ticker': ticker,
                    'buy': (lambda d, t=ticker: (d['close'] > d['BBANDS_UPPER']) & (d['ADX'] > 25)),
                    'sell': (lambda d, t=ticker: (d['close'] < d['BBANDS_MIDDLE']) | (d['ADX'] < 20))
                },
                {
                    'name': f'Volume_{ticker}',
                    'ticker': ticker,
                    'buy': (lambda d, t=ticker: (d['OBV'] > d['OBV'].shift(1)) & (d['MFI'] < 20)),
                    'sell': (lambda d, t=ticker: (d['OBV'] < d['OBV'].shift(1)) | (d['MFI'] > 80))
                }
            ]
            asset_specific_strategies.extend(strategies)
        
        

        # 2. MULTI-TIMEFRAME STRATEGIES (20 strategies)
        # Simulate different timeframes by using different MA periods
        timeframe_strategies = []
        timeframes = [
            {'name': 'Short', 'fast_ma': 3, 'slow_ma': 8, 'rsi_period': 7},
            {'name': 'Medium', 'fast_ma': 10, 'slow_ma': 21, 'rsi_period': 14},
            {'name': 'Long', 'fast_ma': 21, 'slow_ma': 50, 'rsi_period': 21}
        ]
        
        for tf in timeframes:
            for ticker in self.tickers[:7]:  # Use 7 tickers
                strategy = {
                    'name': f'{tf["name"]}TF_{ticker}',
                    'ticker': ticker,
                    'timeframe': tf,
                    'buy': lambda d, tf=tf: (d['5EMA'] > d['20EMA']) & (d['RSI'] > 40),
                    'sell': lambda d, tf=tf: (d['5EMA'] < d['20EMA']) | (d['RSI'] < 60)
                }
                timeframe_strategies.append(strategy)
                if len(timeframe_strategies) >= 20:
                    break
            if len(timeframe_strategies) >= 20:
                break

        # 3. CROSS-ASSET CORRELATION STRATEGIES (20 strategies)
        correlation_strategies = []
        if len(self.tickers) >= 2:
            # Create pairs of assets for correlation-based strategies
            for i in range(min(10, len(self.tickers)-1)):  # Create 20 pair strategies
                ticker1 = self.tickers[i]
                ticker2 = self.tickers[i+1] if i+1 < len(self.tickers) else self.tickers[0]
                
                # Long/Short pair strategy
                correlation_strategies.extend([
                    {
                        'name': f'LongShort_{ticker1}_{ticker2}',
                        'ticker': ticker1,  # Primary ticker
                        'secondary_ticker': ticker2,
                        'buy': lambda d: d['RSI'] < 40,  # Simple condition for primary
                        'sell': lambda d: d['RSI'] > 60
                    },
                    {
                        'name': f'Correlation_{ticker1}_{ticker2}',
                        'ticker': ticker1,
                        'secondary_ticker': ticker2,
                        'buy': lambda d: (d['5EMA'] > d['20EMA']) & (d['MACD'] > d['MACDSIGNAL']),
                        'sell': lambda d: (d['5EMA'] < d['20EMA']) | (d['MACD'] < d['MACDSIGNAL'])
                    }
                ])

        # 4. REGIME-BASED STRATEGIES (20 strategies)
        # Use volatility and trend strength to define regimes
        regime_strategies = []
        regimes = [
            {'name': 'LowVol', 'condition': lambda d: d['ADX'] < 20},
            {'name': 'HighVol', 'condition': lambda d: d['ADX'] > 30},
            {'name': 'Trending', 'condition': lambda d: (d['ADX'] > 25) & (d['5EMA'] > d['20EMA'])},
            {'name': 'Ranging', 'condition': lambda d: (d['ADX'] < 20) & (d['RSI'] > 30) & (d['RSI'] < 70)}
        ]
        
        for regime in regimes:
            for ticker in self.tickers[:5]:  # Use 5 tickers per regime
                strategy = {
                    'name': f'{regime["name"]}_{ticker}',
                    'ticker': ticker,
                    'regime': regime,
                    'buy': lambda d, r=regime: r['condition'](d) & (d['RSI'] > 35),
                    'sell': lambda d, r=regime: ~r['condition'](d) | (d['RSI'] < 65)
                }
                regime_strategies.append(strategy)
                if len(regime_strategies) >= 20:
                    break
            if len(regime_strategies) >= 20:
                break

        # 5. MIXED INDICATOR STRATEGIES (20 strategies)
        mixed_strategies = []
        indicator_combinations = [
            {'buy': lambda d: (d['MACD'] > d['MACDSIGNAL']) & (d['RSI'] > 50) & (d['MFI'] < 80),
             'sell': lambda d: (d['MACD'] < d['MACDSIGNAL']) | (d['RSI'] < 50) | (d['MFI'] > 80)},
            {'buy': lambda d: (d['close'] > d['SAR']) & (d['WILLR'] > -80) & (d['CCI'] > -100),
             'sell': lambda d: (d['close'] < d['SAR']) | (d['WILLR'] < -80) | (d['CCI'] < -100)},
            {'buy': lambda d: (d['SLOWK'] > d['SLOWD']) & (d['MOM'] > 0) & (d['OBV'] > d['OBV'].shift(1)),
             'sell': lambda d: (d['SLOWK'] < d['SLOWD']) | (d['MOM'] < 0) | (d['OBV'] < d['OBV'].shift(1))},
        ]
        
        for i, combo in enumerate(indicator_combinations):
            for j, ticker in enumerate(self.tickers[:7]):  # Mix across tickers
                strategy = {
                    'name': f'Mixed{i}_{ticker}',
                    'ticker': ticker,
                    'buy': combo['buy'],
                    'sell': combo['sell']
                }
                mixed_strategies.append(strategy)
                if len(mixed_strategies) >= 20:
                    break
            if len(mixed_strategies) >= 20:
                break

        # Combine all strategies
        all_strategies = (asset_specific_strategies + timeframe_strategies + 
                         correlation_strategies + regime_strategies + mixed_strategies)
        
        # Limit to 70 strategies
    
        return all_strategies

    def generate_diversified_returns(self, is_training=True):
        """Generate returns for all diversified strategies"""
        strategies = self.create_diversified_strategies()
        #data_dict = self.train_data if is_training else self.test_data
        data_dict = self.all_data
        strategy_returns = {}
        # test 
        test_strategy_returns = {}
        
        for strategy in strategies:
            ticker = strategy['ticker']
            if ticker not in data_dict:
                continue
                
            try:
                data = data_dict[ticker].copy()
                processed_data = self.create_indicators(data)
                
                buy_condition = strategy['buy'](processed_data)
                sell_condition = strategy['sell'](processed_data)
                test_processed_data = processed_data[processed_data['Date'].dt.year >= self.test_period]
                test_buy_condition = strategy['buy'](test_processed_data)
                test_sell_condition = strategy['sell'](test_processed_data)
                test_sharpe, test_returns = self.evaluate_strategy(test_processed_data, test_buy_condition, test_sell_condition)
                sharpe, returns = self.evaluate_strategy(processed_data, buy_condition, sell_condition)
                
                if len(returns) and len(test_returns)  > 0:
                    strategy_returns[strategy['name']] = returns
                    test_strategy_returns[strategy['name']] = test_returns
                    self.strategy_performance[strategy['name']] = {
                        'sharpe': sharpe,
                        'ticker': ticker,
                        'strategy_type': strategy.get('regime', {}).get('name', 'general')
                    }
                    self.strategy_names.append(strategy['name'])
                    
            except Exception as e:
                print(f"Error with strategy {strategy['name']}: {e}")
                continue
        train_data = pd.DataFrame(strategy_returns)
        test_data = pd.DataFrame(test_strategy_returns)
        
        return train_data, test_data

    def analyze_diversity(self, returns_df):
        """Analyze the diversity of the strategy returns"""
        if returns_df.empty:
            return
            
        correlation_matrix = returns_df.corr()
        
        # Print diversity statistics
        print(f"\n=== DIVERSITY ANALYSIS ===")
        print(f"Number of strategies: {len(returns_df.columns)}")
        print(f"Average correlation: {correlation_matrix.mean().mean():.3f}")
        print(f"Min correlation: {correlation_matrix.min().min():.3f}")
        print(f"Max correlation: {correlation_matrix.max().max():.3f}")
        print(f"Std of correlations: {correlation_matrix.std().std():.3f}")
        
        # Show strategy performance distribution
        sharpe_ratios = [self.strategy_performance[col]['sharpe'] for col in returns_df.columns 
                        if col in self.strategy_performance]
        print(f"\nSharpe ratio distribution:")
        print(f"Mean: {np.mean(sharpe_ratios):.3f}")
        print(f"Std: {np.std(sharpe_ratios):.3f}")
        print(f"Min: {np.min(sharpe_ratios):.3f}")
        print(f"Max: {np.max(sharpe_ratios):.3f}")

# Example usage
def create_diversified_dataset():
    """Create a diversified trading strategy dataset"""
    
    # Use a mix of asset classes for true diversification
    tickers = [
        # Large cap stocks
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        # Different sectors
        'JPM', 'JNJ', 'PG', 'XOM', 'WMT',
        # ETFs for broader exposure
        'SPY', 'QQQ', 'IWM', 'EFA', 'GLD'
    ]
    
    diversified_system = DiversifiedTradingStrategies(
        tickers=tickers,
        start_date='2015-01-01',
        end_date='2023-12-31',
        test_period=2020
    )
    
    # Generate training returns
    print("Generating diversified training strategies...")
    train_returns = diversified_system.generate_diversified_returns(is_training=True)
    
    # Generate test returns
    print("Generating diversified test strategies...")
    test_returns = diversified_system.generate_diversified_returns(is_training=False)
    
    # Analyze diversity
    diversified_system.analyze_diversity(train_returns)
    
    return diversified_system, train_returns, test_returns

# Run the example
#if __name__ == "__main__":
#    system, train_data, test_data = create_diversified_dataset()
 #   print(f"\nFinal dataset shapes:")
  #  print(f"Training: {train_data.shape}")
   # print(f"Testing: {test_data.shape}")