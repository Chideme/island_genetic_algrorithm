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
from island_ga import IslandGGA 

from chromosome import Chromosome 
from diversified_strategies import DiversifiedTradingStrategies
from single_data_processing import SingleAssetTI

import itertools
import yfinance as yf
import datetime
import seaborn as sns 

import riskfolio as rp

from pso import PortfolioPSO

import sota_models




def model_results(model,current_train, current_test, strategies, pSize, num_iter, K, m_iter, num_islands):
    if model != 'pso':
        gtsp = IslandGGA(
                data=current_train,
                K=2,
                num_islands=5,
                m_iter=5,
                num_iter=num_iter,
                pSize=pSize,
                strategies=strategies,
                evolve_strategy=model)
        gtsp.evolve()

        # Record training results
       
        train_results = (gtsp.globalBest.fitness_value, gtsp.globalBest.profit, gtsp.globalBest.mdd)

        # Evaluate on test data
        gtsp.globalBest.calculate_chromosome_fitness(current_test, allocated_capital=1)
        test_results = (gtsp.globalBest.fitness_value, gtsp.globalBest.profit, gtsp.globalBest.mdd)

        return train_results, test_results
    

    elif model == 'pso':
        pso= PortfolioPSO(current_train,num_particles=pSize,iterations=num_iter)
        best_portfolio, weights, _ = pso.run()
        train_returns = current_train[list(best_portfolio)]
        train_results = sota_models.portfolio_metrics(best_portfolio,train_returns,weights)
        returns = current_test[list(best_portfolio)]
        returns = returns.dropna()
        test_results = sota_models.portfolio_metrics(best_portfolio,returns,weights)
        
        return  train_results, test_results
        


    


class ModelComparator:
    def __init__(self, stock_tickers,period,start_date, end_date, pSize, K,num_iter, m_iter, num_islands,num_runs,
                optimization_approaches = ["ring", "multikuti", "master_slave", "gga", "pso"],diversified=True):
        self.diversified = diversified
        self.stock_tickers = stock_tickers if self.diversified else stock_tickers[0]
        self.start_date = start_date
        self.end_date = end_date
        self.num_runs = num_runs
        self.pSize = pSize
        self.num_iter = num_iter
        self.optimization_approaches = optimization_approaches
        self.period = period
        self.K = K
        self.m_iter = m_iter
        self.num_islands = num_islands

    def run_comparison(self):

        results = []
        # Convert dates to datetime if they're strings
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date)
        if self.diversified:
            diversified_system = DiversifiedTradingStrategies(
            tickers=self.stock_tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            test_period=self.period
        )
    
            # Generate training returns
            print("Generating diversified training and tesing strategies...")
            train_data, val_data = diversified_system.generate_diversified_returns(is_training=True)
            strategies = diversified_system.strategy_names
           
        else:
            stock_ticker = self.stock_tickers[0]
            data = SingleAssetTI(stock_ticker,self.start_date,self.end_date,self.period)
            data.data_preprocess()
            train_data = data.train_data    
            val_data = data.test_data
            train_data = train_data.dropna()
            val_data = val_data.dropna()
            strategies = data.strategies
        
        for run in range(self.num_runs):
            for model in self.optimization_approaches:
                print("Running model: ", model)
                print("Run: ", run+1, " out of ", self.num_runs)
                # model_results(model,current_train, current_test, strategies, pSize, num_iter, K, m_iter, num_islands)
                train_metrics, test_metrics = model_results(model, train_data, val_data, strategies, self.pSize, self.num_iter, self.K, self.m_iter, self.num_islands)  
                
               
                train_fitness = train_metrics[0]
                train_returns = train_metrics[1]
                train_mdd = train_metrics[2]

                # # store test metrics
                test_fitness = test_metrics[0]
                test_returns = test_metrics[1]
                test_mdd = test_metrics[2]



                    
        
                # Store training results
                results.append({
                    
                    'run': run,
                    'model': model,
                    'phase': 'train',
                    'fitness': train_fitness,
                    'returns': train_returns,
                    'mdd': train_mdd
                })
                
                # Store validation results
                results.append({
                    
                    'run': run,
                    'model': model,
                    'phase': 'validation',
                    'fitness': test_fitness,
                    'returns': test_returns,
                    'mdd': test_mdd
                
                })
        
        return pd.DataFrame(results)
    
    def plot_results(self, results_df, optimization_approaches):
        # Only plot 'returns' and 'mdd'
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metrics = ['returns', 'mdd']
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            filtered_df = results_df[
                (results_df['phase'] == 'validation') & 
                (results_df['model'].isin(optimization_approaches))
            ]
            sns.boxplot(
                data=filtered_df,
                x='model',
                y=metric,
                ax=ax
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        plt.tight_layout()
        plt.show()

# Usage
#comparator = ModelComparator(returns=train_returns,test_returns=test_returns, strategies=strategies)
#results_df = comparator.run_comparison()
#comparator.plot_results(results_df)
#results_df.to_csv('model_comparison_results.csv')