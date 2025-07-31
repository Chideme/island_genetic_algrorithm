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

from single_data_processing import SingleAssetTI

import itertools
import yfinance as yf
import datetime
import seaborn as sns 

import riskfolio as rp

from pso import PortfolioPSO

import sota_models




def model_results(model,current_train, current_test, strategies, pSize=150, num_iter=50):
    if model != 'pso':
        gtsp = IslandGGA(
                data=current_train,
                K=8,
                num_islands=10,
                m_iter=10,
                num_iter=50,
                pSize=150,
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
    def __init__(self, stock_ticker,start_date, end_date, pSize,num_iter, num_runs=6,
                optimization_approaches = ["ring", "multikuti", "master_slave", "gga", "pso"]):
        self.stock_ticker = stock_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.num_runs = num_runs
        self.pSize = pSize
        self.num_iter = num_iter
        self.optimization_approaches = optimization_approaches

    def run_comparison(self):

        results = []
        # Convert dates to datetime if they're strings
        if isinstance(self.start_date, str):
            self.start_date = pd.to_datetime(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = pd.to_datetime(self.end_date)
    
        data = SingleAssetTI(self.stock_ticker,self.start_date,self.end_date,2019)
        data.data_preprocess()
        train_data = data.train_data    
        val_data = data.test_data
        train_data = train_data.dropna()
        val_data = val_data.dropna()
        strategies = data.strategies
    
        for run in range(self.num_runs):
            for model in self.optimization_approaches:
                print("Running model: ", model)
                # model_results(model,current_train, current_test, strategies, pSize=10, num_iter=2)
                train_metrics, test_metrics = model_results(model, train_data, val_data, strategies, self.pSize, self.num_iter)
                
               
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