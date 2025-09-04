import numpy as np
import random
import itertools
import heapq
import math
import talib as ta
import pandas as pd
import multiprocess as mp
import itertools
from typing import List, Tuple, Optional
from functools import lru_cache
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy



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

class Chromosome():
    

    def __init__(self,K,n,b,strategies,num_weight,stop_loss,take_profit):
        self.K = K
        self.n = n
        self.b = b
        self.num_weight = num_weight
        self.strategies = strategies
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.fitness_value = 0
        self.wb=0
        self.profit=0
        self.corr=0
        self.gb=0
        self.mdd = 0
        self.sltp_part = []
        self.group_part = []
        self.weight_part = []
        self.weights=[]
        self.bbb = []
        self.tsps = pd.DataFrame()
        """Initialize with caching support"""
        self._fitness_cache = {}
        self._cache_size = 256
        self._cache_hits = 0
        self._cache_misses = 0

    def __str__(self):
        return f"SLTP: {self.sltp_part}\nGROUP: {self.group_part}\nWEIGHT: {self.weight_part}\nFITNESS: {  self.fitness_value}"
     # init population    
    
    def generateSLTP(self):
        """Generate bits for SLTP Part"""
        l = self.n + self.b
        sltp= [random.randint(0, 1) for _ in range(l)]
        self.sltp_part = sltp
        
    
    def generateGroup(self):
        """Generate Group and assign TS to groups K"""
        x = self.strategies.copy()
        random.shuffle(x)
        groups =np.array_split(x,self.K)
        groups = [group.tolist() for group in groups]
        self.group_part = groups
       

    def generateWeight(self):
        """Generate Weight Part and assign to groups K"""
        weights= [random.random() for _ in range(self.K)]
        if sum(weights) != 1.0:
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights]
        self.weight_part = [round(w,2) for w in weights]
        
    
    
    def create_chromosome(self):
        #init a single chromosome
        self.generateSLTP()
        self.generateGroup()
        self.generateWeight()
        return self
    
    def decode_weights(self):
        weight_part_length = self.K * self.num_weight
        weights = []
        for i in range(0, weight_part_length, self.num_weight):
            gene = self.weight_part[i:i + self.num_weight]
            weight = sum(b * 2**idx for idx, b in enumerate(reversed(gene))) / (2 ** self.num_weight - 1)
            weights.append(weight)
        # Convert to NumPy array and ensure weights sum to 1
        if sum(weights) == 0:
            weights = np.full(self.K, 0.125)
        weights = np.array(weights) / sum(weights) 
        rounded_weights = np.round(weights, 3)
        return rounded_weights
        

    #Fitness Function
        
    def get_max_min(self,p4mc):
        modified ={}
        for i in  p4mc:
            modified[i] = p4mc[i]
        max_key= max(modified, key=modified.get)
        min_key = min(modified, key=modified.get)
        
        return modified[max_key], modified[min_key]


    def normalisation(self,p4mc):
        max_value,min_value = self.get_max_min(p4mc)
        for i in p4mc:
            p4mc[i] = (p4mc[i] - min_value )/ (max_value - min_value)
        return p4mc


    def getWeights1(self):
        weights = self.weight_part.copy()
        K = self.K+1
        group_weight = {k: [] for k in range(K)}
        for i in range(K):
            while weights:
                    x = weights.pop(0)
                    if x == 0:
                        break
                    else:
                        group_weight[i].append(x)
        total_1s = sum([i for i in weights if i == 1])
        for i in group_weight:
            try:
                group_weight[i]= round(len(group_weight[i])/total_1s,2)
            except ZeroDivisionError:
                group_weight[i] = 0
        if all(value == 0 for value in group_weight.values()) or group_weight[0] == 1 :
            for i in group_weight:
                percent = round(1/(self.K+1),2)
                group_weight[i] = percent
        return group_weight
    
    def getWeights(self):
            weights = self.weight_part.copy()
            K = self.K +1
            group_weight = {k: [] for k in range(K)}
            
            # Divide the weights into K groups
            for i in range(K):
                while weights:
                    x = weights.pop(0)
                    if x == 0:
                        break
                    else:
                        group_weight[i].append(x)
            
            total_1s = sum([i for i in self.weight_part if i == 1])
            for i in group_weight:
                try:
                    group_weight[i] = round(len(group_weight[i]) / total_1s, 2)
                except ZeroDivisionError:
                    group_weight[i] = 0
            
            if all(value == 0 for value in group_weight.values()) or group_weight[0] == 1:
                percent = round(1 / (K), 2)
                for i in group_weight:
                    group_weight[i] = percent
            
            return group_weight

      
    
    def getProfit(self, ts_data, allocated_capital):
        weights = self.weight_part
        total = 0
        # Calculate profits for each trading strategy
        ts_profits = (1 + ts_data).cumprod().iloc[-1] -1
        for i, group in enumerate(self.group_part):
            if len(group) !=0:
                if weights[i] != 0:
                    group_profits = ts_profits[group]
                    # Calculate contributions for the original values
                    contribution = group_profits * weights[i] * allocated_capital
                    total += contribution.mean()     
        return total

    

    
    def getMDD(self,ts_data):
        total = 0
        ts_mdd = {}
        weights = self.weight_part
        for i in range(len(self.group_part)):
            group_mdd = []
            weight = weights[i]
            for j in self.group_part[i]:
                # Calculate the cumulative returns for each asset at the end of each month
                cumulative_returns = (1+ ts_data[j]).cumprod()
                # Calculate the cumulative maximum value for each asset at the end of each month
                cumulative_max = cumulative_returns.cummax()
                # Calculate the drawdown for each asset at the end of each month
                drawdown = cumulative_max - cumulative_returns
                # Calculate the maximum drawdown for each asset
                max_drawdown = drawdown.max()
                ts_mdd[j] = max_drawdown
                group_mdd.append(max_drawdown)
            if group_mdd:
                contribution = np.mean(group_mdd) * weight
                total+= contribution
        mdd = total
        return mdd 
    
        
    def getCorrelation(self,monthly_returns):
        weights = np.array(self.weight_part)
        group_returns = pd.DataFrame()
        for i, group in enumerate(self.group_part):
             returns = monthly_returns[group].mean(axis=1)
             group_returns[f"group {i+1}"] = returns
        covariance_matrix = group_returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)
        
        return portfolio_std_dev

    
    def groupBalance(self):
        N = len(self.strategies)
        num_groups = len(self.group_part)
        gb = 0
        for group in self.group_part:
            g_size = len(group)
            g_result = g_size / N if N > 0 else 0
            g = -g_result * np.log(g_result) if g_result > 0 else 0
            gb += g

        max_gb = -np.log(1/num_groups) * num_groups
        normalized_gb = gb / max_gb

        return normalized_gb
    

    def group_dispersion(self):
        group_counts = {}

        # Count the number of strategies in each group for the chromosome
        for i ,group in enumerate(self.group_part):
            group_counts[i] = len(group)
        group_counts = pd.DataFrame(group_counts)
        # Calculate the variance of the group sizes
        variance = group_counts.var()

        # Normalize the variance by the maximum possible variance
        
        dispersion_factor = variance 

        return dispersion_factor


    
    def binary_to_sltp(self):
        """CONVERT SLPT Part to float"""
        
        sl_part,tp_part = self.sltp_part[:self.n],self.sltp_part[self.n:]
        max_sl = sum([np.power(2,i)*1 for i in range(len(sl_part))])
        max_tp = sum([np.power(2,i)*1 for i in range(len(tp_part))])
        sl_part.reverse()
        tp_part.reverse()
        sl  = sum([np.power(2,i)*sl_part[i] for i in range(len(sl_part))])
        tp  = sum([np.power(2,i)*tp_part[i] for i in range(len(tp_part))])
        sl = self.stop_loss/max_sl * sl
        tp = self.take_profit/max_tp * tp
        return sl,tp
    
    
    
  
    

    
    
    def calculate_profit(self,portfolio,monthly_returns):
        """Calculate the profit for a Portfolio (TSP)"""

        weights =  self.decode_weights()
        cumulative_returns =  (1 + monthly_returns[list(portfolio)]).cumprod().iloc[-1]-1
        final_portfolio_return = (cumulative_returns * weights).sum()
    
        return final_portfolio_return
    
    def calculate_mdd(self,portfolio,monthly_returns):
        """Calculate the mdd for a Portfolio (TSP)"""
        weighted_returns = pd.DataFrame()
        weights =  self.decode_weights()
        for i, strategy in enumerate(portfolio):
            weighted_returns[strategy] = monthly_returns[strategy]* weights[i]
        # Calculate the combined weighted returns for the portfolio
        portfolio_returns = weighted_returns.sum(axis=1) 
        cumulative_returns = (1+ portfolio_returns ).cumprod()
        # Calculate the cumulative maximum value for each asset at the end of each month
        previous_peaks = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - previous_peaks) / (1 + previous_peaks)
        mdd = drawdowns.min()
        return mdd


    def calculate_chromosome_fitness11(self, monthly_returns, allocated_capital):
        """
        Fast chromosome-specific fitness calculation
        """
        # Generate portfolios from THIS chromosome's grouping
        portfolios = list(itertools.product(*self.group_part))
        
        # Quick evaluation - sample max 20 portfolios for speed
        sample_size = min(20, len(portfolios))
        if len(portfolios) > sample_size:
            sampled_indices = np.random.choice(len(portfolios), sample_size, replace=False)
            sampled_portfolios = [portfolios[i] for i in sampled_indices]
        else:
            sampled_portfolios = portfolios
            
        # Calculate metrics for sampled portfolios
        profits = [self.calculate_profit(p, monthly_returns) for p in sampled_portfolios]
        mdds = [self.calculate_mdd(p, monthly_returns) for p in sampled_portfolios]
        
        # Create tsps DataFrame
        self.tsps = pd.DataFrame({
            'Portfolio': sampled_portfolios,
            'Profit': profits,
            'MDD': mdds
        })
        
        # FIXED AGGREGATION - Make it chromosome-specific
        self.mdd = self.get_gstp_mdd_fixed(self.tsps)

        self.profit = self.get_gstp_profit_fixed(self.tsps)
        self.gb = self.groupBalance()
        
        # Fitness calculation
        if self.mdd > 0.01:
            fitness = self.profit / self.mdd
        else:
            fitness = self.profit
            
        self.fitness_value = fitness
        return fitness

    def get_gstp_mdd_fixed(self, tsps):
        """
        FIXED: Use best portfolio's MDD instead of mean
        This makes each chromosome unique!
        """
        if tsps.empty:
            return 0.01
            
        # Method 1: Use MDD of the best performing portfolio
        best_portfolio_idx = tsps['Profit'].idxmax()
        mdd = tsps.loc[best_portfolio_idx, 'MDD']
        
        # Method 2: Use worst-case MDD (most conservative)
        # mdd = tsps['MDD'].max()
        
        # Method 3: Weighted average (balance performance and risk)
        # weights = tsps['Profit'] / tsps['Profit'].sum()
        # mdd = (tsps['MDD'] * weights).sum()
        
        return mdd
    
    def get_gstp_profit_fixed(self, tsps):
        """
        FIXED: Use best portfolio's profit instead of mean
        This makes each chromosome unique!
        """
        if tsps.empty:
            return 0
            
        # Method 1: Use profit of the best performing portfolio
        profit = tsps['Profit'].max()
        
        # Method 2: Use top quartile average
        # top_quartile = int(len(tsps) * 0.25) + 1
        # profit = tsps.nlargest(top_quartile, 'Profit')['Profit'].mean()
        
        # Method 3: Risk-adjusted selection
        # tsps['SharpeRatio'] = tsps['Profit'] / tsps['MDD']
        # best_sharpe_idx = tsps['SharpeRatio'].idxmax()
        # profit = tsps.loc[best_sharpe_idx, 'Profit']
        
        return profit


        
    
    def scale_fitness(self,max_fitness,min_fitness):
        # Linear scalin
        #min_fitness = min(population_fitness)
        #max_fitness = max(population_fitness)
        shift = -min_fitness
        self.fitness_value += shift
        desired_min_fitness = 0.1
        desired_max_fitness = 1
        a = (desired_max_fitness - desired_min_fitness) / (max_fitness - min_fitness)
        b = desired_min_fitness - a * min_fitness
        fitness_range = max_fitness - min_fitness
        scaled_fitness = (self.fitness_value - min_fitness) / fitness_range
        #scaled_fitness = a * scaled_fitness + b # a and b are slope and intercept of the linear scaling equation

        self.fitness_value = scaled_fitness

    
    
     ##### Genetic operators
    def hamming_distance(self,chromosome2): 	
        dist_counter = 0 
        #sltp part
        for n in range(len(self.sltp_part)):
            if self.sltp_part[n] != chromosome2.sltp_part[n]:
                dist_counter += 1
        #grouping part
        for n in range(len(self.group_part)):
            if self.group_part[n] != chromosome2.group_part[n]:
                dist_counter += 1
        #weight part
        for n in range(len(self.weight_part)):
            if self.weight_part[n] != chromosome2.weight_part[n]:
                dist_counter += 1
        return dist_counter
    
    ### one point Crossover
    def crossover(self,parent2,r_cross):
        child1 = deepcopy(self)
        child2 = deepcopy(parent2)
        # check for recombination
        if random.random() < r_cross:
            # select crossover point that is not on the end of the string
            index = random.randint(1, len(self.sltp_part)-2)
            # perform crossover on SLTP
            child1.sltp_part = self.sltp_part[index:]+ parent2.sltp_part[:index] 
            child2.sltp_part = parent2.sltp_part[index:] + self.sltp_part[:index] 
        if random.random() < r_cross:
            # perform crossover on weight
            #index = random.randint(1, len(self.weight_part)-2)
            #child1.weight_part = self.weight_part[index:] + parent2.weight_part[:index] 
            #if child1.weight_part.count(0) != self.K+1:
            #    child1.generateWeight()
            #child2.weight_part = parent2.weight_part[index:] + self.weight_part[:index] 
            #if child2.weight_part.count(0) != self.K+1:
            #    child2.generateWeight()

            #### real valued
            weights = [(w1 + w2) / 2.0 for w1, w2 in zip(child1.weight_part, child2.weight_part)]
            if sum(weights) != 1.0:
                # Apply normalization to ensure sum of weights is 1
                sum_weights = sum(weights)
                weights = [w / sum_weights for w in weights]
            child1.weight_part = [round(w,2) for w in weights]
            child2.weight_part =[round(w,2) for w in weights]

        

        return child1,child2
    

    ###Mutation

    def mutation(self,r_mut):
        # check for a mutation
        
            # on SLTP
        for i in range(len(self.sltp_part)):
            if random.random() < r_mut:
            # flip the bit
                self.sltp_part[i] = 1 - self.sltp_part[i]

        # on Weight Part
        #for i in range(len(self.weight_part)):
         #   if random.random() < r_mut:
        #    # flip the bit 
         #       self.weight_part[i] = 1 - self.weight_part[i] 
        #    #check if condition is still maintained
         #       if self.weight_part.count(0) != self.K+1:
         #           self.generateWeight() 
        mutated_weights = self.weight_part.copy()
        for i,weight in enumerate(self.weight_part):
            if random.random() < r_mut:
                perturbation = random.gauss(0, 0.01)
                weight += perturbation
                if weight < 0:
                    weight = 0
                mutated_weights[i]= round(weight,2)
        # Apply normalization to ensure sum of weights is 1
        if sum(mutated_weights) != 1.0:
            sum_weights = sum(mutated_weights)
            mutated_weights = [w / sum_weights for w in mutated_weights]
        self.weight_part = [round(w,2) for w in mutated_weights]

        # on TS part
        if random.random() < r_mut:
            i = random.randrange(self.K)
            left_group_index = (i - 1) % self.K
            right_group_index = (i + 1) % self.K
            if len(self.group_part[left_group_index]) > 0:
                ts_idx = random.randrange(len(self.group_part[left_group_index]))
                ts = self.group_part[left_group_index][ts_idx]
                self.group_part[right_group_index].append(ts)
                self.group_part[left_group_index].remove(ts)
            
                
    ###Inversion
    def inversion(self,r_inv):
        if random.random() < r_inv:
            self.group_part = self.group_part[::-1]
        
    
    #### Optimized Fitness with Caching
    
        
    def calculate_chromosome_fitness(self, monthly_returns, is_training=True):
        """
        Optimized fitness calculation with caching
        """
        # CACHING: Check if we've seen this chromosome before
        chromosome_hash = self._hash_chromosome()
        
        if chromosome_hash in self._fitness_cache and is_training:
            self._cache_hits += 1
            cached_result = self._fitness_cache[chromosome_hash]
            self.fitness_value = cached_result['fitness']
            self.profit = cached_result['profit'] 
            self.mdd = cached_result['mdd']
            self.gb = cached_result['gb']
            # Don't recreate tsps DataFrame - it's expensive and often unused
            return self.fitness_value
        if is_training:
            self._cache_misses += 1
        
        # MAJOR ISSUE 1: You're generating ALL possible portfolios with itertools.product
        # This explodes exponentially! For groups of size [10,10,10] = 1000 portfolios
        # For [20,15,12,8] = 28,800 portfolios!
        
        # SOLUTION 1: Don't generate all portfolios - use smarter sampling
        total_combinations = 1
        for group in self.group_part:
            total_combinations *= len(group)
        
        # If combination space is huge, sample strategically instead
        if total_combinations > 500:  # Adjust threshold as needed
            portfolios = self._strategic_sample_portfolios(max_samples=50)
        else:
            # Only generate all portfolios for small spaces
            portfolios = list(itertools.product(*self.group_part))
            
        # MAJOR ISSUE 2: Random sampling on large lists is inefficient
        # Your current: np.random.choice(len(portfolios), sample_size, replace=False)
        
        # SOLUTION 2: Use systematic sampling for better coverage
        sample_size = min(20, len(portfolios))
        if len(portfolios) > sample_size:
            # Systematic sampling - much faster and better coverage
            step = len(portfolios) // sample_size
            indices = [i * step for i in range(sample_size)]
            sampled_portfolios = [portfolios[i] for i in indices]
        else:
            sampled_portfolios = portfolios
        
        # MAJOR ISSUE 3: Creating DataFrame for each fitness call is expensive
        # SOLUTION 3: Pre-calculate weights once, use numpy operations
        
        weights = self.decode_weights()  # Calculate once
        
        # Vectorized profit/MDD calculation
        profits = []
        mdds = []
        
        for portfolio in sampled_portfolios:
            # Optimized profit - avoid intermediate DataFrames
            portfolio_data = monthly_returns[list(portfolio)]
            cumulative_returns = (1 + portfolio_data).cumprod().iloc[-1] - 1
            profit = (cumulative_returns * weights).sum()
            profits.append(profit)
            
            # Optimized MDD - vectorized operations
            weighted_returns = portfolio_data * weights  # Broadcasting
            portfolio_returns = weighted_returns.sum(axis=1)
            cumulative = (1 + portfolio_returns).cumprod()
            peaks = cumulative.cummax()
            drawdowns = (cumulative - peaks) / (1 + peaks)
            mdd = drawdowns.min()
            mdds.append(mdd)
        
        # MAJOR ISSUE 4: Creating DataFrame just for aggregation is wasteful
        # SOLUTION 4: Direct numpy aggregation
        
        profits = np.array(profits)
        mdds = np.array(mdds)
        
        # Simple aggregation - adjust as needed
        self.mdd = np.mean(mdds)
        self.profit = np.mean(profits) 
        self.gb = self.groupBalance()
        
        # Fitness calculation
        if self.mdd > 0.01:
            fitness = self.profit / self.mdd
        else:
            fitness = self.profit
            
        self.fitness_value = fitness
        
        # CACHE the result
        self._cache_result(chromosome_hash, fitness, self.profit, self.mdd, self.gb)
        
        return fitness

    def _hash_chromosome(self):
        """Create a hash of the chromosome for caching"""
        try:
            # Hash the group_part structure
            if hasattr(self, 'group_part'):
                # Convert to hashable tuple format
                hashable_groups = []
                for group in self.group_part:
                    if hasattr(group, '__iter__') and not isinstance(group, str):
                        hashable_groups.append(tuple(sorted(group)))
                    else:
                        hashable_groups.append(group)
                
                chromosome_repr = tuple(hashable_groups)
                return hash(chromosome_repr)
        except (TypeError, AttributeError):
            pass
        
        # Fallback: use object id (less cache-friendly but safe)
        return id(self.group_part) if hasattr(self, 'group_part') else id(self)

    def _cache_result(self, chromosome_hash, fitness, profit, mdd, gb):
        """Cache the fitness result"""
        # Store result
        self._fitness_cache[chromosome_hash] = {
            'fitness': fitness,
            'profit': profit, 
            'mdd': mdd,
            'gb': gb
        }
        
        # Manage cache size - remove oldest entries if too large
        if len(self._fitness_cache) > self._cache_size:
            # Remove oldest 25% of entries (simple FIFO)
            items_to_remove = len(self._fitness_cache) - int(self._cache_size * 0.75)
            keys_to_remove = list(self._fitness_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                self._fitness_cache.pop(key, None)

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_size': len(self._fitness_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses, 
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def clear_cache(self):
        """Clear the fitness cache"""
        self._fitness_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _strategic_sample_portfolios(self, max_samples=50):
        """
        Sample portfolios strategically without generating full cartesian product
        """
        portfolios = []
        seen = set()
        
        # Strategy 1: Random sampling
        for _ in range(max_samples // 2):
            portfolio = tuple(np.random.choice(group) for group in self.group_part)
            if portfolio not in seen:
                portfolios.append(portfolio)
                seen.add(portfolio)
        
        # Strategy 2: Edge cases (first/last elements from each group)
        import itertools
        edges = [[group[0], group[-1]] if len(group) > 1 else [group[0]] 
                for group in self.group_part]
        
        for portfolio in itertools.product(*edges):
            if len(portfolios) >= max_samples:
                break
            if portfolio not in seen:
                portfolios.append(portfolio)
                seen.add(portfolio)
        
        # Strategy 3: Fill remaining with pure random
        while len(portfolios) < max_samples:
            portfolio = tuple(np.random.choice(group) for group in self.group_part)
            if portfolio not in seen:
                portfolios.append(portfolio)
                seen.add(portfolio)
        
        return portfolios