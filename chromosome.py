import numpy as np
import random
import itertools
import heapq
import math
import talib as ta
import pandas as pd
import multiprocess as mp
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
        normalised_total = 0
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
    
    def generate_trading_signal(self,data,strategy,stop_loss,take_profit):
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
                    if trade_return <= stop_loss or trade_return >= take_profit: 
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'   
                        
                    if row[data.columns.get_loc(strategy)] == 0 and trade_return >= take_profit :
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'
        return monthly_returns
    
    def monthly_returns(self,data):
        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp()
        monthly_returns = {}
        for strategy in self.strategies:
            monthly_returns[strategy] =  self.generate_trading_signal(data,strategy,chromomosome_stop_loss,chromomosome_take_profit)
        monthly_returns = pd.DataFrame.from_dict(monthly_returns)
        return monthly_returns

    def calculate_chromosome_fitness(self,data,allocated_capital):
        monthly_returns = self.monthly_returns(data)
        self.mdd= self.getMDD(monthly_returns)
        self.profit =self.getProfit(monthly_returns,allocated_capital)
        #self.corr = self.getCorrelation(monthly_returns)
        self.gb = self.groupBalance()
        if self.mdd > 0.01:
            fitness = (self.profit / self.mdd)  + self.gb 
        else:
            fitness = self.profit + self.gb 
        self.fitness_value = fitness
        
    
    
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
            child1.weight_part =[round(w,2) for w in weights]

        

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
        
    
    
    