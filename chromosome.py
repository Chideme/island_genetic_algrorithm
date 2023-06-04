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
        weights= [1 for _ in range(self.num_weight)]
        K = self.K
        for i in range(K):
            random_index = random.randrange(K)
            weights[random_index] = 0
        
        self.weight_part = weights
        
    
    
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

      
        

    def getProfit111(self,ts_data,allocated_capital):
        weights = self.getWeights()
        self.weights = weights
        total = 0
        ts_profits = {}
        for i in range(len(self.group_part)):
            group_contribution = 0
            for j in self.group_part[i]:
                cumulative_returns = (1 + ts_data[j]).cumprod()
                profit = cumulative_returns.iloc[-1] - 1
                ts_profits[j]=profit
                contribution = profit*weights[i+1]*allocated_capital 
                group_contribution += contribution 
            try:   
                total+=group_contribution/len(self.group_part[i])  
            except:
                total+= 0
           
        normalised_total = 0 
        ts_profits = self.normalisation(ts_profits)
        for i in range(len(self.group_part)):
            group_contribution = 0
            for j in self.group_part[i]:
                if weights[i+1] !=0:
                    normalised_profit = ts_profits[j]
                    normalised_contribution = normalised_profit*weights[i+1]*allocated_capital
                    group_contribution +=normalised_contribution
            try:
                normalised_total += group_contribution/len(self.group_part[i])  
            except:
                normalised_total +=0
        
        return total,normalised_total
    
    def getProfit(self, ts_data, allocated_capital):
        weights = self.getWeights()
        total = 0
        normalised_total = 0
        
        # Calculate profits for each trading strategy
        ts_profits = (1 + ts_data).cumprod().iloc[-1] - 1
        normalized_profits = (ts_profits - ts_profits.min()) / (ts_profits.max() - ts_profits.min())
        
        for i, group in enumerate(self.group_part):
            if len(group) !=0:
                if weights[i+1] != 0:
                    group_profits = ts_profits[group]
                    
                    # Calculate contributions for the original values
                    contribution = group_profits * weights[i+1] * allocated_capital
                    self.bbb = contribution
                    total += contribution.mean()
                    
                    # Normalize profits and calculate contributions
                    #normalized_profits = (group_profits - group_profits.min()) / (group_profits.max() - group_profits.min())
                    normalized_contribution = normalized_profits[group] * weights[i+1] * allocated_capital
                    normalised_total += normalized_contribution.mean()
            
        return total, normalised_total

    

    
    def getMDD(self,ts_data):
        total = 0
        ts_mdd = {}
        for i in range(len(self.group_part)):
            tsp_mins = []
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
                tsp_mins.append(max_drawdown)
            if tsp_mins:
                total+= max(tsp_mins)
        mdd = total/len(self.group_part)
        normalised_total = 0 
        ts_mdd = self.normalisation(ts_mdd)
        for i in range(len(self.group_part)):
            for j in self.group_part[i]:
                normalised_mdd = ts_mdd[j]
                normalised_contribution = normalised_mdd
                normalised_total += normalised_contribution  
     
        normalised_mdd = normalised_total/len(self.group_part)
        return mdd, normalised_mdd
    
    
    

        
    def getCorrelation(self,ts_data):
        weights = self.getWeights()
        self.weights = weights
        weights_df = pd.DataFrame([weights])
        weights_df = weights_df.drop(0,axis=1)
        total = 0
        combs = list(itertools.product(*self.group_part))
        for ts in  combs:
            tsp = ts_data[list(ts)].corr()
            tsp_var = np.linalg.multi_dot([weights_df.to_numpy(),tsp.to_numpy(),weights_df.to_numpy().T])
            total += tsp_var
        try:
            total = total.item(0)
        except:
            total = total
        return total

    def groupBalance1(self):
        N = len(self.strategies)
        gb = 0
        for group in self.group_part:
            try:
                g_result = len(group)/N
            except ZeroDivisionError:
                g_result = 0
            if g_result == 0:
                g =0
            else:
                g = -g_result * np.log(g_result)
            gb += g
        if gb == 0:
            gb = 1
        return gb
    
    def groupBalance(self):
        N = len(self.strategies)
        num_groups = len(self.group_part)
        gb = 0
        for group in self.group_part:
            g_size = len(group)
            g_result = g_size / N if N > 0 else 0
            g = -g_result * np.log(g_result) if g_result > 0 else 0
            gb += g

        return gb / num_groups

    def weightBalance(self):
        weight_part = self.weight_part.copy()
        gb = 0
        TL = sum([i for i in self.weight_part if i == 1])
        K = self.K+1
        weights = {k: [] for k in range(K)}
        for i in range(K):
            while weight_part:
                    x = weight_part.pop(0)
                    if x == 0:
                        break
                    else:
                        weights[i].append(x)
        
        for i in weights:
            try:
                w = len(weights[i])/TL
            except ZeroDivisionError:
                w = 0
            if w !=0:
                wb = -w * np.log(w)
            else:
                wb = 0
            
            gb += wb
        
        return gb
    
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
    
    def strategy_performance(self,data):
        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp()
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy] =  self.generate_trading_signal(data,strategy,chromomosome_stop_loss,chromomosome_take_profit)
        strategy_performance = pd.DataFrame.from_dict(strategy_performance)
        return strategy_performance

    def calculate_chromosome_fitness(self,data,allocated_capital):
        monthly_returns = self.strategy_performance(data)
        self.mdd,fit_mdd = self.getMDD(monthly_returns)
        self.profit,fit_profit =self.getProfit(monthly_returns,allocated_capital)
        #self.corr = self.getCorrelation(monthly_returns)
        self.gb = self.groupBalance() 
        self.wb = self.weightBalance()
        try:
            fitness = fit_profit * (1/ fit_mdd)  + self.gb + self.wb #* np.power(gb,2)
        except:
            fitness = fit_profit + self.gb + self.wb #* np.power(gb,2)
        self.fitness_value = fitness
    
    
    def scale_fitness(self,max_fitness,min_fitness):
        # Linear scaling
        #min_fitness = min(population_fitness)
        #max_fitness = max(population_fitness)
        shift = -min_fitness
        self.fitness_value += shift
        desired_min_fitness = 0
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
            index = random.randint(1, len(self.weight_part)-2)
            child1.weight_part = self.weight_part[index:] + parent2.weight_part[:index] 
            if child1.weight_part.count(0) != self.K+1:
                child1.generateWeight()
            child2.weight_part = parent2.weight_part[index:] + self.weight_part[:index] 
            if child2.weight_part.count(0) != self.K+1:
                child2.generateWeight()

        return child1,child2
    # 2 point Crossover
    def crossover1(self, parent2, r_cross):
        child1 = deepcopy(self)
        child2 = deepcopy(parent2)
        # Check for recombination
        if random.random() < r_cross:
            # Select two crossover points that are not on the end of the string
            index1 = random.randint(1, len(self.sltp_part)-2)
            index2 = random.randint(index1+1, len(self.sltp_part)-1)
            # Perform crossover on SLTP
            child1.sltp_part = self.sltp_part[:index1] + parent2.sltp_part[index1:index2] + self.sltp_part[index2:]
            child2.sltp_part = parent2.sltp_part[:index1] + self.sltp_part[index1:index2] + parent2.sltp_part[index2:]
            # Perform crossover on weight
            child1.weight_part = self.weight_part[:index1] + parent2.weight_part[index1:index2] + self.weight_part[index2:]
            if child1.weight_part.count(0) != self.K+1:
                child1.generateWeight()
            child2.weight_part = parent2.weight_part[:index1] + self.weight_part[index1:index2] + parent2.weight_part[index2:]
            if child2.weight_part.count(0) != self.K+1:
                child2.generateWeight()
        return child1, child2

    ###Mutation

    def mutation(self,r_mut):
        # check for a mutation
        
            # on SLTP
        for i in range(len(self.sltp_part)):
            if random.random() < r_mut:
            # flip the bit
                self.sltp_part[i] = 1 - self.sltp_part[i]
        # on Weight Part
        for i in range(len(self.weight_part)):
            if random.random() < r_mut:
            # flip the bit 
                self.weight_part[i] = 1 - self.weight_part[i] 
            #check if condition is still maintained
                if self.weight_part.count(0) != self.K+1:
                    self.generateWeight() 

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
        
    
    
    