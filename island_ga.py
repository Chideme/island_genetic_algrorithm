import numpy as np
import random
import itertools
import math
import talib as ta
import pandas as pd
import multiprocess as mp
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
import time

class IslandGGA():

    def __init__(self,num_island,num_iter,data,strategies,pSize,m_iter,N,K,r_cross,r_mut,r_inv,n,b,stop_loss,take_profit,allocated_capital):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.num_islands = num_island
        self.m_iter = m_iter
        self.r_cross= r_cross
        self.r_mut = r_mut
        self.r_inv = r_inv
        self.num_iter = num_iter
        self.n = n
        self.b = b
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.allocated_capital = allocated_capital
        self.num_weight = (self.K*2) + 1
        self.population = []
        self.best = []

        
    def get_last_date_of_month(self,year, month):
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
        
        return last_date


    def binary_to_sltp(self,sltp_part):
        """CONVERT SLPT Part to float"""
        
        sl_part,tp_part = sltp_part[:self.n],sltp_part[self.n:]
        max_sl = sum([np.power(2,i)*1 for i in range(len(sl_part))])
        max_tp = sum([np.power(2,i)*1 for i in range(len(tp_part))])
        sl_part.reverse()
        tp_part.reverse()
        sl  = sum([np.power(2,i)*sl_part[i] for i in range(len(sl_part))])
        tp  = sum([np.power(2,i)*tp_part[i] for i in range(len(tp_part))])
        sl = self.stop_loss/max_sl * sl
        tp = self.take_profit/max_tp * tp

        return sl,tp


    def generate_trading_signal(self,strategy,chromosome_stop_loss,chromosome_take_profit):
        monthly_returns = []
        monthly_return = 0
        trade_freq = 0
        monthly_freq = 0
        market_position = 'out'
        max_loss = 0
        for row in self.data.itertuples(index=False):    
            #close all trade positions at month end. 
            last_date = self.get_last_date_of_month(row.Date.year,row.Date.month)
            if row.Date == last_date :
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
                        avg_monthly_return = 1
                    monthly_returns.append(avg_monthly_return)
                
                    monthly_return = 0
                    monthly_freq = 0         
            else:
                if market_position == 'out' :
                    if row[self.data.columns.get_loc(strategy)] == 1:
                        cost_price = row.close
                        market_position = 'in'
                        
                else:
                    sell_price = row.close
                    trade_return = (sell_price - cost_price)/cost_price
                    if trade_return <= chromosome_take_profit or trade_return >= chromosome_take_profit: 
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'   
                        
                    if row[self.data.columns.get_loc(strategy)] == 0 and trade_return >= chromosome_take_profit :
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'
        
        return monthly_returns

    def strategy_performance(self,sltp_part):
        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp(sltp_part)
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy] = self.generate_trading_signal(strategy,chromomosome_stop_loss,chromomosome_take_profit)
            
        strategy_performance = pd.DataFrame.from_dict(strategy_performance)
        return strategy_performance



    # generate maxdraw down
    def generateMDD(self,strategy,stop_loss,take_profit):
        monthly_returns = []
        monthly_return = 0
        trade_freq = 0
        monthly_freq = 0
        market_position = 'out'
        max_loss = 0
        for row in self.data.itertuples(index=False):    
            #close all trade positions at month end. 
            last_date = self.get_last_date_of_month(row.Date.year,row.Date.month)
            if row.Date == last_date :
                if market_position =='in':
                    sell_price = row.close
                    trade_return = (sell_price - cost_price)/cost_price
                    if trade_return < max_loss:
                            max_loss = trade_return
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
                        avg_monthly_return = 1
                    monthly_returns.append(avg_monthly_return)
                
                    monthly_return = 0
                    monthly_freq = 0         
            else:
                if market_position == 'out' :
                    if row[self.data.columns.get_loc(strategy)] == 1:
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
                        
                    if row[self.data.columns.get_loc(strategy)] == 0 and trade_return >= take_profit :
                        trade_freq +=1
                        monthly_freq+=1
                        if trade_return < max_loss:
                            max_loss = trade_return
                        monthly_return += trade_return
                        market_position = 'out'
    
    
        return max_loss

    def strategy_mdd(self,sltp_part):
        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp(sltp_part)
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy] = self.generateMDD(strategy,chromomosome_stop_loss,chromomosome_take_profit)
            
        
        return strategy_performance


    # init population    
    
    def generateSLTP(self):
        """Generate bits for SLTP Part"""
        l = self.n + self.b
        sltp= [random.randint(0, 1) for _ in range(l)]
        
        return sltp


    def generateGroup(self):
        """Generate Group and assign TS to groups K"""
        x = self.strategies.copy()
        random.shuffle(x)
        groups = [[] for k in range(self.K)]
        while x:
            s = x.pop()
            random_index = random.randrange(self.K)
            groups[random_index].append(s)
        return groups

    def generateWeight(self):
        """Generate Weight Part and assign to groups K"""
        weights= [1 for _ in range(self.num_weight)]
        K = self.K+1
        for i in range(K):
            random_index = random.randrange(K)
            weights[random_index] = 0
        return weights

    def init_population(self):
        population =[]
        for i in range(self.pSize):
            chromosome = [] #c = [[SLTP],[[K],[Weight]]
            chromosome.append(self.generateSLTP()) #SLPT PART
            chromosome.append(self.generateGroup()) # TS & Group Part"
            chromosome.append(self.generateWeight()) #Weight Part
            chromosome.append(1) # default fitness value at init
            population.append(chromosome)
        
        return population


    #Fitness Function
        
    def get_max_min(self,p4mc):
        modified ={}
        for i in  p4mc:
            modified[i] = p4mc[i]
        max_key= max(modified, key=modified.get)
        min_key = min(modified, key=modified.get)
        
        return modified[max_key], modified[min_key]


    def normalisation_mdd(self,p4mc):
        max_value,min_value = self.get_max_min(p4mc)
        for i in p4mc:
            p4mc[i] = (p4mc[i] - min_value )/ (max_value - min_value)
        return p4mc

    def normalisation(trading_strategies_data):
        normalised_ts_data = trading_strategies_data.copy()
        for ts in trading_strategies_data.columns:
            max_value = trading_strategies_data[ts].max()
            min_value = trading_strategies_data[ts].min()
            normalised_ts_data[ts] = (trading_strategies_data[ts] - min_value) / (max_value - min_value)
        return normalised_ts_data

    def getWeights(self,weightPart):
        w = weightPart.copy()
        L = sum([i for i in w if i == 1])
        K = self.K+1
        z = {k: [] for k in range(K)}
        for i in range(K):
            while w:
                    x = w.pop(0)
                    if x == 0:
                        break
                    else:

                        z[i].append(x)
        
        return z
        

    def getProfit(self,p4mc,chromosome):
        weights = self.getWeights(chromosome[2])
        w = chromosome[2]
        L = sum([i for i in w if i == 1])
        for i in weights:
            try:
                weights[i]= round(len(weights[i])/L,2)
            except ZeroDivisionError:
                weights[i] = 0
        total = 0
        for i in range(len(chromosome[1])):
            for j in chromosome[1][i]:
                avg_return = p4mc[j].mean()
                total += avg_return*weights[i+1]*self.allocated_capital
        return total


        
    def getCorrelation(self,ts_data,chromosome):
        weights = self.getWeights(chromosome[2])
        w = chromosome[2]
        L = sum([i for i in w if i == 1])
        for i in weights:
            try:
                weights[i]= round(len(weights[i])/L,2)
            except ZeroDivisionError:
                weights[i] = 0
        weights_df = pd.DataFrame([weights])
        weights_df = weights_df.drop(0,axis=1)
        total = 0
        combs = list(itertools.product(*chromosome[1]))
        for ts in  combs:
            tsp = ts_data[list(ts)].corr()
            tsp_var = np.linalg.multi_dot([weights_df.to_numpy(),tsp.to_numpy(),weights_df.to_numpy().T])
            total += tsp_var
        try:
            total = total.item(0)
        except:
            total = total
        return total

    def groupBalance(self,chromosome):
        N = len(self.strategies)
        gb = 0
        for group in chromosome[1]:
            try:
                g_result = len(group)/N
            except ZeroDivisionError:
                g_result = 0
            if g_result == 0:
                g =0
            else:
                g = -g_result * np.log(g_result)
            gb += g
        return gb

    def weightBalance(self,chromosome):
        gb = 0
        TL = sum([i for i in chromosome[2] if i == 1])
        weights = self.getWeights(chromosome[2])
        
        for i in weights:
            try:
                w = len(weights[i])/TL
            except ZeroDivisionError:
                w = 0
            if w == 0:
                wb = 0  
            else:
                wb = -w * np.log(w)
            if wb:
                gb += wb
        return gb

    def fitness_function(self,chromosome):
        ts_data = self.strategy_performance(chromosome[0])
        normalised_ts_data = self.normalisation(ts_data)
        profit =self.getProfit(normalised_ts_data,chromosome)
        corr = self.getCorrelation(ts_data,chromosome)
        gb = self.groupBalance(chromosome)
        wb = self.weightBalance(chromosome)
        fitness = profit * corr* np.power(gb,2) * wb
        chromosome[3] = fitness 
        return chromosome

   
    ##### Genetic operators
    def hamming_distance(self,chromosome_string1, chromosome_string2): 	
        dist_counter = 0 
        #sltp part
        for n in range(len(chromosome_string1[0])):
            if chromosome_string1[0][n] != chromosome_string2[0][n]:
                dist_counter += 1
        #grouping part
        for n in range(len(chromosome_string1[1])):
            if chromosome_string1[1][n] != chromosome_string2[1][n]:
                dist_counter += 1
        #weight part
        for n in range(len(chromosome_string1[2])):
            if chromosome_string1[2][n] != chromosome_string2[2][n]:
                dist_counter += 1
        return dist_counter

    # Select parents

    def roulette_wheel_selection(self,population):
    
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome[3] for chromosome in population])
        
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome[3]/population_fitness for chromosome in population]
        # Selects one chromosome based on the computed probabilities
        population = np.array(population,dtype=object)
        output = population[np.random.choice(population.shape[0],p=chromosome_probabilities)]
        
        return list(output)#np.random.choice(population, p=chromosome_probabilities)

    def selection(self,population):
        selected = []
        for i in range(len(population)):
            selected.append(self.roulette_wheel_selection(population))
        return selected

    ### Crossover
    def crossover(self,parent1,parent2):
        child1 = parent1
        child2 = parent2
        # check for recombination
        if random.random() < self.r_cross:
            # select crossover point that is not on the end of the string
            index = random.randint(1, len(parent1[0])-2)
            # perform crossover on SLTP
            child1[0] = parent2[0][:index] + parent1[0][index:]
            child2[0] = parent1[0][:index] + parent2[0][index:]
            # perform crossover on weight
            index = random.randint(1, len(parent1[2])-2)
            child1[2] = parent2[2][:index] + parent1[2][index:]
            child2[2] = parent1[2][:index] + parent2[2][index:]
        return child1,child2
    ###Mutation

    def mutation(self,chromosome):
        # on SLTP
        for i in range(len(chromosome[0])):
            # check for a mutation
            if random.random() < self.r_mut:
                # flip the bit
                chromosome[0][i] = 1 - chromosome[0][i]
        # on Weight Part
        for i in range(len(chromosome[2])):
            # check for a mutation
            if random.random() < self.r_mut:
                # flip the bit 
                chromosome[2][i] = 1 - chromosome[2][i] 
        # on TS part
        e = random.random()
        if e < self.r_mut:
                grp_idx1 = random.randrange(len(chromosome[1]))
                grp_idx2 = random.randrange(len(chromosome[1]))
                ts_idx = random.randrange(len(chromosome[1][grp_idx1]))
                ts = chromosome[1][grp_idx1][ts_idx]
                chromosome[1][grp_idx2].append(ts)
                
        return chromosome
    ###Inversion
    def inversion(self,chromosome):
        if random.random() < self.r_inv:
            inverted = chromosome[1][::-1]
            chromosome[1] = inverted
            #chromosome[1][grp_idx2] = ts_1
        return chromosome



######## MIGRATION
    def best_chromosomes(self,population,q):
        q.put([i for i in sorted(population, key=lambda x: x[3])[-self.N:]])

    def select_best_chromosomes(self,population):
        return [i for i in sorted(population, key=lambda x: x[3])[-self.N:]]

    def worst_chromosomes(self,population):
        worst = [i for i in sorted(population, key=lambda x: x[3])[:self.N]]
        
        for i in worst:
            population.remove(i)
        return population

    def genetic_operations(self,island,q):
        """evolve each island per generation"""
        for chromosome in island:
            self.fitness_function(chromosome)
        tempPopu  = self.selection(island)
        children = []
        #Crossover
        for i in range(0, len(tempPopu)-1, 2):
                # get selected parents in pairs
                parent1,parent2 = tempPopu[i],tempPopu[i+1]
                #crossover and mutation and inversion 
                child1,child2 = self.crossover(parent1,parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                child1 = self.inversion(child1)
                child2 = self.inversion(child2)
                children.append(child1)
                children.append(child2)
        island = children
        q.put(island)   

    #############################################

    def generate_trading_signal(self,strategy,stop_loss,take_profit):
        total_strategy_return = 0
        trade_freq = 0
        market_position = 'out'
        max_loss = 0
        for row in self.data.itertuples(index=False): 
            if market_position == 'out' :
                if row[self.data.columns.get_loc(strategy)] == 1:
                    cost_price = row.close
                    market_position = 'in'
            else:
                sell_price = row.close
                trade_return = round(((sell_price - cost_price)/cost_price),2)
                if trade_return <= stop_loss or trade_return >= take_profit: 
                    trade_freq +=1
                    if trade_return < max_loss:
                        max_loss = trade_return
                    total_strategy_return += trade_return
                    market_position = 'out'   
                if row[self.data.columns.get_loc(strategy)] == 0 and trade_return >= take_profit :
                    market_position = 'out'
                    trade_freq +=1
                    if trade_return < max_loss:
                        max_loss = trade_return
                    total_strategy_return += trade_return
        avg_return = round(total_strategy_return/trade_freq,2)
        strategy_performance ={strategy: {"avg_return":avg_return,"mdd":max_loss,"frequency":trade_freq}}
        return strategy_performance
    
    def generateSLTP(self):
        """Generate bits for SLTP Part"""
        l = self.n + self.b
        sltp= [random.randint(0, 1) for _ in range(l)]
        
        return sltp

    def binary_to_sltp(self,sltp_part):
        """CONVERT SLPT Part to float"""
        
        sl_part,tp_part = sltp_part[:self.n],sltp_part[self.n:]
        max_sl = sum([np.power(2,i)*1 for i in range(len(sl_part))])
        max_tp = sum([np.power(2,i)*1 for i in range(len(tp_part))])
        sl_part.reverse()
        tp_part.reverse()
        sl  = sum([np.power(2,i)*sl_part[i] for i in range(len(sl_part))])
        tp  = sum([np.power(2,i)*tp_part[i] for i in range(len(tp_part))])
        sl = self.stop_loss/max_sl * sl
        tp = self.take_profit/max_tp * tp

        return sl,tp

    def strategy_performance(self,sltp_part):

        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp(sltp_part)
        strategy_performance = {}
        for strategy in self.strategies:
            a = self.generate_trading_signal(strategy,chromomosome_stop_loss,chromomosome_take_profit)
            
            strategy_performance[strategy]= a[strategy]
        return strategy_performance

    def generateGroup(self):
        """Generate Group and assign TS to groups K"""
        x = self.strategies.copy()
        random.shuffle(x)
        groups = [[] for k in range(self.K)]
        while x:
            s = x.pop()
            random_index = random.randrange(self.K)
            groups[random_index].append(s)
        return groups

    def generateWeight(self):
        """Generate Weight Part and assign to groups K"""
        weights= [1 for _ in range(self.num_weight)]
        K_W = self.K+1
        for i in range(K_W):
            random_index = random.randrange(K_W)
            weights[random_index] = 0
        return weights
    
    def init_population(self):
        if self.population == []:
            for i in range(self.pSize):
                chromosome = [] #c = [[SLTP],[[K],[Weight]]
                chromosome.append(self.generateSLTP()) #SLPT PART
                chromosome.append(self.generateGroup()) # TS & Group Part"
                chromosome.append(self.generateWeight()) #Weight Part
                chromosome.append(1) #fitness
                self.population.append(chromosome)

    def get_max_min(self,p4mc,metric):
        modified ={}
        for i in  p4mc:
            modified[i] = p4mc[i][metric]
        max_key= max(modified, key=modified.get)
        min_key = min(modified, key=modified.get)
        
        return modified[max_key], modified[min_key]   

    def getRisk(self,p4mc,chromosome):
        all_tsp = 0
        for i in range(len(chromosome[1])):
            
            tsp =[]
            for j in chromosome[1][i]:
                tsp.append(p4mc[j]['mdd'])
            if tsp:
                mdd = min(tsp)
            else:
                mdd =0
            all_tsp += mdd
        all_tsp = all_tsp/len(chromosome[1])
        return all_tsp

    def normalisation(self,p4mc,metric):
        max_value,min_value = self.get_max_min(p4mc,metric)
        for i in p4mc:
            p4mc[i][metric] = (p4mc[i][metric] - min_value )/ (max_value - min_value)
        return p4mc

    def getWeights(self,weightPart):
        w = weightPart.copy()
        L = sum([i for i in w if i == 1])
        K_W = self.K+1
        z = {k: [] for k in range(K_W)}
        for i in range(K_W):
            while w:
                    x = w.pop(0)
                    if x == 0:
                        break
                    else:

                        z[i].append(x)
        
        return z

    def getProfit(self,p4mc,chromosome):
        weights = self.getWeights(chromosome[2])
        w = chromosome[2]
        L = sum([i for i in w if i == 1])
        for i in weights:
            try:
                weights[i]= round(len(weights[i])/L,2)
            except ZeroDivisionError:
                weights[i] = 0
        total = 0
        for i in range(len(chromosome[1])):
            for j in chromosome[1][i]:
                total += p4mc[j]['avg_return']*weights[i+1]*self.allocated_capital
        return total

    def groupBalance(self,chromosome):
        N = len(self.strategies)
        gb = 0
        for group in chromosome[1]:
            try:
                g_result = len(group)/N
            except ZeroDivisionError:
                g_result = 0
            if g_result == 0:
                g =0
            else:
                g = -g_result * np.log(g_result)
            gb += g
        return gb  

    def weightBalance(self,chromosome):
        gb = 0
        TL = sum([i for i in chromosome[2] if i == 1])
        weights = self.getWeights(chromosome[2])
        
        for i in weights:
            try:
                w = len(weights[i])/TL
            except ZeroDivisionError:
                w = 0
            if w == 0:
                wb = 0  
            else:
                wb = -w * np.log(w)
            if wb:
                gb += wb
        return gb


    def fitness_function(self,chromosome):
        performance = self.strategy_performance(chromosome[0])
        p4mc = self.normalisation(performance,'mdd')
        profit =self.getProfit(p4mc,chromosome)
        risk = self.getRisk(p4mc,chromosome)
        
        gb = self.groupBalance(chromosome)
    
        wb = self.weightBalance(chromosome)
        fitness = profit * risk * np.power(gb,2) * wb
        chromosome[3] = fitness 
        return chromosome

    def roulette_wheel_selection(self):
    
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome[3] for chromosome in self.population])
        
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome[3]/population_fitness for chromosome in self.population]
        # Selects one chromosome based on the computed probabilities
        self.population = np.array(self.population,dtype=object)
        output = self.population[np.random.choice(self.population.shape[0],p=chromosome_probabilities)]
        
        return list(output)#np.random.choice(population, p=chromosome_probabilities)

    def selection(self):
        selected = []
        for i in range(len(self.population)):
            selected.append(self.roulette_wheel_selection())
        return selected
    ### Crossover
    def crossover(self,parent1,parent2):
        child1 = parent1
        child2 = parent2
        # check for recombination
        if random.random() < self.r_cross:
            # select crossover point that is not on the end of the string
            index = random.randint(1, len(parent1[0])-2)
            # perform crossover on SLTP
            child1[0] = parent2[0][:index] + parent1[0][index:]
            child2[0] = parent1[0][:index] + parent2[0][index:]
            # perform crossover on weight
            index = random.randint(1, len(parent1[2])-2)
            child1[2] = parent2[2][:index] + parent1[2][index:]
            child2[2] = parent1[2][:index] + parent2[2][index:]
        return child1,child2
    ###Mutation

    def mutation(self,chromosome):
        # on SLTP
        for i in range(len(chromosome[0])):
            # check for a mutation
            if random.random() < self.r_mut:
                # flip the bit
                chromosome[0][i] = 1 - chromosome[0][i]
        # on Weight Part
        for i in range(len(chromosome[2])):
            # check for a mutation
            if random.random() < self.r_mut:
                # flip the bit 
                chromosome[2][i] = 1 - chromosome[2][i] 
        # on TS part
            
        if random.random() < self.r_mut:
                grp_idx1 = random.randrange(len(chromosome[1]))
                grp_idx2 = random.randrange(len(chromosome[1]))
                ts_idx = random.randrange(len(chromosome[1][grp_idx1]))
                ts = chromosome[1][grp_idx1][ts_idx]
                chromosome[1][grp_idx2].append(ts)
                
        return chromosome

    ###Inversion
    def inversion(self,chromosome):
        if random.random() < self.r_inv:
            inverted = chromosome[1][::-1]
            chromosome[1] = inverted
            #chromosome[1][grp_idx2] = ts_1
        return chromosome

    def evolve(self):
        self.init_population()
        self.best = self.population[0]
        for j in range(self.n_iter):
            
            for chromosome in self.population:
                self.fitness_function(chromosome)
            tempPopu  = self.selection()
            children = []
            #Crossover
            for i in range(0, len(tempPopu)-1, 2):

                    # get selected parents in pairs
                    parent1,parent2 = tempPopu[i],tempPopu[i+1]
                    #crossover and mutation and inversion 
                    child1,child2 = self.crossover(parent1,parent2)
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                    child1 = self.inversion(child1)
                    child2 = self.inversion(child2)
                    children.append(child1)
                    children.append(child2)
            self.population = children
            for chromosone in self.population:
                    if chromosone[3] > self.best[3]:
                        self.best = chromosone
