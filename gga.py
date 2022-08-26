import numpy as np
import random
import math
import talib as ta
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,date


class GGA():

    def __init__(self,data, K,pSize,strategies,r_cross,r_mut,r_inv,n_iter,n,b,stop_loss,take_profit,allocated_capital):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.r_cross= r_cross
        self.r_mut = r_mut
        self.r_inv = r_inv
        self.n_iter = n_iter
        self.n = n
        self.b = b
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.allocated_capital = allocated_capital
        self.num_weight = (self.K*2) + 1
        self.population = []
        self.best = []

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
