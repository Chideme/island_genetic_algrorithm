import numpy as np
import random
import math
import talib as ta
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,date




def generate_candidate_trading_signals(data):
    """Based on parameter setting adopted in Chen et al (2021) """
    conditions ={'TS1_CONDITIONS':[
                (data['5EMA'] > data['20EMA']),
                (data['5EMA'] < data['20EMA'])],
                 'TS2_CONDITIONS':[
                (data['RSI'] > 30),
                (data['RSI'] < 70),
                ],
                 'TS3_CONDITIONS':[
                (data['WILLR'] < 80),
                (data['WILLR'] > 20),
                ],
                 'TS4_CONDITIONS':[
                (data['MOM'] > 0 ),
                (data['MOM'] <= 0),
                ],
                 'TS5_CONDITIONS': [
                (data['CCI'] > 100 ),
                (data['CCI'] <= 100),
                ],
                 'TS6_CONDITIONS': [
                (data['SLOWK'] > data['SLOWD']) & (data['SLOWD'] < 20),
                (data['SLOWK'] < data['SLOWD']) & (data['SLOWD'] > 80)],
                'TS7_CONDITIONS': [
                (data['MACD'] > 0 ),
                (data['MACD'] <= 0)],
                'TS8_CONDITIONS': [
                (data['CCI'] > 100 ),
                (data['CCI'] <= -100)]}

    # create a list of the values we want to assign for each condition 1: buy, 0: sell
    values = [1, 0]

    # create a new column and use np.select to assign values to it using our lists as arguments
    for i in conditions:
        data[i] = np.select(conditions[i], values)
    strategies =list(conditions.keys())
    return data,strategies


def generate_trading_signal(data,strategy,stop_loss,take_profit):
    total_strategy_return = 0
    trade_freq = 0
    market_position = 'out'
    max_loss = 0
    for row in data.itertuples(index=False): 
        if market_position == 'out' :
            if row[data.columns.get_loc(strategy)] == 1:
                cost_price = row.close
                market_position = 'in'
        else:
            sell_price = row.close
            trade_return = round(((sell_price - cost_price)/cost_price) * 100,2)
            if trade_return <= stop_loss or trade_return >= take_profit: 
                trade_freq +=1
                if trade_return < max_loss:
                    max_loss = trade_return
                total_strategy_return += trade_return
                market_position = 'out'   
            if row[data.columns.get_loc(strategy)] == 0 and trade_return >= take_profit :
                market_position = 'out'
                trade_freq +=1
                if trade_return < max_loss:
                    max_loss = trade_return
                total_strategy_return += trade_return
    avg_return = round(total_strategy_return/trade_freq,2)
    strategy_performance ={strategy: {"avg_return":avg_return,"mdd":max_loss,"frequency":trade_freq}}
    
    return strategy_performance


def generateSLTP(n,b):
    """Generate bits for SLTP Part"""
    l = n + b
    sltp= [random.randint(0, 1) for _ in range(l)]
    
    return sltp

def binary_to_sltp(sltp,n,b,sl_boundary,tp_boundary):
    """CONVERT SLPT Part to float"""
    
    sl_part,tp_part = sltp[:n],sltp[n:]
    max_sl = sum([np.power(2,i)*1 for i in range(len(sl_part))])
    max_tp = sum([np.power(2,i)*1 for i in range(len(tp_part))])
    sl_part.reverse()
    tp_part.reverse()
    sl  = sum([np.power(2,i)*sl_part[i] for i in range(len(sl_part))])
    tp  = sum([np.power(2,i)*tp_part[i] for i in range(len(tp_part))])
    sl = sl_boundary/max_sl * sl
    tp = tp_boundary/max_tp * tp
    return sl,tp

def strategy_performance(data,strategies,sltp,n,b,stop_loss,take_profit):
    sl, tp = binary_to_sltp(sltp,n,b,stop_loss,take_profit)
    strategy_performance = {}
    for strategy in strategies:
        a = generate_trading_signal(data,strategy,sl,tp)
        
        strategy_performance[strategy]= a[strategy]
    return strategy_performance

def generateGroup(K,strategies):
    """Generate Group and assign TS to groups K"""
    x = strategies.copy()
    random.shuffle(x)
    groups = [[] for k in range(K)]
    while x:
        s = x.pop()
        random_index = random.randrange(K)
        groups[random_index].append(s)
    return groups

def generateWeight(K,num_weight):
    """Generate Weight Part and assign to groups K"""
    weights= [1 for _ in range(num_weight)]
    K = K+1
    for i in range(K):
        random_index = random.randrange(K)
        weights[random_index] = 0
    return weights

def init_population(pSize,n,b,K,num_weight,strategies):
    population =[]
    for i in range(pSize):
        chromosome = [] #c = [[SLTP],[[K],[Weight]]
        chromosome.append(generateSLTP(n,b)) #SLPT PART
        chromosome.append(generateGroup(K,strategies)) # TS & Group Part"
        chromosome.append(generateWeight(K,num_weight)) #Weight Part
        chromosome.append(1) #fitness
        population.append(chromosome)
    
    return population

 #Fitness Function
    
def get_max_min(p4mc,metric):
    modified ={}
    for i in  p4mc:
        modified[i] = p4mc[i][metric]
    max_key= max(modified, key=modified.get)
    min_key = min(modified, key=modified.get)
    
    return modified[max_key], modified[min_key]


    
def getRisk(p4mc,chromosome):
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

def normalisation(p4mc,metric):
    max_value,min_value = get_max_min(p4mc,metric)
    for i in p4mc:
        p4mc[i][metric] = (p4mc[i][metric] - min_value )/ (max_value - min_value)
    return p4mc

def getWeights(weightPart,K):
    w = weightPart.copy()
    L = sum([i for i in w if i == 1])
    K = K+1
    z = {k: [] for k in range(K)}
    for i in range(K):
        while w:
                x = w.pop(0)
                if x == 0:
                    break
                else:

                    z[i].append(x)
    
    return z
    


def getProfit(p4mc,chromosome,allocated_capital,K):
    weights = getWeights(chromosome[2],K)
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
            total += p4mc[j]['avg_return']*weights[i+1]*allocated_capital
    return total

def groupBalance(chromosome,N):
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

def weightBalance(chromosome,K):
    gb = 0
    TL = sum([i for i in chromosome[2] if i == 1])
    weights = getWeights(chromosome[2],K)
    
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

def fitness_function(chromosome,strategies,n,b,stop_loss,take_profit,allocated_capital,K):
    performance = strategy_performance(data,strategies,chromosome[0],n,b,stop_loss,take_profit)
    p4mc = normalisation(performance,'mdd')
    profit =getProfit(p4mc,chromosome,allocated_capital,K)
    risk = getRisk(p4mc,chromosome)
    
    gb = groupBalance(chromosome,len(strategies))
 
    wb = weightBalance(chromosome,K)
    fitness = profit * risk * np.power(gb,2) * wb
    chromosome[3] = fitness 
    return chromosome


def roulette_wheel_selection(population):
  
    # Computes the totallity of the population fitness
    population_fitness = sum([chromosome[3] for chromosome in population])
    
    # Computes for each chromosome the probability 
    chromosome_probabilities = [chromosome[3]/population_fitness for chromosome in population]
    # Selects one chromosome based on the computed probabilities
    population = np.array(population,dtype=object)
    output = population[np.random.choice(population.shape[0],p=chromosome_probabilities)]
    
    return list(output)#np.random.choice(population, p=chromosome_probabilities)

def selection(population):
    selected = []
    for i in range(len(population)):
        selected.append(roulette_wheel_selection(population))
    return selected

def crossover(parent1,parent2,r_cross):
    child1 = parent1
    child2 = parent2
    # check for recombination
    if random.random() < r_cross:
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

def mutation(chromosone, r_mut):
    # on SLTP
    for i in range(len(chromosone[0])):
        # check for a mutation
        if random.random() < r_mut:
            # flip the bit
            chromosone[0][i] = 1 - chromosone[0][i]
     # on Weight Part
    for i in range(len(chromosone[2])):
        # check for a mutation
        if random.random() < r_mut:
            # flip the bit 
            chromosone[2][i] = 1 - chromosone[2][i] 
    # on TS part
        
    if random.random() < r_mut:
            grp_idx1 = random.randrange(len(chromosome[1]))
            grp_idx2 = random.randrange(len(chromosome[1]))
            ts_idx = random.randrange(len(chromosome[1][grp_idx1]))
            ts = chromosome[1][grp_idx1][ts_idx]
            print(ts)
            chromosome[1][grp_idx2].append(ts)
            
    return chromosone

def inversion(chromosome, r_inv):
    if random.random() < r_inv:
        grp_idx1 = random.randrange(len(chromosome[1]))
        grp_idx2 = random.randrange(len(chromosome[1]))
        ts_1 = chromosome[1][grp_idx1]
        ts_2 = chromosome[1][grp_idx2]
        chromosome[1][grp_idx1] = ts_2
        chromosome[1][grp_idx2] = ts_1
    return chromosome

def gga(data,K,pSize,strategies,r_cross,r_mut,r_inv,n_iter,n,b,stop_loss,take_profit,allocated_capital):
    for j in range(n_iter):
        num_weight = (K*2) + 1
        population = init_population(pSize,n,b,K,num_weight,strategies)
        best = population[0]
        for chromosome in population:
            fitness_function(chromosome,strategies,n,b,stop_loss,take_profit,allocated_capital,K)
        tempPopu  = selection(population)
        children = []
        #Crossover
        for i in range(0, len(tempPopu)-1, 2):

                # get selected parents in pairs
                parent1,parent2 = tempPopu[i],tempPopu[i+1]
                #crossover and mutation and inversion 
                child1,child2 = crossover(parent1,parent2,r_cross)
                child1 = mutation(child1, r_mut)
                child2 = mutation(child2, r_mut)
                child1 = inversion(child1, r_inv)
                child2 = inversion(child2, r_inv)
                children.append(child1)
                children.append(child2)
        population = children
        for chromosone in population:
                if chromosone[3] > best[3]:
                    best = chromosone
    return best

def best(queue):
    data,strategies =  generate_candidate_trading_signals(aapl.copy())
    num = gga(data,3,10,strategies,0.9,0.01,0.01,20,4,4,-0.15,15,1000)
    queue.put(num)
