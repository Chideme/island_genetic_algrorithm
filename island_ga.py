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

    def __init__(self,num_islands,num_iter,data,strategies,pSize,m_iter,N,K,r_cross,r_mut,r_inv,n,b,stop_loss,take_profit,allocated_capital):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.num_islands = num_islands
        self.N = N
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
        self.islands = []
        self.best_individuals = []
        self.globalBest  = []
        self.fitness_values = []

    def re_init(self):
        self.islands = []
        self.best_individuals = []
        self.globalBest  = []
        self.fitness_values = []

        
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
        last_date = last_date.date()
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
#################

    def generate_trading_signal(self,data,strategy,stop_loss,take_profit):
        monthly_returns = []
        monthly_return = 0
        trade_freq = 0
        monthly_freq = 0
        market_position = 'out'
        max_loss = 0
        for row in data.itertuples(index=False):    
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


##################


    def strategy_performance(self,sltp_part):
        data = self.data
        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp(sltp_part)
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy] = self.generate_trading_signal(data,strategy,chromomosome_stop_loss,chromomosome_take_profit)  
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
            print(last_date)
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
        groups = np.array_split(x,self.K)
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

    def normalisation(self,trading_strategies_data):
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
            print(tsp)
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
        normalised_ts_data = ts_data
        profit =self.getProfit(normalised_ts_data,chromosome)
        corr = self.getCorrelation(ts_data,chromosome)
        gb = self.groupBalance(chromosome)
        wb = self.weightBalance(chromosome)
        try:
            fitness = profit * (1/corr) * np.power(gb,2) * wb
        except ZeroDivisionError:
            fitness = profit  * np.power(gb,2) * wb
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
    def best_chromosomes(self,population,N,q):
        q.put([i for i in sorted(population, key=lambda x: x[3])[-N:]])

    def select_best_chromosomes(self,population,N):
        return [i for i in sorted(population, key=lambda x: x[3])[-N:]]

    def get_global_best(self):
        best = self.best_individuals[0]
        for individual in self.best_individuals:
            if individual[3] > best[3]:
                best = individual
        return best

    def worst_chromosomes(self,population,N):
        worst = [i for i in sorted(population, key=lambda x: x[3])[:N]]
        
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

    def master_fitness_function(self,island,q):
        """Master slave migration"""
        for chromosome in island:
            ts_data = self.strategy_performance(chromosome[0])
            normalised_ts_data = self.normalisation(ts_data)
            profit =self.getProfit(normalised_ts_data,chromosome)
            corr = self.getCorrelation(ts_data,chromosome)
            gb = self.groupBalance(chromosome)
            wb = self.weightBalance(chromosome)
            fitness = profit * corr* np.power(gb,2) * wb
            chromosome[3] = fitness 
        q.put(island)

    def make_islands(self,population):
        """split list of  into islands for master island migration. thanks chatGPT"""
        # calculate the length of the list
        list_len = len(population)
        # calculate the size of each chunk
        chunk_size = list_len // self.num_islands
        # use slicing to split the list into chunks
        for i in range(0, list_len, chunk_size):
            yield population[i:i + chunk_size]


    def evolve_island_ring(self):
        """Ring Topology implementation"""
        self.re_init()
        #intiate population
        for i in range(self.num_islands):
            self.islands.append(self.init_population())
        #evolve
        for iteration in range(self.num_iter):
            processes = []
            result_queues = []
            for j in range(self.num_islands):
                result_queue = mp.Queue()
                process = mp.Process(target=self.genetic_operations, args=(self.islands[j],result_queue))
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            for process in processes:
                process.join()
            for j in range(self.num_islands):
                self.islands[j] = result_queues[j].get()
        #migration 
            if iteration % self.m_iter ==0:
                for j in range(self.num_islands):
                    processes = []
                    result_queues = []
                    self.islands[j]=self.worst_chromosomes(self.islands[j],self.N)
                    receive_individuals = []
                    for k in range(self.num_islands):
                        if k != j:
                            result_queue = mp.Queue()
                            process = mp.Process(target=self.best_chromosomes, args=(self.islands[k],self.N,result_queue))
                            process.start()
                            processes.append(process)
                            result_queues.append(result_queue)
                    for process in processes:
                        process.join()
                    for result in result_queues:
                        receive_individuals.extend(result.get())
                    for individual in receive_individuals:
                        self.islands[j].append(individual)

        # Return the best individual from each island
    
        processes = []
        result_queues = []
        for i in range(self.num_islands):
            result_queue =mp.Queue()
            process =mp.Process(target=self.best_chromosomes, args=(self.islands[i],1,result_queue))
            process.start()
            processes.append(process)
            result_queues.append(result_queue)
        for process in processes:
            process.join()
        for process in result_queues:
            self.best_individuals.append(process.get()[0])
        self.globalBest = self.get_global_best()

    def evolve_master_slave(self):
        """Master slave impelementation"""
        # Reinitiate evolution values
        self.re_init()
        population = self.init_population()
        self.globalBest = population[0]
        for j in range(self.num_iter):
            processes = []
            result_queues = []
            results = []
            islands = list(self.make_islands(population))
            for i in range(len(islands)):
                result_queue =mp.Queue()
                process =mp.Process(target=self.master_fitness_function, args=(islands[i],result_queue))
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            for process in processes:
                process.join()  
            for process in result_queues:
                results.append(process.get()[0])

            tempPopu  = self.selection(results)
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
            population = children
            for chromosone in population:
                    if chromosone[3] > self.globalBest[3]:
                        self.globalBest = chromosone


    ###########
    def lowest(self,accepted_pool,best):
        """lowest chromosome to replace"""
        lowest = accepted_pool[0]
        score = self.hamming_distance(lowest,best[0])
        for i in range(len(accepted_pool)):
            for j in range(len(best)):
                dist = self.hamming_distance(accepted_pool[i],best[j])
                if dist < score:
                    lowest=accepted_pool[i]
                    score = dist
        return lowest, score


    def multikuti_migration(self,population,immigrants,N,q):
        best = self.select_best_chromosomes(population,N) # pool to select the most different chromosomes from. 
        accepted_immigrants = immigrants # selected chromosomes which are most different chromosomes. 
        lowest_immigrant,dist_score = self.lowest(accepted_immigrants,best)
        for i in range(len(immigrants)):
            for j in range(i, len(best)):
                dist = self.hamming_distance(immigrants[i], best[j])
                if dist > dist_score:
                    if immigrants[i] not in accepted_immigrants:
                        accepted_immigrants.remove(lowest_immigrant)
                        accepted_immigrants.append(immigrants[i])
                        dist_score = dist
                        lowest_immigrant = immigrants[i]
                    else:
                        dist_score = dist
                        lowest_immigrant = immigrants[i]
        q.put(accepted_immigrants)

    def evolve_island_multikuti(self):
        """Multikuti  implementation"""
        # Reinitiate evolution values
        self.re_init()
        #intiate population
        for i in range(self.num_islands):
            self.islands.append(self.init_population())
        #evolve
        for iteration in range(self.num_iter):
            processes = []
            result_queues = []
            for j in range(self.num_islands):
                result_queue = mp.Queue()
                process = mp.Process(target=self.genetic_operations, args=(self.islands[j],result_queue))
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            for process in processes:
                process.join()
            for j in range(self.num_islands):
                self.islands[j] = result_queues[j].get()
        #migration 
            if iteration % self.m_iter ==0:
                for j in range(self.num_islands):
                    processes = []
                    result_queues = []
                    self.islands[j]=self.worst_chromosomes(self.islands[j],self.N)
                    receive_individuals = []
                    for k in range(self.num_islands):
                        if k != j:
                            immigrants = self.select_best_chromosomes(self.islands[k],self.N)
                            result_queue = mp.Queue()
                            process = mp.Process(target=self.multikuti_migration, args=(self.islands[j],immigrants,self.N,result_queue))
                            process.start()
                            processes.append(process)
                            result_queues.append(result_queue)
                    for process in processes:
                        process.join()
                    for result in result_queues:
                        receive_individuals.extend(result.get())
                    for individual in receive_individuals:
                        self.islands[j].append(individual)

        # Return the best individual from each island
    
        processes = []
        result_queues = []
        for i in range(self.num_islands):
            result_queue =mp.Queue()
            process =mp.Process(target=self.best_chromosomes, args=(self.islands[i],1,result_queue))
            process.start()
            processes.append(process)
            result_queues.append(result_queue)
        for process in processes:
            process.join()
        for process in result_queues:
            self.best_individuals.append(process.get()[0])
        self.globalBest = self.get_global_best()

        
    ########## Evaluation functions #################

    
    def evaluate_performance(self,data,sltp_part):
        chromomosome_stop_loss, chromomosome_take_profit = self.binary_to_sltp(sltp_part)
        strategy_performance = {}
        for strategy in self.strategies:
            strategy_performance[strategy] = self.generate_trading_signal(data,strategy,chromomosome_stop_loss,chromomosome_take_profit)  
        strategy_performance = pd.DataFrame.from_dict(strategy_performance)
        return strategy_performance
    
    def evaluate_best_profit(self,data):
        chromosome = self.globalBest
        p4mc = self.evaluate_performance(data,chromosome[0])
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

    