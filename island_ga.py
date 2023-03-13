import numpy as np
import random
import itertools
import math
import talib as ta
import pandas as pd
import multiprocess as mp
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
from chromosome import Chromosome


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
        self.convergence_values = []

    def re_init(self):
        self.islands = []
        self.best_individuals = []
        self.globalBest  = []
        self.fitness_values = []

    def init_population(self):
        population =[]
        for i in range(self.pSize):
            chromosome = Chromosome(self.K,self.n,self.b,self.strategies,self.num_weight,self.stop_loss,self.take_profit)
            population.append(chromosome.create_chromosome())
        
        return population
 
    # Select parents

    def roulette_wheel_selection(self,population):
    
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness_value for chromosome in population])
        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome.fitness_value/population_fitness for chromosome in population]
        # Selects one chromosome based on the computed probabilities
        population = np.array(population,dtype=object)
        output = population[np.random.choice(population.shape[0],p=chromosome_probabilities)]
        
        return output #np.random.choice(population, p=chromosome_probabilities)

    def selection(self,population):
        selected = []
        for i in range(len(population)):
            selected.append(self.roulette_wheel_selection(population))
        return selected

  
    
########
    def get_convergence(self):
        # add new best fitness value to fitness  values to find convergence values
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
        self.convergence_values.append(self.globalBest)

######## MIGRATION

    def best_chromosomes(self,population,N,q):
        q.put([i for i in sorted(population, key=lambda x: x.fitness_value)[-N:]])

    def select_best_chromosomes(self,population,N):
        return [i for i in sorted(population, key=lambda x: x.fitness_value)[-N:]]

    def get_global_best(self):
        best = self.best_individuals[0]
        for individual in self.best_individuals:
            if individual.fitness_value > best.fitness_value:
                best = individual
        return best

    def worst_chromosomes(self,population,N):
        worst = [i for i in sorted(population, key=lambda x: x.fitness_value)[:N]]
        
        for i in worst:
            population.remove(i)
        return population
    
    def update_pop_fitness_values(self,island):
        for chromosome in island:
            chromosome.calculate_chromosome_fitness(self.data,self.allocated_capital)
            if chromosome.fitness_value == 0:
                island.remove(chromosome)
        chromosome = Chromosome(self.K,self.n,self.b,self.strategies,self.num_weight,self.stop_loss,self.take_profit)
        chromosome.create_chromosome()
        chromosome.calculate_chromosome_fitness(self.data,self.allocated_capital)
        island.append(chromosome.create_chromosome())
        return island

    def genetic_operations(self,island,q):
        """evolve each island per generation"""
        island = self.update_pop_fitness_values(island)
        tempPopu  = self.selection(island)
        children = []
        #Crossover
        for i in range(0, len(tempPopu)-1, 2):
                # get selected parents in pairs
                parent1,parent2 = tempPopu[i],tempPopu[i+1]
                #crossover and mutation and inversion 
                child1,child2 = parent1.crossover(parent2,self.r_cross)
                child1.mutation(self.r_mut)
                child2.mutation(self.r_mut)
                child1.inversion(self.r_inv)
                child2.inversion(self.r_inv)
                children.append(child1)
                children.append(child2)
        island = children
        q.put(island)   

    def master_fitness_function(self,island,q):
        """Master slave migration"""
        island = self.update_pop_fitness_values(island)
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
            # add new best fitness value to fitness  values to find convergence values
            self.get_convergence()
                

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
                    child1,child2 = parent1.crossover(parent2,self.r_cross)
                    child1.mutation(self.r_mut)
                    child2.mutation(self.r_mut)
                    child1.inversion(self.r_inv)
                    child2.inversion(self.r_inv)
                    children.append(child1)
                    children.append(child2)
            population = children
            for chromosone in population:
                    if chromosone.fitness_value > self.globalBest.fitness_value:
                        self.globalBest = chromosone
            self.fitness_values.append(self.globalBest)
                        


    ###########
    def lowest(self,accepted_pool,best):
        """lowest chromosome to replace"""
        lowest = accepted_pool[0]
        score = lowest.hamming_distance(best[0])
        for i in range(len(accepted_pool)):
            for j in range(len(best)):
                dist = accepted_pool[i].hamming_distance(best[j])
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
                dist = immigrants[i].hamming_distance(best[j])
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
            self.get_convergence()

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

     

    