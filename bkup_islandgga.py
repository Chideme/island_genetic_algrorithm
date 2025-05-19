import numpy as np
import random
import itertools
import math
import talib as ta
import heapq
import pandas as pd
import multiprocess as mp
import matplotlib.pyplot as plt
from datetime import datetime,date, timedelta
from chromosome import Chromosome


class IslandGGA():

    def __init__(self,num_islands,num_iter,data,strategies,pSize,m_iter,N,K,r_cross,r_mut,r_inv,r_elite,n,b,stop_loss,take_profit,allocated_capital):
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
        self.r_elite = r_elite
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
        self.population = []

    def re_init(self):
        self.islands = []
        self.best_individuals = []
        self.globalBest  = []
        self.convergence_values = []
        self.population = []

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



    
############## get convergence
    def get_convergence(self):
        # get average fitness value in each island
        # Calculate average fitness value for convergnece
        
        """Calculate the convergence value for each island."""
        
        if self.islands:
            convergence = []
            for island in self.islands:
                island_average_fitness = sum([chromosome.fitness_value for chromosome in island ])/len(island)
                convergence.append(island_average_fitness)
            self.convergence_values.append(convergence)
        else:
            average_fitness = sum([chromosome.fitness_value for chromosome in self.population ])/len(self.population)
            self.convergence_values.append(average_fitness)
    
######## MIGRATION


    def best_chromosomes(self, population, N, q):
        best = heapq.nlargest(N, population, key=lambda x: x.fitness_value)
        q.put(best)


    def select_best_chromosomes(self,population,N):
        best = heapq.nlargest(N, population, key=lambda x: x.fitness_value)
        return best
        #return [i for i in sorted(population, key=lambda x: x.fitness_value,reverse=True)[:N]]


    def worst_chromosomes(self,population,N,q):
        #worst = [i for i in sorted(population, key=lambda x: x.fitness_value,reverse=False)[:N]]
        worst = heapq.nsmallest(N, population, key=lambda x: x.fitness_value)
        
        for chromosome in worst:
            population.remove(chromosome)
        q.put(population)
    

    #def worst_chromosomes(self, population, N, q):
      #  worst = heapq.nsmallest(N, population, key=lambda x: x.fitness_value)
       # keep = [indiv for indiv in population if indiv not in worst]
       # q.put(keep)

    
    def update_pop_fitness_values(self,island):
        
        for chromosome in island:
            chromosome.calculate_chromosome_fitness(self.data,self.allocated_capital)
        max_fitness_chromosome = heapq.nlargest(1, island, key=lambda x: x.fitness_value)[0]
        min_fitness_chromosome = heapq.nsmallest(1, island, key=lambda x: x.fitness_value)[0]
        max_fitness = max_fitness_chromosome.fitness_value
        min_fitness = min_fitness_chromosome.fitness_value
        #for chromosome in island:
         #   chromosome.scale_fitness(max_fitness,min_fitness)
        return island

    def genetic_operations(self,population):
        """evolve each island per generation"""
        print("this function")
        #population = self.update_pop_fitness_values(population)
        tempPopu  = self.selection(population)
        #elite_pop = self.select_best_chromosomes(tempPopu,elit_size)
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

        return children
    
    

    def parallel_genetic_operations(self,island,q):
        """evolve each island per generation"""  
        island = self.update_pop_fitness_values(island)
        island = self.genetic_operations(island)
        q.put(island)

    def master_fitness_function(self,island,q):
        #pool = mp.Pool(processes=len(self.islands))
        # iterate over the islands and apply the master fitness function in parallel
        #results_ = [pool.apply_async(self.update_pop_fitness_values, args=(island,)) for island in self.islands]
        # wait for all processes to finish and retrieve the results
        #results= [result.get() for result in results_]
        #pool.close()
        #pool.join()
        """Master slave migration"""
        self.update_pop_fitness_values(island)
        q.put(island)

    def make_islands(self,population):
        """split list of  into islands for  island migration. thanks ChatGPT ：）"""
        # calculate the length of the list
        list_len = len(population)
        # calculate the size of each chunk
        chunk_size = list_len // self.num_islands
        # use slicing to split the list into chunks
        for i in range(0, list_len, chunk_size):
            yield population[i:i + chunk_size]

    def get_global_best(self):
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
        for result in result_queues:
            self.best_individuals.append(result.get()[0])
        self.globalBest = heapq.nlargest(1, self.best_individuals, key=lambda x: x.fitness_value)[0]
        
            
    ########### multikuti helper functions
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

    def migrate_ring(self, left_island_index, right_island_index):
        """Perform migration among the islands in a ring topology."""
        left_queue = mp.Queue()
        right_queue = mp.Queue()
        
        # Select the best individuals to migrate to the left
        left_migrants = self.select_best_chromosomes(self.islands[left_island_index], self.n_migrants)
        for ind in left_migrants:
            left_queue.put(ind)
        
        # Select the best individuals to migrate to the right
        right_migrants = self.select_best_chromosomes(self.islands[right_island_index], self.n_migrants)
        for ind in right_migrants:
            right_queue.put(ind)
        
        # Send and receive migrants
        # Define the migration function
        def migrate(island1, island2, queue1, queue2):
            for _ in range(self.n_migrants):
                # Send an individual to the left
                if not queue1.empty():
                    ind = queue1.get()
                    island1.remove(ind)
                    island2.append(ind)

                # Send an individual to the right
                if not queue2.empty():
                    ind = queue2.get()
                    island2.remove(ind)
                    island1.append(ind)
            # Create and start the migration processes
            left_process = mp.Process(target=migrate, args=(self.islands[left_island_index], self.islands[right_island_index], left_queue, right_queue))
            right_process = mp.Process(target=migrate, args=(self.islands[right_island_index], self.islands[left_island_index], right_queue, left_queue))
            left_process.start()
            right_process.start()
            
            # Wait for the processes to finish
            left_process.join()
            right_process.join()

    def evolve_island_ring(self):
        """Ring Topology implementation"""
        self.re_init()
        #intiate population
        population = self.init_population()
        self.islands = list(self.make_islands(population))
        #evolve
        for iteration in range(self.num_iter):
            processes = []
            result_queues = []
            for j in range(len(self.islands)):
                result_queue = mp.Queue()
                process = mp.Process(target=self.parallel_genetic_operations, args=(self.islands[j],result_queue))
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            for process in processes:
                process.join()
            for j in range(self.num_islands):
                self.islands[j] = result_queues[j].get()  
        #migration 
            if iteration % self.m_iter ==0:
                # Perform migration among islands in a ring topology
                processes = []
                for i in range(self.num_islands):
                    left_island_index = (i - 1) % self.num_islands
                    right_island_index = (i + 1) % self.num_islands
                    self.migrate_ring(left_island_index, right_island_index)
                
                 ## Get best chromosome   
                for j in range(self.num_islands):
                    processes = []
                    result_queues = []
                    #self.islands[j]=self.worst_chromosomes(self.islands[j],self.N)
                    receive_individuals = []
                    for k in range(self.num_islands):
                        if k != j:
                            result_queue = mp.Queue()
                            process = mp.Process(target=self.best_chromosomes, args=(self.islands[k],self.N,result_queue))
                            process.start()
                            processes.append(process)
                            result_queues.append(result_queue)
                            break
                    for process in processes:
                        process.join()
                    for result in result_queues:
                        receive_individuals.extend(result.get())
                    self.islands[j].extend(receive_individuals)
                    
                    #for individual in receive_individuals:
                       # self.islands[j].append(individual)
                # delete worst chromosomes
                processes = []
                result_queues = []
                for i in range(self.num_islands):
                    result_queue =mp.Queue()
                    process =mp.Process(target=self.worst_chromosomes, args=(self.islands[i],self.N,result_queue))
                    process.start()
                    processes.append(process)
                    result_queues.append(result_queue)
                for process in processes:
                    process.join()
                for i in range(self.num_islands):
                    self.islands[i]=result_queues[i].get()
            # update population 
            self.population =   []
            for i in range(len(self.islands)):
                self.population.extend(self.islands[i])

            # add average fitness value to fitness  values to find convergence values
            self.get_convergence()
        # Return the best individual from each island
        self.get_global_best()
        
    

    def evolve_master_slave(self):
        """Master slave impelementation"""
        # Reinitiate evolution values
        self.re_init()
        population = self.init_population()
        self.globalBest = population[0]
        for j in range(self.num_iter):
            processes = []
            result_queues = []
            children = []
            self.islands = list(self.make_islands(population))
        
            for i in range(len(self.islands)):
                result_queue =mp.Queue()
                process =mp.Process(target=self.master_fitness_function, args=(self.islands[i],result_queue))
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            for process in processes:
                process.join()  
            for result in result_queues:
                children.extend(result.get())
            population = self.genetic_operations(children)
            for chromosone in population:
                    if chromosone.fitness_value > self.globalBest.fitness_value:
                        self.globalBest = chromosone
            # Calculate average fitness value for convergence
            self.population = population
            self.islands = []
            self.get_convergence()
            
                        
    def evolve_gga(self):
        """GGA impelementation"""
        # Reinitiate evolution values
        self.re_init()
        population = self.init_population()
        self.globalBest = population[0]
        for j in range(self.num_iter):
            population = self.update_pop_fitness_values(population)
            population = self.genetic_operations(population)
            self.globalBest = heapq.nlargest(1, population, key=lambda x: x.fitness_value)[0]
            # Calculate average fitness value for convergence
            self.population = population
            self.get_convergence()
            

    
    def evolve_island_multikuti(self):
        """Multikuti  implementation"""
        # Reinitiate evolution values
        self.re_init()
        #intiate population
        population = self.init_population()
        self.islands = list(self.make_islands(population))
        #evolve
        for iteration in range(self.num_iter):
            # perform genetic operations
            processes = []
            result_queues = []
            for j in range(self.num_islands):
                result_queue = mp.Queue()
                process = mp.Process(target=self.parallel_genetic_operations, args=(self.islands[j],result_queue))
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            for process in processes:
                process.join()
            for j in range(self.num_islands):
                self.islands[j] = result_queues[j].get()
        #migration 
            if iteration % self.m_iter ==0:
                print("iteration")
                # delete worst chromosomes
                processes = []
                result_queues = []
                for i in range(self.num_islands):
                    result_queue =mp.Queue()
                    process =mp.Process(target=self.worst_chromosomes, args=(self.islands[i],self.N,result_queue))
                    process.start()
                    processes.append(process)
                    result_queues.append(result_queue)
                for process in processes:
                    process.join()
                for i in range(self.num_islands):
                    self.islands[i]=result_queues[i].get()
                #migrate best chromosome
                for j in range(self.num_islands):
                    processes = []
                    result_queues = []
                    #self.islands[j]=self.worst_chromosomes(self.islands[j],self.N)
                    receive_individuals = []
                    for k in range(self.num_islands):
                        if k != j:
                            immigrants = self.select_best_chromosomes(self.islands[k],self.N)
                            result_queue = mp.Queue()
                            process = mp.Process(target=self.multikuti_migration, args=(self.islands[j],immigrants,self.N,result_queue))
                            process.start()
                            processes.append(process)
                            result_queues.append(result_queue)
                            break
                    for process in processes:
                        process.join()
                    for result in result_queues:
                        receive_individuals.extend(result.get())
                    for individual in receive_individuals:
                        self.islands[j].append(individual)
            self.population =   []
            for i in range(len(self.islands)):
                self.population.extend(self.islands[i])
            self.get_convergence()

        # Return the best individual from each island
        self.get_global_best()

############# Elite selection Implemenation
class EliteIslandGGA(IslandGGA):

    #def __init__(self, *args, **kwargs):
        #self.r_elite = r_elite
        #super().__init__(*args, **kwargs)

    def genetic_operations(self,population):
        """uses the elitism approach for selection"""
        n = len(population)
        elit_size = math.ceil(self.r_elite * n)
        elite_pop = self.select_best_chromosomes(population,elit_size)
        children = []
        for _ in range(n - elit_size):
            # get selected parents in pairs
            parent1, parent2 = random.choices(elite_pop, k=2)
            #crossover and mutation and inversion 
            child1,child2 = parent1.crossover(parent2,self.r_cross)
            child1.mutation(self.r_mut)
            child1.inversion(self.r_inv)
            children.append(child1)
        for child in children:
            elite_pop.append(child)
        return elite_pop


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
    