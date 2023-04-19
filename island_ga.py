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

    def __init__(self,data,strategies,num_islands=3,num_iter=150,pSize=100,m_iter=25,n_migrants=2,K=3,r_cross=0.001,r_mut=0.0001,r_inv=0.2,r_elite=0.8,n=8,b=8,stop_loss=-0.15,take_profit=0.15,allocated_capital=1000,selection_strategy="elit",evolve_strategy="ring"):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.num_islands = num_islands
        self.n_migrants = n_migrants
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
        self.selection_strategy = selection_strategy
        self.evolve_strategy = evolve_strategy
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
            chromosome = Chromosome(self.K,self.n_migrants,self.b,self.strategies,self.num_weight,self.stop_loss,self.take_profit)
            population.append(chromosome.create_chromosome())
        
        return population
 
    # Select parents

    def roulette_wheel_selection(self,population):

        def run_wheel(population):
    
            # Computes the totallity of the population fitness
            population_fitness = sum([chromosome.fitness_value for chromosome in population])
            # Computes for each chromosome the probability 
            chromosome_probabilities = [chromosome.fitness_value/population_fitness for chromosome in population]
            # Selects one chromosome based on the computed probabilities
            population = np.array(population,dtype=object)
            output = population[np.random.choice(population.shape[0],p=chromosome_probabilities)]
            
            return output #np.random.choice(population, p=chromosome_probabilities)
        selected = []
        for i in range(len(population)):
            selected.append(run_wheel(population))
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

    def genetic_operations_roul(self,population):
        """evolve each island per generation"""
        #population = self.update_pop_fitness_values(population)
        tempPopu  = self.roulette_wheel_selection(population)
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
    
    def genetic_operations_elite(self,population):
        """uses the elitism approach for selection"""
        n = len(population)
        vg= sum([chromosome.fitness_value for chromosome in population ])/len(population)
        #print(f" Before operations -Island {j} : average fitness = {vg}")
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
        vg= sum([chromosome.fitness_value for chromosome in elite_pop ])/len(elite_pop)
        #print(f" After operations -Island {j} : average fitness = {vg}")
        return elite_pop
    
    def genetic_operations(self,population):
        """run genetic operators based on selection strategy"""
        if self.selection_strategy == "roul":
            return self.genetic_operations_roul(population)
        else:
            return self.genetic_operations_elite(population)
        

    

    def parallel_genetic_operations(self,island,q):
        """evolve each island per generation"""  
        island=self.update_pop_fitness_values(island)
        island =self.genetic_operations(island)
        q.put(island)

    def master_fitness_function(self,island,q):
        """Master slave migration"""
        island=self.update_pop_fitness_values(island)
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
        


    def migrate_nearest(self, left_island_index, right_island_index):
        """Perform migration among the islands using the nearest neighbor strategy."""
        # Compute genetic distance between chromosomes in islands i and j
        distances = []
        for i, ind_i in enumerate(self.islands[left_island_index]):
            for j, ind_j in enumerate(self.islands[right_island_index]):
                distance = ind_i.hamming_distance(ind_j)
                distances.append((i, j, distance))

        # Sort the pairs by genetic distance in ascending order
        #distances.sort(key=lambda x: x[2], reverse=True) 
        distances.sort(key=lambda x: x[2])

        # Select the top n_migrants individuals from island i with the highest genetic distance
        migrants = set()
        for pair in distances:
            i, j, distance = pair
            if len(migrants) == self.n_migrants:
                break
            if i not in migrants:
                migrants.add(i)

        # Select the corresponding individuals from island j with the highest genetic distance and add them to island j
        for i in migrants:
            ind_i = self.islands[left_island_index][i]
            best_j = -1
            best_distance = -1
            for j, ind_j in enumerate(self.islands[right_island_index]):
                if j in migrants:
                    continue
                distance = ind_i.hamming_distance(ind_j)
                if distance > best_distance:
                    best_j = j
                    best_distance = distance
            if best_j != -1:
                ind_j = self.islands[right_island_index][best_j]
                self.islands[left_island_index].remove(ind_i)
                self.islands[right_island_index].remove(ind_j)
                self.islands[left_island_index].append(ind_j)
                self.islands[right_island_index].append(ind_i)




    
    def multikuti_migration(self, left_island_index, right_island_index):
        """Perform migration among the islands in a ring topology."""
        # Compute genetic distance between chromosomes in islands i and j
        distances = []
        for i, ind_i in enumerate(self.islands[left_island_index]):
            for j, ind_j in enumerate(self.islands[right_island_index]):
                distance = ind_i.hamming_distance(ind_j)
                distances.append((i, j, distance))

        # Sort the pairs by genetic distance in descending order
        distances.sort(key=lambda x: x[2], reverse=True) 
        #distances.sort(key=lambda x: x[2])

        # Select the top n_migrants individuals from island i with the highest genetic distance
        migrants = set()
        for pair in distances:
            i, j, distance = pair
            if len(migrants) == self.n_migrants:
                break
            if i not in migrants:
                migrants.add(i)

        # Select the corresponding individuals from island j with the highest genetic distance and add them to island j
        for i in migrants:
            ind_i = self.islands[left_island_index][i]
            best_j = -1
            best_distance = -1
            for j, ind_j in enumerate(self.islands[right_island_index]):
                if j in migrants:
                    continue
                distance = ind_i.hamming_distance(ind_j)
                if distance > best_distance:
                    best_j = j
                    best_distance = distance
            if best_j != -1:
                ind_j = self.islands[right_island_index][best_j]
                self.islands[left_island_index].remove(ind_i)
                self.islands[right_island_index].remove(ind_j)
                self.islands[left_island_index].append(ind_j)
                self.islands[right_island_index].append(ind_i)


    def migrate_ring(self, left_island_index, right_island_index):
        """Perform migration among the islands in a ring topology."""
        left_queue = []
        right_queue = []
        
        # Select the best individuals to migrate to the left
        left_migrants = self.select_best_chromosomes(self.islands[left_island_index], self.n_migrants)
        left_queue.extend(left_migrants)
        
        # Select the best individuals to migrate to the right
        right_migrants = self.select_best_chromosomes(self.islands[right_island_index], self.n_migrants)
        right_queue.extend(right_migrants)
        
        # Send and receive migrants
        for _ in range(self.n_migrants):
            # Send an individual to the left
            if left_queue:
                ind = left_queue.pop()
                self.islands[left_island_index].remove(ind)
                self.islands[right_island_index].append(ind)
            
            # Send an individual to the right
            if right_queue:
                ind = right_queue.pop()
                self.islands[right_island_index].remove(ind)
                self.islands[left_island_index].append(ind)


    def evolve_parallel(self):
        """Ring Topology implementation"""
        print(f"Running {self.evolve_strategy}")
        self.re_init()
        #intiate population
        self.population = self.init_population()
        self.islands = list(self.make_islands(self.population))
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
            for i in range(len(result_queues)):
                pop  = result_queues[i].get()
                self.islands[i] = pop  
            
            if iteration % self.m_iter ==0:
                # Perform migration among islands in a ring topology
                for i in range(self.num_islands):
                    left_island_index = (i - 1) % self.num_islands
                    right_island_index = (i + 1) % self.num_islands
                    print(f"Island {i} Migration -Left island {left_island_index} -Right island {right_island_index}")
                    if self.evolve_strategy =="ring":
                        self.migrate_ring(left_island_index, right_island_index)
                    if self.evolve_strategy == "multikuti":
                        self.multikuti_migration(left_island_index, right_island_index)
                    if self.evolve_strategy == "nearest":
                        self.migrate_nearest(left_island_index, right_island_index)
                    best = heapq.nlargest(1, self.islands[i], key=lambda x: x.fitness_value)[0]
                    #print(f"Island {i} - Generation {iteration}: Best fitness = {best.fitness_value}")
            # update population 
            self.population =   self.islands
            for i  in range(len(self.islands)):
                pop = self.islands[i]
            # add average fitness value to fitness  values to find convergence values
            self.get_convergence()  
            self.get_global_best()
            print(f"Generation {iteration}: Best fitness = {self.globalBest.fitness_value}")
        # Return the best individual from each island
        self.get_global_best()
        

    def evolve_master_slave(self):
        """Master slave impelementation"""
        print(f"Running {self.evolve_strategy}")
        # Reinitiate evolution values
        self.re_init()
        self.population = self.init_population()
        self.globalBest = self.population[0]
        for j in range(self.num_iter):
            processes = []
            result_queues = []
            children = []
            self.islands = list(self.make_islands(self.population))
        
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
            self.population = self.genetic_operations(children)
            self.globalBest = heapq.nlargest(1, self.population, key=lambda x: x.fitness_value)[0]
            # Calculate average fitness value for convergence
            self.islands = []
            self.get_convergence()
            self.get_global_best()
            print(f"Generation {j}: Best fitness = {self.globalBest.fitness_value}")
            
                        
    def evolve_gga(self):
        """GGA impelementation"""
        print(f"Running {self.evolve_strategy}")
        # Reinitiate evolution values
        self.re_init()
        self.population = self.init_population()
        self.globalBest = self.population[0]
        for j in range(self.num_iter):
            self.population = self.update_pop_fitness_values(self.population)
            self.population = self.genetic_operations(self.population)
            self.globalBest = heapq.nlargest(1, self.population, key=lambda x: x.fitness_value)[0]
            # Calculate average fitness value for convergence
            self.get_convergence()
            self.get_global_best()
            print(f"Generation {j}: Best fitness = {self.globalBest.fitness_value}")
            

    def evolve(self):
        """evolve based on strategy"""
        if self.evolve_strategy == "master_slave":
            self.evolve_master_slave()
        elif self.evolve_strategy == "gga":
            self.evolve_gga()
        else:
            self.evolve_parallel()
############# Elite selection Implemenation
class EliteIslandGGA(IslandGGA):

    #def __init__(self, *args, **kwargs):
        #self.r_elite = r_elite
        #super().__init__(*args, **kwargs)
     def __init__(self,data,strategies,num_islands,num_iter,pSize,m_iter,n_migrants,K,r_cross,r_mut,r_inv,r_elite,n,b,stop_loss,take_profit,allocated_capital,selection_strategy):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.num_islands = num_islands
        self.n_migrants = n_migrants
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
        self.selection_strategy = selection_strategy
    

    