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
from copy import deepcopy
import time
import traceback
from contextlib import contextmanager

class IslandGGA():

    def __init__(self,data,strategies,num_islands=8,num_iter=50,pSize=100,m_iter=5,n_migrants_rate=0.5,K=4,r_cross=0.6,r_mut=0.01,r_inv=0.1,r_elite=0.5,n=8,b=8,stop_loss=-0.15,take_profit=0.15,allocated_capital=1,selection_strategy="elit",evolve_strategy="ring"):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.num_islands = num_islands
        #self.n_migrants_rate = n_migrants_rate
        self.n_migrants =math.ceil(n_migrants_rate * (pSize // num_islands))
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
        self.convergence_times = []
        self.island_convergence = []
        self.population = []
        self.diversity_data = []

    def re_init(self):
        self.islands = []
        self.best_individuals = []
        self.globalBest  = []
        self.convergence_values = []
        self.convergence_times = []
        self.population = []

    def init_population(self):
        population =[]
        for i in range(self.pSize):
            chromosome = Chromosome(self.K,self.n,self.b,self.strategies,self.num_weight,self.stop_loss,self.take_profit)
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
                island_average_fitness = sum([chromosome.fitness_value for chromosome in island ])/len(island)  if len(island) > 0 else 0           
                convergence.append(island_average_fitness)
            #self.island_convergence.append(convergence)
            self.convergence_values.append(np.mean(convergence))
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


    def worst_chromosomes(self,population,N):
        #worst = [i for i in sorted(population, key=lambda x: x.fitness_value,reverse=False)[:N]]
        worst = heapq.nsmallest(N, population, key=lambda x: x.fitness_value)
        
        return worst
    

    #def worst_chromosomes(self, population, N, q):
      #  worst = heapq.nsmallest(N, population, key=lambda x: x.fitness_value)
       # keep = [indiv for indiv in population if indiv not in worst]
       # q.put(keep)

    
    def update_pop_fitness_values(self,island):
           
        for chromosome in island:
            chromosome.calculate_chromosome_fitness(self.data,self.allocated_capital)
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
            child2.mutation(self.r_mut)
            child2.inversion(self.r_inv)
            child1.calculate_chromosome_fitness(self.data,self.allocated_capital)
            child2.calculate_chromosome_fitness(self.data,self.allocated_capital)
            if child2.fitness_value > child1.fitness_value:
                children.append(child2)
            else:
                children.append(child1)
        for child in children:
            elite_pop.append(child)
        
        return elite_pop
    
    def genetic_operations(self,population):
        """run genetic operators based on selection strategy"""
        if self.selection_strategy == "roul":
            return self.genetic_operations_roul(population)
        else:
            return self.genetic_operations_elite(population)
        


    def operations(self, island,q):
        """evolve each island per generation"""
       
        island = self.genetic_operations(island)
        island = self.update_pop_fitness_values(island)

        
        q.put(island)

    
    @contextmanager
    def cleanup_processes(self, processes, queues):
        """Context manager to ensure proper cleanup"""
        try:
            yield processes, queues
        finally:
            # Force cleanup of all processes and queues
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                process.close()
            
            # Clear queues to prevent memory leaks
            for queue in queues:
                try:
                    while not queue.empty():
                        queue.get_nowait()
                except:
                    pass
                queue.close()
                queue.join_thread()

    def parallel_genetic_operations(self):
        """
        IMPROVED: Evolve each island per generation with proper cleanup
        """
        if not self.islands:
            print("Warning: No islands to process")
            return
            
        processes = []
        result_queues = []
        
        try:
            # Create processes and queues
            for j in range(len(self.islands)):
                result_queue = mp.Queue()
                # Deep copy the island to avoid shared state issues
                island_copy = deepcopy(self.islands[j])
                process = mp.Process(
                    target=self.safe_operations, 
                    args=(island_copy, result_queue, j)
                )
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            
            # Wait for all processes with timeout
            results = [None] * len(self.islands)
            completed = [False] * len(processes)
            
            for i, process in enumerate(processes):
                try:
                    process.join(timeout=30)  # 30 second timeout per process
                    if process.is_alive():
                        print(f"Warning: Process {i} timed out, terminating...")
                        process.terminate()
                        process.join(timeout=5)
                    completed[i] = not process.is_alive()
                except Exception as e:
                    print(f"Error joining process {i}: {e}")
            
            # Collect results from successful processes
            for j in range(len(self.islands)):
                if completed[j]:
                    try:
                        if not result_queues[j].empty():
                            results[j] = result_queues[j].get_nowait()
                        else:
                            print(f"Warning: No result from process {j}")
                            results[j] = self.islands[j]  # Keep original
                    except Exception as e:
                        print(f"Error getting result from queue {j}: {e}")
                        results[j] = self.islands[j]  # Keep original
                else:
                    print(f"Process {j} failed, keeping original island")
                    results[j] = self.islands[j]  # Keep original
            
            # Update islands with results
            for j, result in enumerate(results):
                if result is not None:
                    self.islands[j] = result
                    
        except Exception as e:
            print(f"Critical error in parallel operations: {e}")
            traceback.print_exc()
        
        finally:
            # CRITICAL: Cleanup all processes and queues
            with self.cleanup_processes(processes, result_queues):
                pass

    def safe_operations(self, island, result_queue, island_id):
        """
        IMPROVED: Safe wrapper for operations with error handling
        """
        try:
            # Process the island
            self.operations(island,result_queue)
            #evolved_island = self.operations(island,result_queue)
            #result_queue.put(evolved_island)
            
        except Exception as e:
            print(f"Error in island {island_id} operations: {e}")
            # Put original island back if operation fails
            result_queue.put(island)
        
        finally:
            # Ensure queue is marked as done
            try:
                result_queue.close()
            except:
                pass

    def master_fitness_function(self):
        """
        IMPROVED: Master-slave fitness calculation with proper cleanup
        """
        if not self.population:
            print("Warning: No population to process")
            return []
            
        self.islands = list(self.make_islands(self.population))
        processes = []
        result_queues = []
        
        try:
            # Create processes
            for i in range(len(self.islands)):
                result_queue = mp.Queue()
                # Deep copy to avoid shared state
                island_copy = deepcopy(self.islands[i])
                process = mp.Process(
                    target=self.safe_fitness_fun, 
                    args=(island_copy, result_queue, i)
                )
                process.start()
                processes.append(process)
                result_queues.append(result_queue)
            
            # Wait for completion with timeout
            children = []
            for i, process in enumerate(processes):
                try:
                    process.join(timeout=60)  # 60 second timeout
                    if process.is_alive():
                        print(f"Fitness process {i} timed out, terminating...")
                        process.terminate()
                        process.join(timeout=5)
                    
                    # Get result if available
                    if not result_queues[i].empty():
                        result = result_queues[i].get_nowait()
                        children.extend(result)
                    else:
                        print(f"Warning: No fitness result from process {i}")
                        
                except Exception as e:
                    print(f"Error in fitness process {i}: {e}")
            
            return children
            
        except Exception as e:
            print(f"Critical error in master fitness function: {e}")
            traceback.print_exc()
            return []
        
        finally:
            # CRITICAL: Cleanup
            with self.cleanup_processes(processes, result_queues):
                pass

    def safe_fitness_fun(self, island, result_queue, island_id):
        """
        IMPROVED: Safe fitness function with error handling
        """
        try:
            # Calculate fitness for the island
            self.update_pop_fitness_values(island)
            result_queue.put(island)
            
        except Exception as e:
            print(f"Error calculating fitness for island {island_id}: {e}")
            result_queue.put(island)  # Return original island
        
        finally:
            try:
                result_queue.close()
            except:
                pass

    def reset_multiprocessing(self):
        """
        CRITICAL: Call this between runs to reset multiprocessing state
        """
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset any class-level multiprocessing state
            if hasattr(self, '_mp_context'):
                delattr(self, '_mp_context')
                
        except Exception as e:
            print(f"Warning during multiprocessing reset: {e}")

    def parallel_genetic_operations1(self):
        """evolve each island per generation"""  
        #evolve
        processes = []
        result_queues = []
        for j in range(len(self.islands)):
            result_queue = mp.Queue()
            process = mp.Process(target=self.operations, args=(self.islands[j],result_queue))
            process.start()
            processes.append(process)
            result_queues.append(result_queue)
        for process in processes:
            process.join()
        for j in range(len(self.islands)):
            self.islands[j] = result_queues[j].get() 

    def fitness_fun1(self,island,q):
        """Master slave migration"""
        self.update_pop_fitness_values(island)
        q.put(island) 

    def master_fitness_function1(self):
        """Master slave migration""" 
        self.islands = list(self.make_islands(self.population))
        processes = []
        result_queues = []
        children = []
        for i in range(len(self.islands)):
            result_queue =mp.Queue()
            process =mp.Process(target=self.fitness_fun, args=(self.islands[i],result_queue))
            process.start()
            processes.append(process)
            result_queues.append(result_queue)
        for process in processes:
            process.join()
        for result in result_queues:
            children.extend(result.get())
        return children

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
        best = heapq.nlargest(1, self.population, key=lambda x: x.fitness_value)[0]
        if best.fitness_value > self.globalBest.fitness_value:
                self.globalBest = deepcopy(best)

    def population_diversity(self, island):
        """Calculate average Hamming distance between all chromosome pairs in an island."""
        if len(island) <= 1:
            return 0.0
        
        total_distance = 0
        count = 0
        for i in range(len(island)):
            for j in range(i+1, len(island)):
                # Use the chromosome's hamming_distance method
                total_distance += island[i].hamming_distance(island[j])
                count += 1
        
        return total_distance / count

    
    def track_diversity(self):
        """Track diversity for the entire population (all islands combined)."""
        # Combine all chromosomes from all islands into one population
        all_chromosomes = []
        if self.islands:
            for island in self.islands:
                all_chromosomes.extend(island)
            
            # Calculate diversity for the entire population
            diversity = self.population_diversity(all_chromosomes)
        else:
            diversity =self.population_diversity(self.population)
        
        return diversity

# Usage in your main loop:
# diversity_data = []
# for generation in range(max_generations):
#     # Your evolution logic here
#     diversities = self.track_diversity()
#     diversity_data.append(diversities)

        


    def migrate_nearest1(self, left_island_index, right_island_index):
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



    def migrate_nearest(self, left_island_index, right_island_index):
        """Perform migration among the islands using the nearest neighbor strategy."""
        # Compute genetic distance between chromosomes in islands i and j
        distances = []
        for i, ind_i in enumerate(self.islands[left_island_index]):
            for j, ind_j in enumerate(self.islands[right_island_index]):
                distance = ind_i.hamming_distance(ind_j)
                distances.append((i, j, distance))

        # Sort by distance - use ascending for "nearest" (most similar)
        distances.sort(key=lambda x: x[2])
        
        # Select pairs for migration (closest matches)
        migrated_left = set()
        migrated_right = set()
        migrations = []
        
        for i, j, distance in distances:
            if len(migrations) >= self.n_migrants:
                break
            if i not in migrated_left and j not in migrated_right:
                migrations.append((i, j))
                migrated_left.add(i)
                migrated_right.add(j)
        
        # Perform the actual swaps
        for i, j in migrations:
            ind_left = self.islands[left_island_index][i]
            ind_right = self.islands[right_island_index][j]
            
            # Swap individuals
            self.islands[left_island_index][i] = ind_right
            self.islands[right_island_index][j] = ind_left


    def migrate_random(self):
        """Each island sends migrants to a randomly selected island with replacement."""
        # Create migration plan to avoid conflicts
        migration_plan = []
        
        for i in range(self.num_islands):
            # Select target island
            target = random.choice([j for j in range(self.num_islands) if j != i])
            
            # Select migrants from source
            migrants_indices = random.sample(range(len(self.islands[i])), 
                                        min(self.n_migrants, len(self.islands[i])))
            
            # Select individuals to replace in target
            replace_indices = random.sample(range(len(self.islands[target])), 
                                        min(self.n_migrants, len(self.islands[target])))
            
            migration_plan.append((i, target, migrants_indices, replace_indices))
        
        # Execute migrations (swap individuals)
        for source, target, migrant_idx, replace_idx in migration_plan:
            for m_idx, r_idx in zip(migrant_idx, replace_idx):
                # Swap individuals between islands
                migrant = self.islands[source][m_idx]
                replaced = self.islands[target][r_idx]
                
                self.islands[source][m_idx] = replaced
                self.islands[target][r_idx] = migrant


    def migrate_star(self):
        """Perform star topology migration with island 0 as the hub."""
        hub = 0
        
        # Collect all migrants that will go to hub
        hub_incoming = []
        spoke_replacements = {}  # Store what each spoke will receive
        
        # Phase 1: Collect migrants from spokes
        for i in range(1, self.num_islands):
            # Select best migrants from spoke
            migrants = self.select_best_chromosomes(self.islands[i], self.n_migrants)
            hub_incoming.extend(migrants)
            
            # Select what hub will send to this spoke
            hub_migrants = self.select_best_chromosomes(self.islands[hub], self.n_migrants)
            spoke_replacements[i] = hub_migrants
        
        # Phase 2: Execute the exchanges
        # Remove migrants from spokes and add hub's contributions
        for i in range(1, self.num_islands):
            spoke_migrants = self.select_best_chromosomes(self.islands[i], self.n_migrants)
            hub_contributions = spoke_replacements[i]
            
            # Remove spoke's migrants
            for ind in spoke_migrants:
                if ind in self.islands[i]:
                    self.islands[i].remove(ind)
            
            # Add hub's contributions to spoke
            self.islands[i].extend(hub_contributions)
        
        # Remove hub's contributions from hub and add spoke migrants
        for i in range(1, self.num_islands):
            hub_contributions = spoke_replacements[i]
            for ind in hub_contributions:
                if ind in self.islands[hub]:
                    self.islands[hub].remove(ind)
        
        # Add all incoming migrants to hub
        self.islands[hub].extend(hub_incoming)

    def migrate_fully_connected(self):
        """Each island exchanges migrants with all other islands."""
        # Create migration plan to avoid conflicts
        migration_plan = {}
        
        # Plan all migrations first
        for i in range(self.num_islands):
            migrants = self.select_best_chromosomes(self.islands[i], 
                                                self.n_migrants * (self.num_islands - 1))
            migration_plan[i] = migrants
        
        # Execute migrations - each island receives from all others
        for target in range(self.num_islands):
            # Remove migrants that this island will send out
            outgoing = migration_plan[target]
            for ind in outgoing:
                if ind in self.islands[target]:
                    self.islands[target].remove(ind)
            
            # Add incoming migrants from all other islands
            migrants_per_source = self.n_migrants
            for source in range(self.num_islands):
                if source != target:
                    # Take migrants from the planned migrants of source island
                    source_migrants = migration_plan[source]
                    start_idx = target * migrants_per_source if target < source else (target - 1) * migrants_per_source
                    end_idx = start_idx + migrants_per_source
                    
                    incoming = source_migrants[start_idx:end_idx]
                    self.islands[target].extend(incoming)


    # Alternative simpler approach for migrate_random:
    def migrate_random_simple(self):
        """Simple random migration - each island swaps individuals with a random partner."""
        for i in range(self.num_islands):
            # Pick a random partner
            partner = random.choice([j for j in range(self.num_islands) if j != i])
            
            # Skip if this pair was already processed
            if i > partner:  # Only process each pair once
                continue
                
            # Select individuals to swap
            i_migrants = random.sample(range(len(self.islands[i])), 
                                    min(self.n_migrants, len(self.islands[i])))
            p_migrants = random.sample(range(len(self.islands[partner])), 
                                    min(self.n_migrants, len(self.islands[partner])))
            
            # Perform swaps
            for idx_i, idx_p in zip(i_migrants, p_migrants):
                self.islands[i][idx_i], self.islands[partner][idx_p] = \
                    self.islands[partner][idx_p], self.islands[i][idx_i]
    
    def multikuti_migration(self, left_island_index, right_island_index):
        """Perform migration among the islands ."""
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
        #left_worst = self.worst_chromosomes(self.islands[left_island_index], self.n_migrants)
        left_queue.extend(left_migrants)
        
        # Select the best individuals to migrate to the right
        right_migrants = self.select_best_chromosomes(self.islands[right_island_index], self.n_migrants)
        #right_worst =  self.worst_chromosomes(self.islands[right_island_index], self.n_migrants)
        right_queue.extend(right_migrants)
        
        # Send and receive migrants
        for _ in range(self.n_migrants):
            # Send an individual to the left
            #if right_worst:
              #  w = right_worst.pop()
               # self.islands[right_island_index].remove(w)
            if left_queue:
                ind = left_queue.pop()
                self.islands[left_island_index].remove(ind)
                self.islands[right_island_index].append(ind)
            
            
            # Send an individual to the right
            if right_queue:
               ind = right_queue.pop()
               self.islands[right_island_index].remove(ind)
               self.islands[left_island_index].append(ind)

    def migration1(self):
        """Perform island migrations"""
        # Perform migration among islands in a ring topology
        for i in range(self.num_islands):
            left_island_index = (i - 1) % self.num_islands
            right_island_index = (i + 1) % self.num_islands
            
            print(f"Island {i} Migration -Left island {left_island_index} -Right island {right_island_index}")
            if self.evolve_strategy =="ring":
                self.migrate_ring(left_island_index, right_island_index)
            elif self.evolve_strategy == "multikuti":
                self.multikuti_migration(left_island_index, right_island_index)
            elif self.evolve_strategy == "nearest":
                self.migrate_nearest(left_island_index, right_island_index)
            #best = heapq.nlargest(1, self.islands[i], key=lambda x: x.fitness_value)[0]
            #print(f"Island {i} - Generation {iteration}: Best fitness = {best.fitness_value}")


    def migration(self):
        """Perform island migrations"""
        for i in range(self.num_islands):
            left_island_index = (i - 1) % self.num_islands
            right_island_index = (i + 1) % self.num_islands

            print(f"Island {i} Migration - Left {left_island_index} - Right {right_island_index}")

            if self.evolve_strategy == "ring":
                for i in range(self.num_islands):
                    self.migrate_ring(left_island_index, right_island_index)
            elif self.evolve_strategy == "multikuti":
                self.multikuti_migration(left_island_index, right_island_index)
            elif self.evolve_strategy == "nearest":
                self.migrate_nearest(left_island_index, right_island_index)
            elif self.evolve_strategy == "star":
                self.migrate_star()
            elif self.evolve_strategy == "random":
                self.migrate_random()
            elif self.evolve_strategy == "fully_connected":
                self.migrate_fully_connected()



    def evolve_parallel(self):
        """island implementation"""
        print(f"Running {self.evolve_strategy}")
        self.re_init()
        start_time = time.time()
        #intiate population
        self.population = self.init_population()
        self.update_pop_fitness_values(self.population)
        self.globalBest= deepcopy(heapq.nlargest(1, self.population, key=lambda x: x.fitness_value)[0])
        self.islands = list(self.make_islands(self.population))
        #evolve
        for iteration in range(self.num_iter):
            self.parallel_genetic_operations()
            if iteration % self.m_iter ==0:
                if iteration != 0:
                    if self.n_migrants != 0:
                        self.migration()
            # update population 
            self.population=[]
            for island in self.islands:
                self.population.extend(island)
            self.get_convergence()
            self.convergence_times.append(time.time() - start_time)
            self.get_global_best()
            diversities = self.track_diversity()
            self.diversity_data.append(diversities)
            print(f"Generation {iteration +1}: Best fitness = {self.globalBest.fitness_value}  Average Fitness = {self.convergence_values[-1]}")

        # Return the best individual from each island
        self.get_global_best()
        

    def evolve_master_slave(self):
        """Master slave impelementation"""
        print(f"Running {self.evolve_strategy}")
        # Reinitiate evolution values
        self.re_init()
        start_time = time.time()
        self.population = self.init_population()
        self.globalBest = deepcopy(self.population[0])
        self.population = self.master_fitness_function()
        for iteration in range(self.num_iter):
            self.population = self.genetic_operations(self.population)
            self.population = self.master_fitness_function()
            # Calculate average fitness value for convergence
            self.islands =[]
            self.get_convergence()
            self.convergence_times.append(time.time() - start_time)
            self.get_global_best()
            diversities = self.track_diversity()
            self.diversity_data.append(diversities)
            print(f"Generation {iteration +1}: Best fitness = {self.globalBest.fitness_value}  Average Fitness = {self.convergence_values[-1]}")
            

    def evolve_gga(self):
        """GGA impelementation"""
        print(f"Running {self.evolve_strategy}")
        # Reinitiate evolution values
        self.re_init()
        start_time = time.time()
        self.population = self.init_population()
        self.globalBest = deepcopy(self.population[0])
        self.population = self.update_pop_fitness_values(self.population)
        for iteration in range(self.num_iter): 
            self.population = self.genetic_operations(self.population)
            self.population = self.update_pop_fitness_values(self.population)
            # Calculate average fitness value for convergence
            self.islands = []
            self.get_convergence()
            self.convergence_times.append(time.time() - start_time)
            self.get_global_best()
            diversities = self.track_diversity()
            self.diversity_data.append(diversities)
            print(f"Generation {iteration +1}: Best fitness = {self.globalBest.fitness_value}  Average Fitness = {self.convergence_values[-1]}")

    def evolve(self):
        
        """evolve based on strategy"""
        if self.evolve_strategy == "master_slave":
            self.evolve_master_slave()
        elif self.evolve_strategy == "gga":
            self.evolve_gga()
        else:
            self.evolve_parallel()
        
    

    