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
import psutil
import os

class IslandGGA():

    def __init__(self,data,strategies,num_islands=8,num_iter=50,pSize=100,m_iter=5,n_migrants_rate=0.5,K=4,r_cross=0.6,r_mut=0.01,r_inv=0.1,r_elite=0.2,n=8,b=8,stop_loss=-0.15,take_profit=0.15,selection_strategy="elit",evolve_strategy="ring", max_workers=None):
        self.data = data
        self.K = K
        self.pSize = pSize
        self.strategies = strategies
        self.num_islands = num_islands
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
        
        # CPU optimization parameters
        if max_workers is None:
            self.max_workers = max(1, min(mp.cpu_count() - 1, num_islands))
        else:
            self.max_workers = min(max_workers, mp.cpu_count(), num_islands)
        
        self.cpu_limit = 80  # Maximum CPU percentage
        self.process_pool = None
        print(f"Initialized with {self.max_workers} max workers for {num_islands} islands")

    def re_init(self):
        self.islands = []
        self.best_individuals = []
        self.globalBest  = []
        self.convergence_values = []
        self.convergence_times = []
        self.population = []
        # Clean up any existing processes
        self.cleanup_resources()

    def cleanup_resources(self):
        """Clean up multiprocessing resources"""
        try:
            if self.process_pool:
                self.process_pool.terminate()
                self.process_pool.join(timeout=2)
                self.process_pool = None
            
            import gc
            gc.collect()
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def init_population(self):
        population =[]
        for i in range(self.pSize):
            chromosome = Chromosome(self.K,self.n,self.b,self.strategies,self.num_weight,self.stop_loss,self.take_profit)
            population.append(chromosome.create_chromosome())
        
        return population
 
    def roulette_wheel_selection(self,population):
        def run_wheel(population):
            population_fitness = sum([chromosome.fitness_value for chromosome in population])
            if population_fitness == 0:
                return random.choice(population)
            chromosome_probabilities = [chromosome.fitness_value/population_fitness for chromosome in population]
            population = np.array(population,dtype=object)
            output = population[np.random.choice(population.shape[0],p=chromosome_probabilities)]
            return output

        selected = []
        for i in range(len(population)):
            selected.append(run_wheel(population))
        return selected

    def get_convergence(self):
        if self.islands:
            convergence = []
            for island in self.islands:
                island_average_fitness = sum([chromosome.fitness_value for chromosome in island ])/len(island)  if len(island) > 0 else 0           
                convergence.append(island_average_fitness)
            self.convergence_values.append(np.mean(convergence))
        else:
            average_fitness = sum([chromosome.fitness_value for chromosome in self.population ])/len(self.population)
            self.convergence_values.append(average_fitness)

    def select_best_chromosomes(self,population,N):
        best = heapq.nlargest(N, population, key=lambda x: x.fitness_value)
        return best

    def worst_chromosomes(self,population,N):
        worst = heapq.nsmallest(N, population, key=lambda x: x.fitness_value)
        return worst
    
    def update_pop_fitness_values(self,island):
        for chromosome in island:
            chromosome.calculate_chromosome_fitness(self.data,is_training=True)
        return island

    def genetic_operations_roul(self,population):
        tempPopu  = self.roulette_wheel_selection(population)
        children = []
        for i in range(0, len(tempPopu)-1, 2):
            parent1,parent2 = tempPopu[i],tempPopu[i+1]
            child1,child2 = parent1.crossover(parent2,self.r_cross)
            child1.mutation(self.r_mut)
            child2.mutation(self.r_mut)
            child1.inversion(self.r_inv)
            child2.inversion(self.r_inv)
            children.append(child1)
            children.append(child2)
        return children
    
    def genetic_operations_elite(self,population):
        n = len(population)
        elit_size = math.ceil(self.r_elite * n)
        elite_pop = self.select_best_chromosomes(population,elit_size)
        children = []
        for _ in range(n - elit_size):
            parent1, parent2 = random.choices(elite_pop, k=2)
            child1,child2 = parent1.crossover(parent2,self.r_cross)
            child1.mutation(self.r_mut)
            child1.inversion(self.r_inv)
            child2.mutation(self.r_mut)
            child2.inversion(self.r_inv)
            child1.calculate_chromosome_fitness(self.data,is_training=True)
            child2.calculate_chromosome_fitness(self.data,is_training=True)
            if child2.fitness_value > child1.fitness_value:
                children.append(child2)
            else:
                children.append(child1)
        for child in children:
            elite_pop.append(child)
        return elite_pop
    
    def genetic_operations(self,population):
        if self.selection_strategy == "roul":
            return self.genetic_operations_roul(population)
        else:
            return self.genetic_operations_elite(population)

    def monitor_cpu_usage(self):
        """Monitor current CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_limit:
                print(f"High CPU usage detected: {cpu_percent:.1f}%")
                time.sleep(0.2)  # Brief pause to let CPU cool down
            return cpu_percent
        except:
            return 0

    # OPTIMIZED: Use process pool instead of individual processes
    def parallel_genetic_operations_optimized(self):
        """CPU-optimized parallel genetic operations using process pool"""
        if not self.islands:
            print("Warning: No islands to process")
            return
            
        # Monitor CPU before starting
        self.monitor_cpu_usage()
        
        # Determine number of workers (don't exceed islands or max_workers)
        num_workers = min(len(self.islands), self.max_workers)
        
        try:
            # Use process pool for better resource management
            with mp.Pool(processes=num_workers) as pool:
                # Prepare tasks - deep copy islands to avoid shared state
                tasks = [(deepcopy(island), i) for i, island in enumerate(self.islands)]
                
                # Process islands in parallel with timeout
                results = pool.starmap_async(
                    self.safe_island_evolution, 
                    tasks
                ).get(timeout=60)  # 60 second total timeout
                
                # Update islands with results
                for i, result in enumerate(results):
                    if result is not None:
                        self.islands[i] = result
                    # else keep original island
                        
        except mp.TimeoutError:
            print("Warning: Parallel operations timed out")
        except Exception as e:
            print(f"Error in parallel operations: {e}")
            # Fall back to sequential processing
            self.sequential_genetic_operations()

    def safe_island_evolution(self, island, island_id):
        """Safe wrapper for evolving a single island"""
        try:
            # Monitor CPU within worker process
            if psutil.cpu_percent(interval=0.05) > 90:
                time.sleep(0.1)
            
            # Evolve the island
            evolved_island = self.genetic_operations(island)
            evolved_island = self.update_pop_fitness_values(evolved_island)
            return evolved_island
            
        except Exception as e:
            print(f"Error evolving island {island_id}: {e}")
            return island  # Return original island on error

    def sequential_genetic_operations(self):
        """Fallback sequential processing"""
        print("Falling back to sequential processing")
        for i in range(len(self.islands)):
            try:
                self.islands[i] = self.genetic_operations(self.islands[i])
                self.islands[i] = self.update_pop_fitness_values(self.islands[i])
                
                # Add small delay to prevent CPU overload
                if i % 2 == 0:
                    time.sleep(0.05)
                    
            except Exception as e:
                print(f"Error in sequential processing of island {i}: {e}")

    # OPTIMIZED: Batch processing for very large populations
    def batch_parallel_operations(self, batch_size=None):
        """Process islands in batches to limit CPU usage"""
        if not self.islands:
            return
            
        if batch_size is None:
            batch_size = self.max_workers
            
        print(f"Processing {len(self.islands)} islands in batches of {batch_size}")
        
        for i in range(0, len(self.islands), batch_size):
            batch_end = min(i + batch_size, len(self.islands))
            batch_islands = self.islands[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(self.islands)-1)//batch_size + 1}")
            
            # Process batch
            original_islands = self.islands
            self.islands = batch_islands
            self.parallel_genetic_operations_optimized()
            
            # Update original islands
            for j, evolved_island in enumerate(self.islands):
                original_islands[i + j] = evolved_island
            
            self.islands = original_islands
            
            # Brief pause between batches
            time.sleep(0.1)
            self.monitor_cpu_usage()

    # Keep your original method as fallback but with improvements
    def parallel_genetic_operations(self):
        """Choose between optimized or batch processing based on system load"""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 70 or memory_percent > 80:
            print(f"High system load (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%) - using batch processing")
            self.batch_parallel_operations(batch_size=max(1, self.max_workers // 2))
        else:
            self.parallel_genetic_operations_optimized()

    # OPTIMIZED: Master-slave with process pool
    def master_fitness_function_optimized(self):
        """CPU-optimized master-slave fitness calculation"""
        if not self.population:
            print("Warning: No population to process")
            return []
            
        self.islands = list(self.make_islands(self.population))
        
        # Monitor system resources
        self.monitor_cpu_usage()
        
        num_workers = min(len(self.islands), self.max_workers)
        
        try:
            with mp.Pool(processes=num_workers) as pool:
                # Process fitness calculations in parallel
                tasks = [(deepcopy(island), i) for i, island in enumerate(self.islands)]
                results = pool.starmap_async(
                    self.safe_fitness_calculation, 
                    tasks
                ).get(timeout=120)  # 2 minute timeout
                
                # Collect results
                children = []
                for result in results:
                    if result:
                        children.extend(result)
                        
                return children
                
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return self.sequential_fitness_calculation()

    def safe_fitness_calculation(self, island, island_id):
        """Safe fitness calculation for individual island"""
        try:
            # CPU check
            if psutil.cpu_percent(interval=0.05) > 90:
                time.sleep(0.1)
                
            updated_island = self.update_pop_fitness_values(island)
            return updated_island
            
        except Exception as e:
            print(f"Error calculating fitness for island {island_id}: {e}")
            return island

    def sequential_fitness_calculation(self):
        """Fallback sequential fitness calculation"""
        children = []
        for i, island in enumerate(self.islands):
            try:
                updated_island = self.update_pop_fitness_values(island)
                children.extend(updated_island)
                if i % 2 == 0:
                    time.sleep(0.05)
            except Exception as e:
                print(f"Error in sequential fitness calculation for island {i}: {e}")
                children.extend(island)
        return children

    # Use optimized version
    def master_fitness_function(self):
        return self.master_fitness_function_optimized()

    def make_islands(self,population):
        list_len = len(population)
        chunk_size = list_len // self.num_islands
        for i in range(0, list_len, chunk_size):
            yield population[i:i + chunk_size]
    
    def get_global_best(self):
        if self.population:
            best = heapq.nlargest(1, self.population, key=lambda x: x.fitness_value)[0]
            if not self.globalBest or best.fitness_value > self.globalBest.fitness_value:
                self.globalBest = deepcopy(best)

    def population_diversity(self, island):
        if len(island) <= 1:
            return 0.0
        
        total_distance = 0
        count = 0
        for i in range(len(island)):
            for j in range(i+1, len(island)):
                total_distance += island[i].hamming_distance(island[j])
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def track_diversity(self):
        all_chromosomes = []
        if self.islands:
            for island in self.islands:
                all_chromosomes.extend(island)
            diversity = self.population_diversity(all_chromosomes)
        else:
            diversity = self.population_diversity(self.population)
        
        return diversity

    # Migration methods (keeping your original implementations)
    def migrate_ring(self, left_island_index, right_island_index):
        left_queue = []
        right_queue = []
        
        left_migrants = self.select_best_chromosomes(self.islands[left_island_index], self.n_migrants)
        left_queue.extend(left_migrants)
        
        right_migrants = self.select_best_chromosomes(self.islands[right_island_index], self.n_migrants)
        right_queue.extend(right_migrants)
        
        for _ in range(self.n_migrants):
            if left_queue:
                ind = left_queue.pop()
                self.islands[left_island_index].remove(ind)
                self.islands[right_island_index].append(ind)
            
            if right_queue:
               ind = right_queue.pop()
               self.islands[right_island_index].remove(ind)
               self.islands[left_island_index].append(ind)

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

    def migration(self):
        """Migration with CPU monitoring"""
        self.monitor_cpu_usage()
        
        if self.evolve_strategy == "ring":
            for i in range(self.num_islands):
                left_island_index = (i - 1) % self.num_islands
                right_island_index = (i + 1) % self.num_islands
                self.migrate_ring(left_island_index, right_island_index)
        # Add other migration strategies as needed

    def evolve_parallel(self):
        """Optimized parallel evolution"""
        print(f"Running {self.evolve_strategy} with {self.max_workers} workers")
        self.re_init()
        start_time = time.time()
        
        # Initialize population
        self.population = self.init_population()
        self.update_pop_fitness_values(self.population)
        self.globalBest = deepcopy(heapq.nlargest(1, self.population, key=lambda x: x.fitness_value)[0])
        self.islands = list(self.make_islands(self.population))
        
        # Evolution loop
        for iteration in range(self.num_iter):
            # Monitor system resources periodically
            if iteration % 10 == 0:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                print(f"System: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
            
            # Evolve islands
            self.parallel_genetic_operations()
            
            # Migration
            if iteration % self.m_iter == 0 and iteration != 0 and self.n_migrants != 0:
                print("Performing migration")
                self.migration()
            
            # Update population
            self.population = []
            for island in self.islands:
                self.population.extend(island)
                
            self.get_convergence()
            self.convergence_times.append(time.time() - start_time)
            self.get_global_best()
            diversities = self.track_diversity()
            self.diversity_data.append(diversities)
            
            print(f"Generation {iteration +1}: Best fitness = {self.globalBest.fitness_value:.6f}  Average Fitness = {self.convergence_values[-1]:.6f}")
            
            # Brief pause to prevent system overload
            time.sleep(0.01)
        
        self.get_global_best()
        self.cleanup_resources()

    # Keep your other evolution methods
    def evolve_master_slave(self):
        print(f"Running master-slave with {self.max_workers} workers")
        self.re_init()
        start_time = time.time()
        self.population = self.init_population()
        self.globalBest = deepcopy(self.population[0]) if self.population else None
        self.population = self.master_fitness_function()
        
        for iteration in range(self.num_iter):
            self.population = self.genetic_operations(self.population)
            self.population = self.master_fitness_function()
            self.islands = []
            self.get_convergence()
            self.convergence_times.append(time.time() - start_time)
            self.get_global_best()
            diversities = self.track_diversity()
            self.diversity_data.append(diversities)
            print(f"Generation {iteration +1}: Best fitness = {self.globalBest.fitness_value:.6f}  Average Fitness = {self.convergence_values[-1]:.6f}")
            
            if iteration % 10 == 0:
                time.sleep(0.05)  # Periodic pause

    def evolve_gga(self):
        print("Running standard GGA")
        self.re_init()
        start_time = time.time()
        self.population = self.init_population()
        self.globalBest = deepcopy(self.population[0]) if self.population else None
        self.population = self.update_pop_fitness_values(self.population)
        
        for iteration in range(self.num_iter): 
            self.population = self.genetic_operations(self.population)
            self.population = self.update_pop_fitness_values(self.population)
            self.islands = []
            self.get_convergence()
            self.convergence_times.append(time.time() - start_time)
            self.get_global_best()
            diversities = self.track_diversity()
            self.diversity_data.append(diversities)
            print(f"Generation {iteration +1}: Best fitness = {self.globalBest.fitness_value:.6f}  Average Fitness = {self.convergence_values[-1]:.6f}")
            
            if iteration % 5 == 0:
                time.sleep(0.02)

    def evolve(self):
        """Main evolution method with cleanup"""
        try:
            if self.evolve_strategy == "master_slave":
                self.evolve_master_slave()
            elif self.evolve_strategy == "gga":
                self.evolve_gga()
            else:
                self.evolve_parallel()
        finally:
            # Ensure cleanup happens
            self.cleanup_resources()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup_resources()