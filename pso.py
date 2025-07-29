import numpy as np
import pandas as pd

class PortfolioPSO:
    def __init__(self, returns_df, num_assets_to_select=None, num_particles=50, iterations=100, 
                 inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5):
        self.returns_df = returns_df
        self.n_assets = returns_df.shape[1]
        self.k = num_assets_to_select if num_assets_to_select else self.n_assets
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = inertia_weight  # Inertia weight
        self.c1 = cognitive_weight  # Cognitive weight
        self.c2 = social_weight  # Social weight
        
        # Initialize particles
        self.particles = self._initialize_particles()
        self.velocities = np.zeros((num_particles, self.n_assets * 2))
        
        # Initialize best positions
        self.personal_best_positions = self.particles.copy()
        self.personal_best_fitness = np.array([self._fitness_function(p) for p in self.particles])
        
        # Initialize global best
        self.global_best_idx = np.argmax(self.personal_best_fitness)
        self.global_best_position = self.personal_best_positions[self.global_best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[self.global_best_idx]
        
        # Store results
        self.fitness_history = []
        self.best_portfolio = None
        self.best_weights = None

    def _initialize_particles(self):
        particles = np.zeros((self.num_particles, self.n_assets * 2))
        
        for i in range(self.num_particles):
            # Initialize selection part
            selection = np.zeros(self.n_assets)
            selected_indices = np.random.choice(range(self.n_assets), self.k, replace=False)
            selection[selected_indices] = 1
            
            # Initialize weights part
            weights = np.random.uniform(0, 1, self.n_assets)
            weights[selection == 0] = 0
            weights[selection == 1] /= weights[selection == 1].sum()
            
            # Combine selection and weights
            particles[i] = np.concatenate((selection, weights))
            
        return particles

    def _decode_particle(self, particle):
        selection = particle[:self.n_assets].astype(int)
        weights = particle[self.n_assets:]
        return selection, weights

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

    def _fitness_function(self, particle):
        selection, weights = self._decode_particle(particle)
        selected_assets = self.returns_df.columns[selection == 1]
        selected_weights = weights[np.where(selection == 1)[0]]
        
        # Ensure non-negative weights
        selected_weights = np.maximum(selected_weights, 0)
        
        # Normalize weights to sum to 1
        total_weight = np.sum(selected_weights)
        if total_weight > 0:
            selected_weights /= total_weight
        else:
            selected_weights = np.ones_like(selected_weights) / len(selected_weights)
        
        # Compute portfolio returns
        selected_returns = self.returns_df[selected_assets].dot(selected_weights).dropna()
        
        # Calculate Sharpe ratio
        return self.calculate_sharpe_ratio(selected_returns)

    def _fix_particle(self, particle):
        selection = particle[:self.n_assets].astype(int)
        
        # Fix selection part
        selection = np.round(selection)  # Convert to binary
        while selection.sum() > self.k:
            selection[np.random.choice(np.where(selection == 1)[0])] = 0
        while selection.sum() < self.k:
            selection[np.random.choice(np.where(selection == 0)[0])] = 1
            
        # Fix weights part
        weights = particle[self.n_assets:]
        weights = np.maximum(weights, 0)  # Ensure non-negative weights
        weights[selection == 0] = 0  # Zero weights for non-selected assets
        
        # Normalize weights for selected assets
        selected_sum = weights[selection == 1].sum()
        if selected_sum > 0:
            weights[selection == 1] /= selected_sum
            
        return np.concatenate((selection, weights))

    def update_particle(self, particle_idx):
        # Update velocity
        r1, r2 = np.random.rand(2)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.particles[particle_idx])
        social_component = self.c2 * r2 * (self.global_best_position - self.particles[particle_idx])
        self.velocities[particle_idx] = (self.w * self.velocities[particle_idx] + 
                                       cognitive_component + social_component)
        
        # Update position
        self.particles[particle_idx] += self.velocities[particle_idx]
        
        # Fix particle to maintain constraints
        self.particles[particle_idx] = self._fix_particle(self.particles[particle_idx])
        
        # Update personal best
        current_fitness = self._fitness_function(self.particles[particle_idx])
        if current_fitness > self.personal_best_fitness[particle_idx]:
            self.personal_best_positions[particle_idx] = self.particles[particle_idx].copy()
            self.personal_best_fitness[particle_idx] = current_fitness
            
            # Update global best
            if current_fitness > self.global_best_fitness:
                self.global_best_position = self.particles[particle_idx].copy()
                self.global_best_fitness = current_fitness

    def run(self):
        for iteration in range(self.iterations):
            # Update all particles
            for i in range(self.num_particles):
                self.update_particle(i)
            
            # Store average fitness for this iteration
            current_fitness = np.mean([self._fitness_function(p) for p in self.particles])
            self.fitness_history.append(current_fitness)
        
        # Get final best portfolio
        selection, weights = self._decode_particle(self.global_best_position)
        self.best_portfolio = self.returns_df.columns[selection == 1].tolist()
        self.best_weights = weights[selection == 1].tolist()
        
        return self.best_portfolio, self.best_weights, self.fitness_history