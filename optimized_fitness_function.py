import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import numba
from functools import lru_cache

class OptimizedSingleThreadFitness:
    def __init__(self, cache_size: int = 256):
        self.cache_size = cache_size
        self._fitness_cache = {}
        self._portfolio_cache = {}
        self.tsps = None
        
    def calculate_chromosome_fitness(self, monthly_returns: pd.DataFrame, allocated_capital: float) -> float:
        """
        Optimized single-thread fitness calculation for island GA
        """
        # 1. CACHE CHECK: Hash the chromosome configuration
        chromosome_hash = self._hash_chromosome()
        if chromosome_hash in self._fitness_cache:
            cached_result = self._fitness_cache[chromosome_hash]
            self.fitness_value = cached_result['fitness']
            self.profit = cached_result['profit']
            self.mdd = cached_result['mdd']
            self.gb = cached_result['gb']
            return self.fitness_value
        
        # 2. SMART PORTFOLIO GENERATION: Avoid full cartesian product
        portfolios = self._generate_portfolios_smart()
        
        # 3. ADAPTIVE SAMPLING: Scale sample size based on portfolio space
        sampled_portfolios = self._adaptive_sample(portfolios)
        
        # 4. VECTORIZED METRICS: Calculate all at once with NumPy
        profits, mdds = self._calculate_metrics_vectorized(sampled_portfolios, monthly_returns)
        
        # 5. FAST AGGREGATION: Use NumPy instead of pandas
        self.mdd = self._aggregate_mdd_fast(mdds, profits)
        self.profit = self._aggregate_profit_fast(profits, mdds)
        self.gb = self.groupBalance()
        
        # 6. CACHE RESULT
        fitness = self._calculate_fitness_score()
        self._cache_result(chromosome_hash, fitness, self.profit, self.mdd, self.gb)
        
        return fitness
    
    def _hash_chromosome(self) -> int:
        """Fast chromosome hashing for caching"""
        try:
            # Create a tuple representation of group_part for hashing
            if hasattr(self, 'group_part'):
                group_repr = tuple(
                    tuple(sorted(group)) if hasattr(group, '__iter__') and not isinstance(group, str)
                    else group for group in self.group_part
                )
                return hash(group_repr)
        except (TypeError, AttributeError):
            pass
        
        # Fallback: use object id (less cache-friendly but safe)
        return id(self)
    
    def _generate_portfolios_smart(self) -> List:
        """
        Generate portfolios efficiently without creating massive cartesian products
        """
        if not hasattr(self, 'group_part') or not self.group_part:
            return []
        
        # Calculate total combinations
        total_combinations = 1
        for group in self.group_part:
            group_size = len(group) if hasattr(group, '__len__') else 1
            total_combinations *= group_size
            
            # Early exit if space is too large
            if total_combinations > 50000:
                return self._sample_from_large_space(max_samples=200)
        
        # For manageable spaces, generate systematically
        if total_combinations <= 1000:
            import itertools
            return list(itertools.product(*self.group_part))
        else:
            # For medium spaces, use strategic sampling
            return self._sample_from_medium_space(target_samples=min(500, total_combinations // 10))
    
    def _sample_from_large_space(self, max_samples: int) -> List:
        """Sample from very large portfolio spaces"""
        portfolios = []
        seen = set()
        max_attempts = max_samples * 3
        
        for _ in range(max_attempts):
            if len(portfolios) >= max_samples:
                break
                
            # Random selection from each group
            portfolio = tuple(
                np.random.choice(group) if hasattr(group, '__len__') and len(group) > 0
                else group for group in self.group_part
            )
            
            if portfolio not in seen:
                portfolios.append(portfolio)
                seen.add(portfolio)
        
        return portfolios
    
    def _sample_from_medium_space(self, target_samples: int) -> List:
        """Systematic sampling from medium-sized spaces"""
        import itertools
        
        # Use systematic sampling with some randomness
        portfolios = list(itertools.islice(
            itertools.product(*self.group_part), 
            target_samples * 2
        ))
        
        if len(portfolios) <= target_samples:
            return portfolios
        
        # Systematic sampling with random offset
        step = len(portfolios) // target_samples
        offset = np.random.randint(0, step)
        indices = range(offset, len(portfolios), step)[:target_samples]
        
        return [portfolios[i] for i in indices]
    
    def _adaptive_sample(self, portfolios: List, base_sample_size: int = 20) -> List:
        """
        Adaptive sampling based on portfolio space size
        """
        if not portfolios:
            return []
        
        n_portfolios = len(portfolios)
        
        # Scale sample size based on total portfolios
        if n_portfolios <= 50:
            sample_size = n_portfolios  # Use all for small spaces
        elif n_portfolios <= 200:
            sample_size = min(50, n_portfolios)  # Use up to 50 for medium spaces
        else:
            # For large spaces, use adaptive sizing
            sample_size = min(100, max(base_sample_size, int(np.sqrt(n_portfolios))))
        
        if n_portfolios <= sample_size:
            return portfolios
        
        # Use numpy for faster sampling
        indices = np.random.choice(n_portfolios, sample_size, replace=False)
        return [portfolios[i] for i in sorted(indices)]
    
    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _profit_calculation_core(portfolio_weights: np.ndarray, returns: np.ndarray) -> float:
        """Numba-optimized core profit calculation - customize for your logic"""
        # Placeholder - replace with your actual profit calculation
        weighted_returns = np.dot(returns, portfolio_weights)
        return np.sum(weighted_returns)
    
    @staticmethod
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _mdd_calculation_core(portfolio_weights: np.ndarray, returns: np.ndarray) -> float:
        """Numba-optimized core MDD calculation - customize for your logic"""
        # Placeholder - replace with your actual MDD calculation
        portfolio_returns = np.dot(returns, portfolio_weights)
        cumsum = np.cumsum(portfolio_returns)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = (running_max - cumsum) / np.maximum(running_max, 1e-8)
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    def _calculate_metrics_vectorized(self, portfolios: List, monthly_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast vectorized calculation without internal parallelization
        """
        if not portfolios:
            return np.array([]), np.array([])
        
        n_portfolios = len(portfolios)
        profits = np.zeros(n_portfolios, dtype=np.float64)
        mdds = np.zeros(n_portfolios, dtype=np.float64)
        
        # Convert monthly_returns to ensure it's a contiguous array
        returns_array = np.ascontiguousarray(monthly_returns, dtype=np.float64)
        
        # Process portfolios in batches for better cache performance
        batch_size = min(50, n_portfolios)
        
        for i in range(0, n_portfolios, batch_size):
            end_idx = min(i + batch_size, n_portfolios)
            batch_portfolios = portfolios[i:end_idx]
            
            for j, portfolio in enumerate(batch_portfolios):
                try:
                    # Convert portfolio to numpy array
                    if isinstance(portfolio, (list, tuple)):
                        portfolio_array = np.array(portfolio, dtype=np.float64)
                    else:
                        # Fallback to original methods
                        profits[i + j] = self.calculate_profit(portfolio, monthly_returns)
                        mdds[i + j] = self.calculate_mdd(portfolio, monthly_returns)
                        continue
                    
                    # Use numba-optimized calculations
                    profits[i + j] = self._profit_calculation_core(portfolio_array, returns_array)
                    mdds[i + j] = self._mdd_calculation_core(portfolio_array, returns_array)
                    
                except (ValueError, TypeError):
                    # Fallback to original methods for complex portfolio structures
                    profits[i + j] = self.calculate_profit(portfolio, monthly_returns)
                    mdds[i + j] = self.calculate_mdd(portfolio, monthly_returns)
        
        return profits, mdds
    
    def _aggregate_mdd_fast(self, mdds: np.ndarray, profits: np.ndarray) -> float:
        """Fast MDD aggregation using NumPy"""
        if len(mdds) == 0:
            return 0.0
        
        # Weight by positive profits, fallback to equal weights
        positive_profits = np.maximum(profits, 0)
        total_weight = np.sum(positive_profits)
        
        if total_weight > 1e-8:
            return float(np.average(mdds, weights=positive_profits))
        else:
            return float(np.mean(mdds))
    
    def _aggregate_profit_fast(self, profits: np.ndarray, mdds: np.ndarray) -> float:
        """Fast profit aggregation using NumPy"""
        if len(profits) == 0:
            return 0.0
        
        # Simple mean - can be customized based on your aggregation strategy
        return float(np.mean(profits))
    
    def _calculate_fitness_score(self) -> float:
        """Calculate final fitness score"""
        if self.mdd > 0.01:
            fitness = self.profit / self.mdd
        else:
            fitness = self.profit
        
        self.fitness_value = fitness
        return fitness
    
    def _cache_result(self, chromosome_hash: int, fitness: float, profit: float, mdd: float, gb: float):
        """Cache the calculated result"""
        self._fitness_cache[chromosome_hash] = {
            'fitness': fitness,
            'profit': profit,
            'mdd': mdd,
            'gb': gb
        }
        
        # Manage cache size using simple FIFO
        if len(self._fitness_cache) > self.cache_size:
            # Remove oldest 25% of entries
            items_to_remove = len(self._fitness_cache) - int(self.cache_size * 0.75)
            keys_to_remove = list(self._fitness_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                self._fitness_cache.pop(key, None)


# Usage: Replace your existing method with this optimized version
# Make sure to implement _profit_calculation_core and _mdd_calculation_core 
# with your actual profit and MDD calculation logic