import numpy as np
import pandas as pd
import random
from datetime import datetime


def get_last_date_of_month(year, month):
    """Helper to find the last date of a month."""
    if month == 12:
        return datetime(year + 1, 1, 1) - pd.Timedelta(days=1)
    else:
        return datetime(year, month + 1, 1) - pd.Timedelta(days=1)

class PSOPortfolioOptimizer:
    def __init__(self, data, strategies, num_particles=30, iterations=30, 
                 inertia_weight=0.7, cognitive_weight=2.5, social_weight=2.5):
        self.data = data
        self.returns_df = data  # monthly returns
        self.strategies = strategies
        self.n = len(strategies)
        self.k = len(strategies)  # selects all by default; adjust for sparse
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = inertia_weight
        self.c1 = cognitive_weight
        self.c2 = social_weight

        self.particles = self._init_particles()
        self.velocities = np.zeros_like(self.particles)

        self.pbest = self.particles.copy()
        self.pbest_fitness = np.array([self._fitness_function(p) for p in self.particles])
        self.gbest_index = np.argmax(self.pbest_fitness)
        self.gbest = self.pbest[self.gbest_index].copy()
        self.gbest_fitness = self.pbest_fitness[self.gbest_index]

        self.fitness_history = []
        self.gbest_profit = 0
        self.gbest_mdd = 0

    def _init_particles(self):
        particles = []
        for _ in range(self.num_particles):
            selection = np.zeros(self.n)
            selected_idx = np.random.choice(self.n, self.k, replace=False)
            selection[selected_idx] = 1
            weights = np.random.rand(self.n)
            weights[selection == 0] = 0
            weights = weights / weights.sum()
            particle = np.concatenate((selection, weights))
            particles.append(particle)
        return np.array(particles)

    def _decode(self, particle):
        selection = particle[:self.n].round().astype(int)
        weights = particle[self.n:]
        weights[selection == 0] = 0
        if weights.sum() == 0:
            weights[selection == 1] = 1 / max(selection.sum(), 1)
        else:
            weights = weights / weights.sum()
        return selection, weights


    def _get_profit(self, returns):
        return (1 + returns).cumprod().iloc[-1] - 1

    def _get_mdd(self, returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (peak - cumulative)
        return drawdown.max()

    

    def _fix(self, particle):
        selection = particle[:self.n].round().astype(int)
        weights = particle[self.n:]
        # Fix selection size
        while selection.sum() > self.k:
            idx = np.random.choice(np.where(selection == 1)[0])
            selection[idx] = 0
        while selection.sum() < self.k:
            idx = np.random.choice(np.where(selection == 0)[0])
            selection[idx] = 1
        # Fix weights
        weights[selection == 0] = 0
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights[selection == 1] = 1 / self.k
        return np.concatenate([selection, weights])
    

    def _generate_trading_signal(self, data, strategy):
        monthly_returns = []
        monthly_return = 0
        monthly_freq = 0
        market_position = 'out'

        for row in data.itertuples(index=False):
            last_date = get_last_date_of_month(row.Date.year, row.Date.month)
            if row.Date == last_date:
                if market_position == 'in':
                    sell_price = row.close
                    trade_return = (sell_price - cost_price) / cost_price
                    monthly_return += trade_return
                    monthly_freq += 1
                    market_position = 'out'
                try:
                    avg_return = monthly_return / monthly_freq
                except ZeroDivisionError:
                    avg_return = 0
                monthly_returns.append(avg_return)
                monthly_return = 0
                monthly_freq = 0
            else:
                if market_position == 'out' and row._asdict()[strategy] == 1:
                    cost_price = row.close
                    market_position = 'in'
                elif market_position == 'in':
                    sell_price = row.close
                    trade_return = (sell_price - cost_price) / cost_price
                    if row._asdict()[strategy] == 0:
                        monthly_return += trade_return
                        monthly_freq += 1
                        market_position = 'out'

        return monthly_returns

    def _portfolio_monthly_returns(self, selection, weights):
        selected_strategies = [s for s, sel in zip(self.strategies, selection) if sel == 1]
        if not selected_strategies:
            return pd.Series([0])

        returns = {}
        for strategy in selected_strategies:
            returns[strategy] = self._generate_trading_signal(self.data, strategy)
        df = pd.DataFrame.from_dict(returns)
        # Align weight length to selected strategies
        selected_weights = weights[selection == 1]
        selected_weights = selected_weights / selected_weights.sum()
        portfolio_return = df.dot(selected_weights)
        return portfolio_return

    def _fitness_function(self, particle):
        selection, weights = self._decode(particle)
        portfolio_returns = self._portfolio_monthly_returns(selection, weights)
        profit = self._get_profit(portfolio_returns)
        mdd = self._get_mdd(portfolio_returns)
        return profit / mdd if mdd > 0.01 else profit



    def run(self):
        for _ in range(self.iterations):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (self.pbest[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = self._fix(self.particles[i])
                fit = self._fitness_function(self.particles[i])
                if fit > self.pbest_fitness[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_fitness[i] = fit
                    if fit > self.gbest_fitness:
                        self.gbest = self.particles[i].copy()
                        self.gbest_fitness = fit
            self.fitness_history.append(self.gbest_fitness)

        # Decode gbest and compute final metrics
        selection, weights = self._decode(self.gbest)
        returns = self._portfolio_monthly_returns(selection, weights)
        self.gbest_profit = self._get_profit(returns)
        self.gbest_mdd = self._get_mdd(returns)
        return self.gbest_profit, self.gbest_mdd, self.gbest_fitness

    def evaluate_on_data(self, data):
        selection, weights = self._decode(self.gbest)
        selected_strategies = [s for s, sel in zip(self.strategies, selection) if sel == 1]
        selected_weights = weights[selection == 1]

        if not selected_strategies:
            return 0, 0, 0

        # Normalize weights
        selected_weights /= selected_weights.sum()

        # Step 1: Compute monthly returns per selected strategy
        monthly_returns = {}
        for strategy in selected_strategies:
            monthly_returns[strategy] = self._generate_trading_signal(data, strategy)

        monthly_df = pd.DataFrame(monthly_returns)

        # Step 2: Compute portfolio returns
        portfolio_returns = monthly_df.dot(selected_weights)

        # Step 3: Compute metrics
        profit = self._get_profit(portfolio_returns)
        mdd = self._get_mdd(portfolio_returns)
        fitness = profit / mdd if mdd > 0.01 else profit
        return profit, mdd, fitness
