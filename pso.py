import random
import numpy as np

def fitness_function(position):
    # Define your fitness function here
    return position[0] ** 2 + position[1] ** 2

def particle_swarm_optimization(swarm_size, num_iterations, dimensions, bounds):
    # Initialize the particle's position and velocity
    particles = np.zeros((swarm_size, dimensions))
    velocity = np.zeros((swarm_size, dimensions))
    for i in range(swarm_size):
        for j in range(dimensions):
            particles[i][j] = random.uniform(bounds[j][0], bounds[j][1])
            velocity[i][j] = random.uniform(-abs(bounds[j][1] - bounds[j][0]), abs(bounds[j][1] - bounds[j][0]))

    # Initialize the best positions of the particles and the global best position
    pbest_position = particles.copy()
    gbest_position = np.zeros((1, dimensions))
    pbest_fitness_value = np.zeros(swarm_size) + float('inf')
    gbest_fitness_value = float('inf')

    for i in range(swarm_size):
        fitness_value = fitness_function(particles[i])
        if fitness_value < pbest_fitness_value[i]:
            pbest_fitness_value[i] = fitness_value
            pbest_position[i] = particles[i]
        if fitness_value < gbest_fitness_value:
            gbest_fitness_value = fitness_value
            gbest_position = particles[i]

    w = 0.5
    c1 = 2
    c2 = 2

    for i in range(num_iterations):
        for j in range(swarm_size):
            # Update the velocity of the particles
            velocity[j] = w * velocity[j] + c1 * random.uniform(0, 1) * (pbest_position[j] - particles[j]) + \
                          c2 * random.uniform(0, 1) * (gbest_position - particles[j])
            particles[j] = particles[j] + velocity[j]

            # Update the best positions of the particles and the global best position
            fitness_value = fitness_function(particles[j])
            if fitness_value < pbest_fitness_value[j]:
                pbest_fitness_value[j] = fitness_value
                pbest_position[j] = particles[j]
            if fitness_value < gbest_fitness_value:
                gbest_fitness_value = fitness_value
                gbest_position = particles[j]

    return gbest_position, gbest_fitness_value

if __name__ == '__main__':
    swarm_size = 100
    num_iterations = 1000
    dimensions = 2
    bounds = [(-10, 10), (-10, 10)]
    result = particle_swarm_optimization(swarm_size, num_iterations, dimensions, bounds)
 
