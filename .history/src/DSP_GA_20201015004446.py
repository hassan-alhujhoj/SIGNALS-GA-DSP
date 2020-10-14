"""
Genetic Algorithms for Digital Signal Processing
"""
import numpy as np

""" 
cal_pop_fitness:    Calculating the population fitness for Digital Signal Processing
                    The Signal to Noise Ratio (SNR) was used as the fitness function, with higher signal to noise ratios giving better results
"""
def cal_pop_fitness(Waveform, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = np.empty((pop.shape[0], 1))
    for i in range(1, pop.shape[0]):
        fitness[i, 0] = Waveform.PM(pop[i])
    return fitness


""" 
select_mating_pool: Choose the mating pool for the parent genomes
                    The parents are returned
"""
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        # If taking num_parents from the previous iteration, 
        # the best are taken. The fitness of others is set to be low so 
        # the next best can be found 
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


""" 
crossover:          The crossover function swaps the genes of the parents
                    to find more optimal genes. 
"""
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)  # Gene Swapping here

    #Swaps the genes of the caried parents
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


""" 
mutation:           Mutation adds random variation to genes to find 
                    other gene combinations
"""
def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


""" 
create_population:  The create_population function initialises the population data
                    with appropriate data values. 
"""
def create_population(pop_size):
    f1 = np.random.uniform(low=30, high=50, size=(pop_size[0], 1))
    f2 = np.random.uniform(low=50, high=100, size=(pop_size[0], 1))
#    TW = np.random.uniform(low=0,  high=5,   size=(pop_size[0], 1))
#    BW = np.random.uniform(low=5,  high=10,  size=(pop_size[0], 1))
    return np.column_stack((f1, f2))