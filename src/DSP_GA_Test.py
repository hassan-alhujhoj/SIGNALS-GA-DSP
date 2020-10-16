"""
Genetic Algorithms for Digital Signal Processing Test Case Generator
Created on Mon Oct 05 20:01:05 2020
Last Edited on  Mon Oct 12 2020 by Luke Trenberth
"""

import matplotlib.pyplot as plt
import csv, ast
from DSP_Genetic_Algorithms import GA_filter, DSP_Signal

def test_all_files():
    input_num = 2
    mating_parent_number = 3
    solutions_per_population = 10
    num_generations = 50
    with open("SNR.csv", mode="w") as SNR_file:
        SNR_writer = csv.writer(SNR_file, delimiter=',')
        # Conduct a Genetic Algorithm approximation
        best_outputs= []
        for i in range (1, 28):
            filename = "Signal_Files/ECG{}.txt".format(i)
            waveform = waveform = DSP_Signal(filename)
            best_soln, best_soln_fitness, best_output = GA_filter(waveform, 
                                                           input_num, solutions_per_population, 
                                                           mating_parent_number, num_generations)
            best_outputs.append(best_output)
            SNR_writer.writerow([i, best_output])
            print("File {} complete".format(i))
        
    
def plot_file_SNR():
    with open("SNR.csv", mode="r") as SNR_file:
        file_nums = []
        SNR_finals = []
        SNR_reader = csv.reader(SNR_file, delimiter=',')
        for line in SNR_reader:
            if len(line) > 0:
                i = int(line[0])
                file_nums.append(i)
                SNR = ast.literal_eval(line[1])
                SNR_finals.append(SNR[-1])
        plt.plot(file_nums, SNR_finals)
        plt.show()
        
        
def plot_num_generations():
    iters = []
    SNR = []
    
    input_num = 2
    mating_parent_number = 3
    solutions_per_population = 10
    
    with open("iters.csv", mode="w") as SNR_file:
        SNR_writer = csv.writer(SNR_file, delimiter=',')
        for num_generations in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
            filename = "Signal_Files/ECG15.txt"
            waveform = DSP_Signal(filename)
            best_soln, best_soln_fitness, best_output = GA_filter(waveform, 
                                                           input_num, solutions_per_population, 
                                                           mating_parent_number, num_generations)
            SNR_writer.writerow([num_generations, best_soln_fitness])
            print("{} finished".format(num_generations))

#plot_best_outputs()
#plot_num_generations()        
    

#The GA_filter function filters an input waveform 
def GA_filter(waveform, input_num, solutions_per_population, mating_parent_number, num_generations):
    

    # Defining the population size.
    pop_size = (solutions_per_population,input_num) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    #Creating the initial population.
    new_population = ga.create_population(pop_size)
    
    
    best_outputs = []
    for generation in range(num_generations):
        # Measuring the fitness of each chromosome in the population.
        fitness = ga.cal_pop_fitness(waveform, new_population)
        # The best result in the current iteration.
        best_outputs.append(np.max(fitness))
        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(new_population, fitness, 
                                          mating_parent_number)
        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], input_num))
        # Adding some variations to the offspring using mutation.
        offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        
#        if (generation < 20):
#            print("{}\n {}\n\n".format(new_population, pop_fitness))
        if (generation%10 == 0 and generation != 0):
            print("{} Generations Completed".format(generation))
            
        
    # Getting the best solution after iterating finishing all generations.
    #At first, the fitness is calculated for each solution in the final generation.
    fitness = ga.cal_pop_fitness(waveform, new_population)
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(fitness == np.max(fitness))[0]
    return new_population[best_match_idx, :], fitness[best_match_idx][0][0], best_outputs