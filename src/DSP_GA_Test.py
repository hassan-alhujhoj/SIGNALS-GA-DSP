"""
Genetic Algorithms for Digital Signal Processing Test Case Generator
Created on Mon Oct 05 20:01:05 2020
Last Edited on  Mon Oct 12 2020 by Luke Trenberth
"""
import numpy as np
import matplotlib.pyplot as plt
import csv, ast
from DSP_main import GA_filter, DSP_Signal
import DSP_GA as ga
import time

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
def ConvergencePlot(waveform, input_num, mating_parent_number, max_gen=20, convergence_point = 100000):
    GensPerPopSize = []
    PopSize = []
    for population_size in range(5, 50):
        generation = 1
        # Defining the population size.
        pop_size = (population_size,input_num) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
        #Creating the initial population.
        new_population = ga.create_population(pop_size)
        fitness = ga.cal_pop_fitness(waveform, new_population)
        while (np.max(fitness) < convergence_point):
            # Measuring the fitness of each chromosome in the population.
            fitness = ga.cal_pop_fitness(waveform, new_population)
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
            
            generation += 1
            if (generation > max_gen):
                print("{} Did Not Converge".format(population_size))
                generation = max_gen
                break
        GensPerPopSize.append(generation)
        PopSize.append(population_size)
        print("{} Population Size Complete".format(population_size))
        
    return GensPerPopSize, PopSize

#The GA_filter function filters an input waveform 
def ExecutionTime(waveform, input_num, mating_parent_number, num_generations):
    GensPerPopSize = []
    PopSize = []
    for population_size in range(5, 50):
        time = time.time()
        # Defining the population size.
        pop_size = (population_size,input_num) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
        #Creating the initial population.
        new_population = ga.create_population(pop_size)
        fitness = ga.cal_pop_fitness(waveform, new_population)
        while (np.max(fitness) < convergence_point):
            # Measuring the fitness of each chromosome in the population.
            fitness = ga.cal_pop_fitness(waveform, new_population)
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
            
            generation += 1
            if (generation > max_gen):
                print("{} Did Not Converge".format(population_size))
                generation = max_gen
                break
        GensPerPopSize.append(generation)
        PopSize.append(population_size)
        print("{} Population Size Complete".format(population_size))
        
    return GensPerPopSize, PopSize


# Implementation of a Parks-McLellan Filter using Genetic Algorithms
def main():
    waveform = DSP_Signal("src/Signal_files/ECG15.txt")
    
    # Fixed Parameters, found by trial and error s
    f_count = 2
    mating_parent_number = 3
    
    # Conduct a Genetic Algorithm approximation
    GensPerPopSize, PopSize = ConvergencePlot(waveform, f_count, mating_parent_number)
    
    plt.figure(1)
    plt.plot(PopSize, GensPerPopSize, "-k", label="Fittest Output")
    plt.title("Generations to Convergence")
    plt.xlabel("Population Size")
    plt.ylabel("Number of Generations")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

main()