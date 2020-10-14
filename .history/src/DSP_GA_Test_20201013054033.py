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
            filename = "Signal_Files/enel420_grp_{}.txt".format(i)
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
            filename = "Signal_Files/enel420_grp_15.txt"
            waveform = DSP_Signal(filename)
            best_soln, best_soln_fitness, best_output = GA_filter(waveform, 
                                                           input_num, solutions_per_population, 
                                                           mating_parent_number, num_generations)
            SNR_writer.writerow([num_generations, best_soln_fitness])
            print("{} finished".format(num_generations))

#plot_best_outputs()
plot_num_generations()        
            