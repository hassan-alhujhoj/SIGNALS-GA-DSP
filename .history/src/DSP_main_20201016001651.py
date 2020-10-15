"""
Genetic Algorithms for Digital Signal Processing
Created on Mon Oct 05 20:01:05 2020
Last Edited on  Mon Oct 12 2020 by Luke Trenberth
TODO tidy up this code and to finalise it. Add up the third FIR filter method in here too.
"""
from os import major
import numpy as np
import matplotlib
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import DSP_GA as ga

class DSP_Signal():
    def __init__(self, filename, fs=1024, N_Coeff=400):
        file = open(filename, "r")
        self.y_0 = []
        for line in file:
            words = line.split("  ")
            for word in words:
                if word != "":
                    self.y_0.append(float(word))
        self.fs = fs
        self.N = len(self.y_0)
        self.N_2 = int(self.N/2)
        self.t = [x/self.fs for x in list(range(0, self.N))]
        self.f = [x*self.fs/self.N for x in list(range(0, self.N_2))]
        self.P_0 = np.var(self.y_0)
        self.FFT_0 = fft(self.y_0)
        
        self.N_Coeff = N_Coeff  # Number of coefficients
        
    #Window Filtering method for the data class
    def WF(self, GA_data):
        #GA Data: [noise_f_1, noise_f_2, width]
        
        # Window Filtering
        self.width_WF = 8 # Width of stop band, Hz
        self.band_1 = [GA_data[0] -GA_data[2]/2, GA_data[0]+GA_data[2]/2] # Define band 1 bounds
        self.band_2 = [GA_data[1] -GA_data[2]/2, GA_data[1]+GA_data[2]/2] # Define band 2 bounds
        
        self.filter1_WF = signal.firwin(self.N_Coeff+1, self.band_1, window='hann', pass_zero='bandstop', fs=self.fs) # Filter for noise frequency 1
        self.filter2_WF = signal.firwin(self.N_Coeff+1, self.band_2, window='hann', pass_zero='bandstop', fs=self.fs) # Filter for noise frequency 2
        self.filter_WF = signal.convolve(self.filter1_WF, self.filter2_WF)  # Combined filter for noise frequencies
        self.y_WF = signal.lfilter(self.filter_WF, 1, self.y_0) # Apply noise filters to original data
        self.f_WF, self.h_WF = signal.freqz(self.filter_WF, 1, fs=self.fs) #
        self.FFT_WF = fft(self.y_WF)
        return self.SNR(self.y_WF)
        
    #Parks McLellan Filtering Method
    def PM(self, GA_data, TW =3, BW=5):
        # Filter Bands for filtering frequency 1 & 2
        f_1 = GA_data[0]
        f_2 = GA_data[1]
        if len(GA_data) > 2:
            TW = GA_data[2]
        if len(GA_data) > 3:
            BW = GA_data[3]
        
        band1_PM = [0, f_1 -BW/2-TW, f_1 -BW/2, f_1+BW/2, f_1+BW/2+TW, self.fs/2]
        band2_PM = [0, f_2 -BW/2-TW, f_2 -BW/2, f_2+BW/2, f_2+BW/2+TW, self.fs/2]
        gain_PM = [1, 0, 1]
        
        # Create filters for filtering frequency 1 & 2
        filter1_PM = signal.remez(self.N_Coeff+1, band1_PM, gain_PM, fs=self.fs) # Filter frequency 1
        filter2_PM = signal.remez(self.N_Coeff+1, band2_PM, gain_PM, fs=self.fs) # Filter frequency 2
        filter_PM = signal.convolve(filter1_PM, filter2_PM) # Combined Filter
        
        self.y_PM = signal.lfilter(filter_PM, 1, self.y_0) # Filter original data in time domain
        self.f_PM, self.h_PM = signal.freqz(filter_PM, 1, fs=self.fs) # Return filter frequency response
        self.FFT_PM = fft(self.y_PM) # Filtered data frequency domain response
        return self.SNR(self.y_PM)
        
    # TODO Frequency Sampling Filtering Method. THIS IS COPIED FROM ASSIGNMENT I.
    def FS(self, fs):
        trans_FS = 4    # Width of transition from pass band to stop band, Hz
        width_FS = 8    # Width of the stop band, Hz
        band1_FS = [0, noise_f[0] -width_FS/2-trans_FS, noise_f[0] -width_FS/2, noise_f[0]+width_FS/2, noise_f[0]+width_FS/2+trans_FS, fs/2]
        band2_FS = [0, noise_f[1] -width_FS/2-trans_FS, noise_f[1] -width_FS/2, noise_f[1]+width_FS/2, noise_f[1]+width_FS/2+trans_FS, fs/2]
        gain_FS = [1, 1, 0, 0, 1, 1] # Gain coefficients of bands

        filter1_FS = signal.firwin2(N_Coeff+1, band1_FS, gain_FS, fs=self.fs) # Filter for noise frequency 1
        filter2_FS = signal.firwin2(N_Coeff+1, band2_FS, gain_FS, fs=self.fs) # Filter for noise frequency 2
        filter_FS = signal.convolve(filter1_FS, filter2_FS) # Filter for both noise frequencies

        y_FS = signal.lfilter(filter_FS, 1, self.y_0) # Apply filter to time domain data
        f_FS, h_FS = signal.freqz(filter_FS, 1, fs=fs) # Filter Response
        FFT_FS = fft(y_FS) # Filtered Frequency Domain Response
        return 0
    
    # TODO maybe add IIR filtering method in here but that might be to much. Don't know tho.
    def IIR(self, freq1=31.456, freq2=74.36, BW=5):

        deg1 = 2 * np.pi * (freq1 / self.fs)
        deg2 = 2 * np.pi * (freq2 / self.fs)
        r = 1 - (BW / self.fs) * np.pi

        # Assign the coefficients for first and second filters
        a = 1 * 1
        b = (1 * -np.exp(-deg1 * 1j)) + (1 * -np.exp(deg1 * 1j))
        c = (1 * -np.exp(-deg1 * 1j)) * (1 * -np.exp(deg1 * 1j))

        d = 1 * 1 * 1j
        e = (-r * np.exp(-deg1 * 1j)) + (-r * np.exp(deg1 * 1j))
        f = (-r *  np.exp(-deg1 * 1j)) * (-r * np.exp(deg1 * 1j))

        g = 1 * 1
        h = (-1 * np.exp(-deg2 * 1j)) + (-1 * np.exp(deg2 * 1j))
        ii = (-1 * np.exp(-deg2 * 1j)) * (-1 * np.exp(deg2 * 1j))

        j = 1 * 1
        k = (-r * np.exp(-deg2 * 1j)) + (-r * np.exp(deg2 * 1j))
        l = (-r * np.exp(-deg2 * 1j)) * (-r * np.exp(deg2 * 1j))

        # Calculte the gain of the overall transfer function
        Wf = 2 * np.pi * 10
        ND_array = [np.exp(0), np.exp(np.i * Wf), np.exp(-2 * Wf)]
        H_Z1_dot = np.dot(ND_array,[a, b, c])
        H_Z2_dot = np.dot(ND_array, [d, e, f])
        Gain = abs(H_Z2_dot / H_Z1_dot)

        # convlute the the de/numerator of the first transfer function with de/numerator of the second funcion
        NUM_Z = np.array( np.convolve( [a, b, c], [g, h, ii] ) )
        DEN_Z = np.array( np.convolve( [d, e, f], [j, k, l] ) )

        self.IIR_w, self.IRR_H = signal.freqz(Gain * NUM_Z, DEN_Z, self.N)
        self.IIR_f = self.fs * self.IIR_w / (2 * np.pi)

    #Returns a Signal to Noise Ratio for a given input Power
    def SNR (self, y):
        return self.P_0  - np.var(y)
    
    # Plots a Fast Fourier Transform for simple graphing
    def FFTplot(self, f, FFT, title="ECG Signal Frequency Spectrum"):
        plt.figure()
        plt.plot(f, abs(FFT)[:self.N_2])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Voltage (uV)")
        plt.title(title)
        plt.show()
        
    def IIRplot(self, outputs, dpi=200):
        plt.figure(2, figsize=(5, 10), dpi=dpi)
        plt.plot(self.IIR_f, abs(self.IIR_H), "-g", label="IIR Filter")
        plt.title("R Filter")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (uV)")
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig('wiki/IIR_magnitude.png', dpi=dpi)
        plt.show()
        

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
    

# Implementation of a Parks-McLellan Filter using Genetic Algorithms
def main():
    waveform = DSP_Signal("Signal_files/ECG15.txt")
    
    # Fixed Parameters, found by trial and error s
    f_count = 2
    mating_parent_number = 3
    pop_size = 20
    num_generations = 10
    my_dpi = 200 #dots per inch (resolution of an image)
    
    # Conduct a Genetic Algorithm approximation
    best_soln, best_soln_fitness, best_outputs = GA_filter(waveform, 
                                                           f_count, pop_size, 
                                                           mating_parent_number, num_generations)    
    print("Best solution : \n", best_soln)
    print("Best solution fitness : \n", best_soln_fitness)
    plt.figure(1)
    plt.plot(best_outputs, "-k", label="Fittest Output")
    plt.title("Fitness of ECG Signal using GA Algorithm")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness (Signal to Noise Ratio)")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('wiki/{}Gen{}Pop.png'.format(num_generations, pop_size))
    plt.show()

    
   
    waveform.FFTplot(waveform.f, waveform.FFT_0, title="Before filtering")
    waveform.PM(best_soln[0])
    waveform.FFTplot(waveform.f, waveform.FFT_PM, title="After Filtering")
    
main()

