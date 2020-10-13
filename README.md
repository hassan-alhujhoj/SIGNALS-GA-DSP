# ENEL420 - Genetic Algorithm for Digital Signal Processing
<img src="wiki/10kGen.PNG" alt="Main" width="600"/>
<img src="wiki/bf_filtering_10kGen.PNG" alt="Main" width="600"/>
<img src="wiki/bf_filtering_100Gen50Pop.PNG" alt="Main" width="600"/>

## Description
This is an code that uses the concept of Darwin's throery of natural selection. The code uses genetic algorithm 
to filter an noisy ECG singal that has two fundamental interfernce frequencies using the fitness of the generated frequency
population. The ECG is filtered by either an FIR or IIR filter. A SNR is then optained to determine the fitness of the population.

### Rejection Frequencies
- `f(1) = 31.456`
- `f(2) = 74.36 Hz`

### GA Operators
1. Crossover
2. Mutation

### FIR Filters
I. Window Function  
II. Parks-McClellan Filter  

## Contributors
* [Hassan Alhujhoj](https://github.com/hassan-alhujhoj)
* [Luke Trenberth](https://eng-git.canterbury.ac.nz/ltr28)