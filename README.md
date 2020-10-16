# ENEL420 - Genetic Algorithm for Digital Signal Processing
<img src="doc/wiki/bf_filtering_10kGen.png" alt="fig2" width="1200"/>
<img src="doc/wiki/10kGen.png" alt="fig1" width="1200"/>
<img src="doc/wiki/af_filtering_100Gen.png" alt="fig3" width="1200"/>

## Description
This is a code that uses the concept of Darwin's theory of natural selection. The code uses Genetic Algorithms (GA) 
to filter noisy ECG singals that has two fundamental interference frequencies using the fitness of the generated frequency
population. The ECG is filtered by either an FIR or IIR filter. A Signal to Noise Ratio (SNR) is then obtained to determine the fitness of the population.

### GA Cycle
<img src="doc/wiki/Flowchart.png" alt="fig4" width="300" align="center"/>

### Rejection Frequencies
- `30Hz <= f <= 100Hz`

### GA Operators
1. Crossover
2. Mutation

### FIR Filters
1. Window Filter  
2.  Parks-McClellan Filter  
3. Frequency Selection Filter  

### IIR Filter
* Bi-quad 2-Pole notch filter  

## Contributors
* [Hassan Alhujhoj](https://github.com/hassan-alhujhoj)
* [Luke Trenberth](https://github.com/ltr28)