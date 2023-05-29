# MaCQFF

This repository contains the code for my semester project on "Approximation Techniques for Multi-Agent Coverage Control". The main practical contributions of this work are based on the following papers:

1. ["Near-Optimal Multi-Agent Learning for Safe Coverage Control"](https://proceedings.neurips.cc/paper_files/paper/2022/file/60dc26558762425a465cb0409fc3dc52-Paper-Conference.pdf) by Prajapat et al., 2022
2. ["Efficient High Dimensional Bayesian Optimization with Additivity and Quadrature Fourier Features"](https://proceedings.neurips.cc/paper_files/paper/2018/hash/4e5046fc8d6a97d18a5f54beaed54dea-Abstract.html) by Mutny and Krause, 2018
3. ["Lazier than Lazy Greedy"](https://las.inf.ethz.ch/files/mirzasoleiman15lazier.pdf) by Mirzasoleiman et al., 2015

The idea is to make the multi-agent learning algorithms from [1] more scalable using approximation methods for Gaussian processes (GP) and submodular function maximization. We chose the algorithms from [2] and [3], and call the resulting scheme "MaCQFF".

Since our implementation builds on the code by Prajapat et al., we refer to the [official repository](https://github.com/manish-pra/SafeMaC) for dependencies and running instructions. To use the quadrature Fourier features (QFF) approximation, set 'qff' to 'True' in the params file in main.py. To use the Stochastic-Greedy algorithm, set 'stochastic_greedy' to 'True'.
