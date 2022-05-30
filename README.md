# RanBox
Source code of anomaly detection finder in the copula space

This repository contains source code for the programs RanBox and RanBoxIter. The algorithms are two versions of an anomaly finding routine which transforms multidimensional data into a copula space, and scans it to localize a small multi-dimensional interval which contains a local overdensity. 

The code is written in very basic c++, with root (root.cern.ch) libraries to produce histograms and scatterplots useful to test the algorithm performance and diagnostics. 

More information is available as comments in the source files, including
- information on how to compile the code
- information on parameters to run the code
- information on datasets available to test the code

Along with the RanBox.cpp and RanBoxIter.cpp sources, this directory provides copies of the data used to test it and report its performance
in the article "Anomaly detection in the copula space", https://arxiv.org/abs/2106.05747

The data are copied and minimally processed from the UCI repository (https://archive.ics.uci.edu/ml/index.php)
- the "HEPMASS" dataset with a signal at 1000 GeV
- a "MiniBooNE" dataset
- a dataset of credit card frauds
- a dataset of eeg scans

For more information please send inquiries to tommaso.dorigo@gmail.com
