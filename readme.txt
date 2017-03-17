Author: Andrew Shidel


To run parts 1 and 2:
    ./run_single

To run part 3:
    ./run_multi

To generate the graphs:
    python part1_graph.py
    python part2_graph.py

Requirements:
    g++ must exist in your path and be C++ 11 compatable
    python2.7
    matplotlib
    numpy

About the project:
This project uses a hybrid approach of python and C++.
The bulk of the work is done in a C++ ANN library that I made a few years ago. The library has been heavly modified to only contain code relevent to this project. 
The core code can be found in NeuralNetwork.h and NeuralNetwork.cpp.
The files multi_class.cpp and single_class.cpp call the core code and both contain main functions. 
Data is piped into one of these two programs using the helper scripts nn_single.py and nn_multi.py. The files contain the code which reads the data files, standardizes, randomizes, and passes it to one of the two main c++ files.
The files part1.graph and part2.graph are generated from the c++ programs. They can then be displayed by running part1_graph.py and part2_graph.py.

