#!/bin/bash
g++ -std=c++11 NeuralNetwork.cpp -c -o nn.o
g++ -std=c++11 single_class.cpp nn.o -o nn_single
g++ -std=c++11 multi_class.cpp nn.o -o nn_multi
