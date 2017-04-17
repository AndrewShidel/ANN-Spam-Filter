#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <iterator>
#include <sstream>
#include <fstream>
using namespace std;

vector<double> split(const string &s, char delim);
int round(vector<double> in);

int main(int argc, char** argv) {
    ofstream graphFile;
    graphFile.open ("part1.graph");
    for (int prec=1; prec<=16; prec++) {
        vector<int> sizes = vector<int>();
        sizes.push_back(1);
        sizes.push_back(2);
        sizes.push_back(400);

        NeuralNetwork net = NeuralNetwork(sizes, 1);
        net.precision = prec;
        TrainingOptions options = TrainingOptions();
        options.errorThresh = pow(0.000000001, 2);
        options.iterations = 10000;
        options.log = true;
        options.logger = &graphFile;
        options.logPeriod = 9999;
        options.momentum = 0;
        options.learningRate = 0.5;
        vector<TrainingItem> items = vector<TrainingItem>();

        vector<double> inputs, outputs;
        inputs.push_back(1);
        for (int i=0; i<sizes[sizes.size()-1]; ++i) {
            outputs.push_back(((double)rand())/RAND_MAX);
        }
        items.push_back(TrainingItem(inputs, outputs));
        TrainingData data = TrainingData(items);
        net.train(data, options);
    }
    int numCorrect = 0;
    int numSamples = 0;
/*
    for (string line; getline(cin, line);) {
        if (line == "done") break;
        inputs = split(line, ',');
        vector<double> resultV = net.run(inputs, 0);
        getline(cin, line);
        vector<double> realResults = split(line, ',');
        for (int i=0; i<realResults.size(); ++i) {
            if (realResults[i] == 1.0) {
                if (round(resultV) == i) {
                    numCorrect++;
                }
            }
        }
        numSamples++;
    }
    cout << "\nTesting Error: " << 1 - ((double)numCorrect)/numSamples << "\n";
*/
    graphFile.close();
    return 0;
}

int round(vector<double> in) {
    int maxIndex = 0;
    for (int i=0; i<in.size(); ++i) {
        if (in[i] >= in[maxIndex])
            maxIndex = i;
    }
    return maxIndex;
}

// From http://stackoverflow.com/questions/236129/split-a-string-in-c
template<typename Out>
void split(const string &s, char delim, Out result) {
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        *(result++) = item;
    }
}
vector<double> split(const string &s, char delim) {
    vector<string> elems;
    vector<double> result;
    split(s, delim, back_inserter(elems));
    for (int i=0; i<elems.size(); ++i) {
        result.push_back(stod(elems[i]));
    }
    return result;
}
