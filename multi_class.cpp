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
    vector<int> sizes = vector<int>();
    sizes.push_back(21);
    sizes.push_back(20);
    sizes.push_back(3);
    
    ofstream graphFile;
    graphFile.open ("part1.graph");
    
    NeuralNetwork net = NeuralNetwork(sizes, 2000);
    TrainingOptions options = TrainingOptions();
    options.errorThresh = pow(0.000000001, 2);
    options.iterations = 1000;
    options.log = true;
    options.logger = &graphFile;
    options.logPeriod = 1;
    options.momentum = 0;
    options.learningRate = 0.5;
    vector<TrainingItem> items = vector<TrainingItem>();
    
    cout << "Reading Input File\n";

    bool input = true;
    vector<double> inputs, outputs;
    for (string line; getline(cin, line);) {
        if (line == "done") break;
        if (input)
            inputs = split(line, ',');
        else {
            outputs = split(line, ',');
            items.push_back(TrainingItem(inputs, outputs));
        }
        input = !input;
    }
    
    cout << "Done Reading\n";

    TrainingData data = TrainingData(items);
    net.train(data, options);
    
    int numCorrect = 0;
    int numSamples = 0;
    
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

