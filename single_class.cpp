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
void precVsRecal(vector<double> testIns, vector<double> testOuts); 

int main(int argc, char** argv) {
    vector<int> sizes = vector<int>();
    sizes.push_back(57);
    sizes.push_back(20);
    sizes.push_back(1);
    
    ofstream graphFile;
    graphFile.open ("part1.graph");
    
    NeuralNetwork net = NeuralNetwork(sizes, 4000);
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
    
    vector<double> testIns = Vector();
    vector<double> testOuts = Vector();

    for (string line; getline(cin, line);) {
        if (line == "done") break;
        inputs = split(line, ',');
        double result = net.run(inputs, 0)[0];
        testIns.push_back(result);
        if (result >= 0.5)
            result = 1.0;
        else
            result = 0.0;
        
        getline(cin, line);
        cout << "Est: " << result << ", Real: " << line << "\n";
        double realResult = stod(line);
        testOuts.push_back(realResult);
        if (realResult == result)
            numCorrect++;
        numSamples++;
    }
    cout << "Testing Error: " << 1 - ((double)numCorrect)/numSamples << "\n";
    precVsRecal(testIns, testOuts);
    graphFile.close();
    return 0;
}

void precVsRecal(vector<double> testIns, vector<double> testOuts) { 
    ofstream graphFile;
    graphFile.open ("part2.graph");
    double t = 0.0;
    while(t <= 1.0) {
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;
        for (int i=0; i<testIns.size(); ++i) {
            double result = testIns[i];
            if (result >= t)
                result = 1.0;
            else
                result = 0.0;
            double realVal = testOuts[i];
            if (result == realVal) {
                if (result == 1)
                    TP++;
                else
                    TN++;
            }else{
                if (result == 1)
                    FP++;
                else
                    FN++;
            }
        }
        if (TP+FN != 0.0 && FP+FP != 0.0)
            graphFile << ((double)TP)/(TP+FN) << "," << ((double)TP)/(TP+FP) << "\n";
        t += 0.1;
    }
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

