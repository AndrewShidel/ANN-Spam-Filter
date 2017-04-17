#ifndef __X_H_INCLUDED__   // if x.h hasn't been included yet...
#define __X_H_INCLUDED__   //   #define this so the compiler knows it has been included



#include <vector>
#include <iostream>
#include <map>
#include <float.h>

#define NULL_CONNECTION -DBL_MAX
#define MEM_NODE -DBL_MAX+1

class TrainingItem;
class TrainingData;
class TrainingOptions;
class TrainingResults;

typedef std::vector<double> Vector;
typedef std::vector< Vector > Vector2D;
typedef std::vector< std::vector< std::vector<double> > > Vector3D;

typedef unsigned long long int ull;

int run_square_array(void);

class NeuralNetwork {
public:
    Vector3D weights;
    Vector3D initialWeights;
    Vector3D deltas;
    Vector2D errors;
    Vector2D biases;

    Vector3D outputs;
    Vector3D changes;

    int trainCount = 0;
    double error = 1.0;
    int generation = 0;
    int precision = 16;

    std::vector<int> sizes; //Size of each layer

    NeuralNetwork();
	NeuralNetwork(std::vector<int> sizes, int trainingSize);
	Vector run(Vector input, int index);
    Vector run_gpu(std::vector<double> input);
	TrainingResults train(TrainingData data, TrainingOptions options);
	double trainPattern(const Vector& input, const Vector& target, double learningRate, double momentum);
	void calculateDeltas(const Vector& target, int index);
	void adjustWeights(double learningRate, double momentum, int trainingSize);
    NeuralNetwork clone();

    static double randomWeight();
    static Vector zeros(int size);
    static Vector randos(int size);
    static double meanSquaredError(Vector errors);
    static double absError(Vector v1, Vector v2);

    static Vector makeInputVector(int length, double args[]) {
        return Vector(args, args+length);
    }


private:
	int outputLayer; // Index of the output layer
};

class TrainingItem {
public:
	Vector input;
	Vector output;

	TrainingItem(Vector input, Vector output) {
		this->input = input;
		this->output = output;
	}
};

class TrainingData {
public:
	std::vector<TrainingItem> items;
    TrainingData(){}
	TrainingData(std::vector<TrainingItem> _items) {
		items = _items;
	}
	void addItem(TrainingItem item) {
		items.push_back(item);
	}
	TrainingItem get(int item) {
		return items[item];
	}
	int size() {
		return items.size();
	}
};

class TrainingOptions {
public:
	int iterations = 20000;
    double errorThresh = 0.005;
    std::ostream *logger = &std::cout;
    bool log = false;
    int logPeriod = 10;
    double learningRate = 0.3;
    double momentum = 0.1;
    void* callback = NULL;
    int callbackPeriod = 10;
};

class TrainingResults {
public:
	double error = 0;
	int iterations = 0;
	TrainingResults(double error, int iterations) {
		this->error = error;
		this->iterations = iterations;
	}
};

#endif
