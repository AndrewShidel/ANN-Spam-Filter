#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>

int bijectiveMap(int a, int b) {
    return (a + b) * (a + b + 1) / 2 + a;
}

NeuralNetwork::NeuralNetwork(){}

NeuralNetwork::NeuralNetwork(std::vector<int> sizes, int trainingSize) {
	this->sizes = sizes;
	this->outputLayer = (int)sizes.size()-1;

	this->biases = Vector2D();
	this->weights = Vector3D();
	this->outputs = Vector3D();

	this->deltas = Vector3D();
	this->changes = Vector3D();
	this->errors = Vector2D();
    
    this->biases.push_back(randos(this->sizes[0]));
    this->weights.push_back(Vector2D());
    this->changes.push_back(Vector2D());

    for (int i=0; i<trainingSize; ++i) {
        this->deltas.push_back(Vector2D());
        this->outputs.push_back(Vector2D());
	    for (int layer = 0; layer <= this->outputLayer; layer++) {
		    int size = this->sizes[layer];
		    this->deltas[i].push_back(zeros(size));
		    this->outputs[i].push_back(zeros(size));
        }
    }

	for (int layer = 0; layer <= this->outputLayer; layer++) {
		int size = this->sizes[layer];
		this->errors.push_back(zeros(size));

		if (layer > 0) {
			this->biases.push_back(randos(size));
			this->weights.push_back(Vector2D());
			this->changes.push_back(Vector2D());

			for (int node = 0; node < size; node++) {
				int prevSize = this->sizes[layer-1];
				this->weights[layer].push_back(randos(prevSize));
                //this->weights[layer].push_back(zeros(prevSize));
				this->changes[layer].push_back(zeros(prevSize));
			}
		}
	}
}

Vector NeuralNetwork::run(std::vector<double> input, int index) {
	this->outputs[index][0] = input;
	for (int layer = 1; layer <= this->outputLayer; layer++) {
		for (int node = 0; node < this->sizes[layer]; node++) {
			Vector& weights = this->weights[layer][node];

			double sum = this->biases[layer][node];
			for (int k=0; k< weights.size(); k++) {
                sum += weights[k] * input[k];
			}
            
            double outputValue = 1/(1+exp(-1*sum)); 
            this->outputs[index][layer][node] = outputValue;
		}
		input = this->outputs[index][layer];
	}
	return input;
}

TrainingResults NeuralNetwork::train(TrainingData data, TrainingOptions options) {
    this->trainCount++;
	int iterations = options.iterations;
	double errorThresh = options.errorThresh;
	double learningRate = options.learningRate;
	double momentum = options.momentum;
	bool logInfo = options.log;
	int logPeriod = options.logPeriod;

	double error = 1.0;
	int i=0;
	for (; i < iterations && error > errorThresh; i++) {
		double sum = 0.0;
        for (int k=0; k<data.size(); ++k) {
            this->run(data.get(k).input, k);
            this->calculateDeltas(data.get(k).output, k);
            double error = meanSquaredError(this->errors[this->outputLayer]);
            sum += error;
        }
        this->adjustWeights(learningRate, momentum, data.size());

        error = sum/data.size();
		if (logInfo && (i%logPeriod == 0)) {
		    (*options.logger) << i << "," << error << "\n";
            std::cout << "Iterations = " << i << ", Error = " << error << "\r";
            std::cout.flush();
		}
        
	}
    this->error = error;
	return TrainingResults(error, i);
}

double NeuralNetwork::trainPattern(const Vector& input, const Vector& target, double learningRate, double momentum) {
    int index = 0;
	this->run(input, index);
    int trainingSize = 1;
	this->calculateDeltas(target, index);
	this->adjustWeights(learningRate, momentum, trainingSize);

	return meanSquaredError(this->errors[this->outputLayer]);
}

void NeuralNetwork::calculateDeltas(const Vector& target, int index) {
	for (int layer = this->outputLayer; layer>=0; layer--) {
		for (int node = 0; node < this->sizes[layer]; node++) {
			double output = this->outputs[index][layer][node];

			double error = 0.0;
			if (layer == this->outputLayer) {
				error = target[node] - output;
			} else {
				Vector& deltas = this->deltas[index][layer+1];
				for (int k=0; k < deltas.size(); k++) {
                    double weight = this->weights[layer + 1][k][node];
                    error += deltas[k] * weight;
				}
			}
			this->errors[layer][node] = error;
			this->deltas[index][layer][node] = error*output*(1-output);
		}
	}
}

void NeuralNetwork::adjustWeights(double learningRate, double momentum, int trainingSize) {
    for (int index=0; index<trainingSize; ++index) {
        for (int layer = 1; layer <= this->outputLayer; layer++) {
            const Vector& incoming = this->outputs[index][layer - 1];

            for (int node = 0; node < this->sizes[layer]; node++) {
                double delta = this->deltas[index][layer][node];
                for (int k=0; k<incoming.size(); k++) {
                    double change = this->changes[layer][node][k];

                    change = (learningRate * delta * incoming[k])
                                + (momentum * change);

                    this->changes[layer][node][k] += change;
                }
                this->biases[layer][node] = 0; //+= learningRate * delta;
            }
        }
    }
    for (int layer = 1; layer <= this->outputLayer; layer++) {
        for (int node = 0; node < this->sizes[layer]; node++) {
            const Vector& incoming = this->outputs[0][layer - 1];
            for (int k=0; k<incoming.size(); k++) {
                double change = this->changes[layer][node][k];
                change = change/trainingSize;
                this->weights[layer][node][k] += change;
                this->changes[layer][node][k] = 0;
            }
        }
    }
}

double NeuralNetwork::randomWeight() {
	return ((double)rand() / RAND_MAX)*0.4 - 0.2;
}

Vector NeuralNetwork::zeros(int size) {
	return Vector(size, 0.0);
}

Vector NeuralNetwork::randos(int size) {
	Vector vect = Vector();
	for (int i=0; i<size; i++) {
		vect.push_back(randomWeight());
	}
	return vect;
}

double NeuralNetwork::meanSquaredError(Vector errors) {
	double sum = 0.0;
	for (int i=0; i<errors.size(); i++) {
		sum += pow(errors[i], 2);
	}
	return sum / errors.size();
}

