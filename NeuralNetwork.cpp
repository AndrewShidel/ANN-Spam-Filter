#include "NeuralNetwork.h"
#include <cmath>
#include <math.h>
#include <iostream>

int bijectiveMap(int a, int b) {
    return (a + b) * (a + b + 1) / 2 + a;
}
ull lpow(ull base, int exp);
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
    ull roundBase = lpow(10, this->precision);
	for (int layer = 1; layer <= this->outputLayer; layer++) {
		for (int node = 0; node < this->sizes[layer]; node++) {
			Vector& weights = this->weights[layer][node];

			double sum = this->biases[layer][node];
			for (int k=0; k< weights.size(); k++) {
                double weight = ((double)llround(weights[k] * roundBase)) / roundBase;
                if (layer == this->outputLayer && node == 0) {
                    std::cout.precision(17);
                    //std::cout << "w: " << weight << "\n";

                }
                sum += weight * input[k];
			}

            double outputValue = 1/(1+exp(-1*sum));
            this->outputs[index][layer][node] = outputValue;
            if (layer == this->outputLayer && node == 0) {
                //std::cout << "o: " << this->outputs[index][layer][node] << "\n";
            }
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
    //this->precision = 16;
	double error = 1.0;
	int i=0;
	for (; i < iterations && error > errorThresh; i++) {
		double sum = 0.0;
        for (int k=0; k<data.size(); ++k) {
            this->run(data.get(k).input, k);
            //std::cout << "a: " << data.get(k).output[0] << "\n";
            this->calculateDeltas(data.get(k).output, k);
            //double error = meanSquaredError(this->errors[this->outputLayer]);
            double error = absError(this->outputs[0][this->outputLayer], data.get(k).output);
            sum += error;
        }
        this->adjustWeights(learningRate, momentum, data.size());
        error = sum/data.size();
        /*if (error < 0.01 && this->precision > 1) {
            this->precision--;
        }*/

		if (logInfo && (i%logPeriod == 0) && i!=0) {
		    (*options.logger) << i << "," << error << "\n";
            //sleep(500000);
            //std::cout << "Iterations = " << i << ", Error = " << error <<  ", Precision = " << this->precision << "\n";
            std::cout << error << "\n";
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

double NeuralNetwork::absError(Vector v1, Vector v2) {
	double sum = 0.0;
	for (int i=0; i<v1.size(); i++) {
		//sum += pow(errors[i], 2);
        sum += std::abs(v1[i] - v2[i]);
	}
	return sum / v1.size();
}

double NeuralNetwork::meanSquaredError(Vector errors) {
	double sum = 0.0;
	for (int i=0; i<errors.size(); i++) {
		sum += pow(errors[i], 2);
        //sum += std::abs(errors[i]);
	}
	return sum / errors.size();
}

ull lpow(ull base, int exp) {
    ull result = 1ULL;
    while (exp) {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}
