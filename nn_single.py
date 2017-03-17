import sys
import csv
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import heapq
import signal
from sklearn import svm

numEntries = -1
numFeatures = -1

# features = [[x1, x2, ...], [y1, y2, ...], ...]
features = []
featureSums = []
featureMeans = []
featureStdDevs = []
classes = []

TP = 0
TN = 0
FP = 0
FN = 0

# Calculates and print precision, recall, f-measure, and accuracy
def print_stats():
    print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    precision = TP/float(TP+FP)
    recall = TP/float(TP+FN)
    f_measure = 2*precision*recall/float(precision+recall)
    accuracy = (TP+TN)/float(TP+TN+FP+FN)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f-measure: " + str(f_measure))
    print("accuracy: " + str(accuracy))

# Handles control-c
def signal_handler(signal, frame):
    print_stats()
    sys.exit(0)

def L1Dist(p1, p2):
    if len(p1) is not len(p2):
        raise ValueError('Points must have the same dimentionality')
    sum = 0
    for i in range(len(p1)):
        sum += abs(p1[i] - p2[i])
    return sum

# Read the data from csv
with open('spambase.data', 'rt') as csvFile:
    csvReader = csv.reader(csvFile)
    for row in csvReader:
        # Ignore the first labels row
        if numEntries is -1:
            numEntries = 0
            continue

        numEntries = numEntries+1
        if numFeatures is -1:
            numFeatures = len(row)-1
            features = [[] for i in range(numFeatures)]
            featureSums = [0]*numFeatures
            featureMeans = [0]*numFeatures
            featureStdDevs = [0]*numFeatures

        i = 0
        for item in row:
            item = float(item)
            if i is not len(row)-1:
                featureSums[i-1] += item
                features[i-1].append(item)
            else:
                classes.append(int(item))
            i = i+1
# Compute the mean and standard deviation
for i in range(numFeatures):
    featureMeans[i] = featureSums[i]/numEntries
    sum = 0.0
    for item in features[i]:
        sum += (item-featureMeans[i])**2
    average = sum/numEntries
    featureStdDevs[i] = math.sqrt(average)

# Standardize the data
featuresStd = copy.deepcopy(features)
for i in range(numFeatures):
    for j in range(numEntries):
        featuresStd[i][j] = (features[i][j]-featureMeans[i])/featureStdDevs[i]

numEntriesTrain = int(math.floor(numEntries * (2.0/3.0)))
numEntriesTest = int(math.ceil(numEntries * (1.0/3.0)))

featuresTest = [[] for i in features]
featuresStdTest = [[] for i in features]

random.seed(0)

# Randomize the data and select training and testing data
index_shuf = range(numEntries-1)
random.shuffle(index_shuf)
for j in range(len(featuresStd)):
    featuresStdTest[j] = [featuresStd[j][i] for i in index_shuf[numEntriesTrain:]]
    featuresTest[j] = [features[j][i] for i in index_shuf[numEntriesTrain:]]
    featuresStd[j] = [featuresStd[j][i] for i in index_shuf[:numEntriesTrain]]
    features[j] = [features[j][i] for i in index_shuf[:numEntriesTrain]]

# Randomize and seperate the training data
classesTest = [classes[i] for i in index_shuf[numEntriesTrain:]]
classes = [classes[i] for i in index_shuf[:numEntriesTrain]]

# Transpose the data

featuresStd = np.asarray(featuresStd).T.tolist()
features = np.asarray(features).T.tolist()
featuresStdTest = np.asarray(featuresStdTest).T.tolist()
featuresTest = np.asarray(featuresTest).T.tolist()
sizes = [len(featuresStd[0]), 20, 1]


for featureIdx in range(len(featuresStd)):
    print(','.join(str(x) for x in featuresStd[featureIdx]))
    print(classes[featureIdx])
print("done")

for featureIdx in range(len(featuresStdTest)):
    print(','.join(str(x) for x in featuresStdTest[featureIdx]))
    print(classesTest[featureIdx])
print("done")
