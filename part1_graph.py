import csv
import matplotlib.pyplot as plt

x = []
y = []
# Read the data from csv
with open('part1.graph', 'rt') as csvFile:
    csvReader = csv.reader(csvFile)
    for row in csvReader:
        x.append(int(row[0]))
        y.append(float(row[1]))
plt.plot(y, color="b", label="RMSE of Test Data")
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
