import csv
import matplotlib.pyplot as plt

x = []
y = []
# Read the data from csv
with open('part2.graph', 'rt') as csvFile:
    csvReader = csv.reader(csvFile)
    for row in csvReader:
        x.append(float(row[0]))
        y.append(float(row[1]))
plt.plot(x,y, color="b")
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()
