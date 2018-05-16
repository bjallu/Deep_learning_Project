import matplotlib.pyplot as plt
import numpy as np

dataFile = open('data.txt', 'r')
lines = dataFile.readlines()


clusters = np.load('labels_from_clustering.npy')
y = np.bincount(clusters)
ii = np.nonzero(y)[0]
print(np.vstack((ii,y[ii])).T)


loss = []
acc = []
acc5 = []
test_loss = []
test_acc = []
test_acc5 = []

for i, line in enumerate(lines):
    line = line.replace("\t", ' ')
    line = line.replace("\n", '')
    line = line.split(' ')
    loss.append(line[0])
    acc.append(line[1])
    acc5.append(line[2])
    test_loss.append(line[3])
    test_acc.append(line[4])
    test_acc5.append(line[5])


plt.plot(loss)
plt.plot(acc)
plt.plot(acc5)
'''
plt.plot(test_loss)
plt.plot(test_acc)
plt.plot(test_acc5)
'''
plt.show()