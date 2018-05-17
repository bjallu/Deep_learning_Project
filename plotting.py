import numpy as np
import itertools
import random
from sklearn import *
import matplotlib.pyplot as plt

'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    axes = plt.gca()
    axes.set_xlim([0, 200])
    axes.set_ylim([0, 200])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


##### Confusion Matrix #####
classes = []
with open('classes.txt') as f:
    for line in f:
        classes.append(line)

confusionMatrix = np.load('ConfusionMatrix.npy')
np.fill_diagonal(confusionMatrix, 0)
plot_confusion_matrix(confusionMatrix, classes)


##### Count cluster sizes #####
clusters = np.load('labels_from_clustering.npy')
y = np.bincount(clusters)
ii = np.nonzero(y)[0]
print(np.vstack((ii,y[ii])).T)
'''

##### Plot training data #####
dataFile = open('Training_Data.txt', 'r')
lines = dataFile.readlines()

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



plt.clf()
plt.figure(1)
plt.plot(loss, label='Train - Loss')
plt.plot(test_loss, label='Train - Loss')
plt.legend()


plt.subplot(212)
plt.plot(acc, label='Train - Top 1 accuracy')
plt.plot(acc5, label='Train - Top 5 accuracy')
plt.plot(test_acc, label='Test - Top 1 accuracy')
plt.plot(test_acc5, label='Test - Top 5 accuracy')
plt.legend()
plt.show()