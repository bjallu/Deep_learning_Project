import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from input_pipe import *
from keras.models import load_model
import pickle
from sklearn.cluster import SpectralClustering
import shutil
from distutils.dir_util import copy_tree


def save_model(i):
    model_name = str(i) + 'Expert.h5'
    expert.save(model_name)

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def create_sub_directories(clusters):

    # if cluster number does not exists
    for c in clusters:
        tempPathTrain = '../tiny-imagenet-200/val/'
        tempPathVal = '../tiny-imagenet-200/val/'

############ Create and train experts ############
number_of_experts = 10
confusionMatrix = np.load('ConfusionMatrix.npy')

clustering = SpectralClustering(n_clusters = number_of_experts, affinity='precomputed')
d = clustering.fit(confusionMatrix)

path = '../tiny-imagenet-200'
classFile = open('classes.txt', 'r')
lines = classFile.readlines()

# Create number_of_experts of directories
for i in range(number_of_experts):

    tempPathCluster = path + str(i)
    # check if cluster main directory exists, if not create and create subdirectories

    if not os.path.exists(tempPathCluster):
        os.makedirs(tempPathCluster)
        tempPathTrain = tempPathCluster + "/train"
        tempPathVal = tempPathCluster + "/val"
        os.makedirs(tempPathTrain)
        os.makedirs(tempPathVal)

# copy the images into the correct directory
for i, line in enumerate(lines):
    categoryCluster = d.labels_[i]
    categoryName = line.strip('\n')

    # create a image directory in all the cluster training sets but copy only images if cluster training set
    # is equal to the category's assigned cluster
    for cluster in range(0, number_of_experts):

        tempTrain = path + str(cluster) + "/train/" + str(categoryName)
        tempVal = path + str(cluster) + "/val/" + str(categoryName)

        if(cluster == categoryCluster):
            # Copy over training images
            srcTrain = path + "/train/" + str(categoryName)
            copy_tree(srcTrain, tempTrain)

            # Copy over val images
            srcTrain = path + "/val/" + str(categoryName)
            copy_tree(srcTrain, tempVal)

        else:
            # If the images does not belong to the cluster, just add an empty folder
            os.makedirs(tempTrain)
            os.makedirs(tempVal)

datagen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=[0.9, 1.5],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1./255)
epochs = 1

# trains the experts one by one
for i in range(number_of_experts):

    expert = load_model('1526388233Model.h5')
    conv_model = expert.get_layer('model_1')
    input_shape = conv_model.layers[0].output_shape[1:3]

    batch_size = 90

    train_dir = '../tiny-imagenet-200' + str(i) + '/train'
    test_dir = '../tiny-imagenet-200' + str(i) + '/val'

    generator_train = datagen_train.flow_from_directory(
        directory=train_dir,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=True,
        save_to_dir=None)

    generator_test = datagen_test.flow_from_directory(
        directory=test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=False)

    steps_per_epoch = generator_test.n / batch_size
    steps_test = generator_test.n / batch_size

    expert_history = expert.fit_generator(generator=generator_train,
                                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                                      validation_data=generator_test,
                                      validation_steps=steps_test)

    expert_result = expert.evaluate_generator(generator_test, steps=steps_test)
    print("Test-set classification accuracy: {0:.2%}".format(expert_result[1]))

    # saves model
    save_model(i)