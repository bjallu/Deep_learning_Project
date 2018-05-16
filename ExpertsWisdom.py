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

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

def predict(image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    return base_model.predict(img_array)


############ Create and train experts ############
number_of_experts = 10
labels = np.load('labels_from_clustering.npy')

expert_model_list = []

for i in range(4):
    expertName = str(i) + "Expert.h5"
    model = load_model(expertName)
    expert_model_list.append(model)

base_model = load_model('1526388233Model.h5')
conv_model = base_model.get_layer('model_1')
input_shape = conv_model.layers[0].output_shape[1:3]

datagen_test = ImageDataGenerator(rescale=1./255)
test_dir = '../tiny-imagenet-2000/val'
batch_size = 1

generator_test = datagen_test.flow_from_directory(
        directory=test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=False)

steps_test = generator_test.n / batch_size

base_result = base_model.predict_generator(generator_test, steps=steps_test, verbose=1)

intitial_predictions = np.argmax(base_result, axis=1)
final_predictions = []
true_predictions = []

intitial_correct = 0
correct = 0
total = 0
for i, initial in enumerate(intitial_predictions):
    expert = labels[initial]
    expert_to_consult = expert_model_list[0] # !!!!!!!!
    image_path = test_dir + '/' + generator_test.filenames[i]

    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)
    img_array = np.expand_dims(np.array(img_resized), axis=0)/255

    if(len(img_array.shape) < 4):
        rgbimg = PIL.Image.new("RGB", img.size)
        rgbimg.paste(img)
        img_resized = rgbimg.resize(input_shape, PIL.Image.LANCZOS)
        img_array = np.expand_dims(np.array(img_resized), axis=0)/255

    final = expert_to_consult.predict(img_array)
    # final = expert_to_consult.predict_generator(generator_test[i], steps=steps_test, verbose=1)

    intial_p = intitial_predictions[i]
    final_p = np.argmax(final, axis=1)
    true_p = generator_test.classes[i]

    if(true_p == intial_p):
        intitial_correct += 1


    if (true_p == final_p[0]):
        correct += 1

    final_predictions.append(final_p)
    true_predictions.append(true_p)

    total += 1


print("Inital acc")
print(intitial_correct / total)

print("Final acc")
print(correct / total)



print("Hello")