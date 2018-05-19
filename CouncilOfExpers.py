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
number_of_experts = 36
labels = np.load('labels_from_clustering_third_generation.npy')

'''
expert_model_list = []

for i in range(number_of_experts):
    expertName = str(i) + "MichaelPhelps.h5"
    model = load_model(expertName)
    expert_model_list.append(model)
'''

base_model = load_model('1526388233Model.h5')
# base_model_final = load_model('1526443982Mother_Model.h5')
conv_model = base_model.get_layer('model_1')
input_shape = conv_model.layers[0].output_shape[1:3]

datagen_test = ImageDataGenerator(rescale=1./255)
test_dir = '../tiny-imagenet-200/val'
batch_size = 1

generator_test = datagen_test.flow_from_directory(
        directory=test_dir,
        target_size=input_shape,
        batch_size=batch_size,
        shuffle=False)

steps_test = generator_test.n / batch_size

base_result = base_model.predict_generator(generator_test, steps=steps_test, verbose=1)
generator_test.reset()
# base_final_result = base_model_final.predict_generator(generator_test, steps=steps_test, verbose=1)
# generator_test.reset()

council_resuts = np.zeros_like(base_result)
council_resuts_squared = np.zeros_like(base_result)

for i in range(number_of_experts):
    expertName = str(i) + "MichaelPhelps.h5"
    model = load_model(expertName)
    expert_result = model.predict_generator(generator_test, steps=steps_test, verbose=1)
    generator_test.reset()
    expert_result_squared = np.power(expert_result, 2)
    council_resuts += expert_result
    council_resuts_squared += expert_result_squared
    del model
    del expert_result
    del expert_result_squared
'''
for e in expert_model_list:
    expert_result = e.predict_generator(generator_test, steps=steps_test, verbose=1)
    generator_test.reset()
    expert_result = np.power(expert_result, 2)
    council_resuts += expert_result
'''

base_predictions = np.argmax(base_result, axis=1)
# base_final_predictions = np.argmax(base_final_result, axis=1)
expert_predictions = np.argmax(council_resuts, axis=1)
expert_predictions_squared = np.argmax(council_resuts_squared, axis=1)

total_results = []

base_correct = 0
expert_correct = 0
expert_correct_squared = 0
base_final_correct = 0
total = 0

for i, initial in enumerate(base_predictions):

    base_p = base_predictions[i]
    # base_final_p = base_final_predictions[i]
    expert_p = expert_predictions[i]
    expert_p_squared = expert_predictions_squared[i]
    true_p = generator_test.classes[i]

    if(true_p == base_p):
        base_correct += 1

    '''
    if(true_p == base_final_p):
        base_final_correct += 1
    '''
    if (true_p == expert_p):
        expert_correct += 1

    if (true_p == expert_p_squared):
        expert_correct_squared += 1

    total += 1

    results = [base_correct / total, expert_correct / total, expert_correct_squared/total]
    # results = [base_correct / total, base_final_correct / total, expert_correct / total, expert_correct_squared / total]
    print(results)
    total_results.append(results)

file = open('MichaelPhelps36.txt', 'w+')

for line in total_results:
    file.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')

file.close()