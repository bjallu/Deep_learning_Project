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

for i in range(number_of_experts):
    expertName = str(i) + "Expert.h5"
    model = load_model(expertName)
    expert_model_list.append(model)

base_model = load_model('1526388233Model.h5')
base_model_final = load_model('1526443982Mother_Model.h5')
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

base_result = base_model.predict_generator(generator_test, steps=steps_test)
generator_test.reset()
base_final_result = base_model_final.predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_1_result = expert_model_list[0].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_2_result = expert_model_list[1].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_3_result = expert_model_list[2].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_4_result = expert_model_list[3].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_5_result = expert_model_list[4].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_6_result = expert_model_list[5].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_7_result = expert_model_list[6].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_8_result = expert_model_list[7].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_9_result = expert_model_list[8].predict_generator(generator_test, steps=steps_test)
generator_test.reset()
expert_10_result = expert_model_list[9].predict_generator(generator_test, steps=steps_test)

council_resuts = expert_1_result + expert_2_result + expert_3_result + expert_4_result + expert_5_result + expert_7_result + expert_8_result + expert_9_result + expert_10_result

base_predictions = np.argmax(base_result, axis=1)
base_final_predictions = np.argmax(base_final_result, axis=1)
expert_predictions = np.argmax(council_resuts, axis=1)

true_predictions = []

base_correct = 0
expert_correct = 0
base_final_correct = 0
total = 0
for i, initial in enumerate(base_predictions):

    base_p = base_predictions[i]
    base_final_p = base_final_predictions[i]
    expert_p = expert_predictions[i]
    true_p = generator_test.classes[i]

    if(true_p == base_p):
        base_correct += 1

    if(true_p == base_final_p):
        base_final_correct += 1

    if (true_p == expert_p):
        expert_correct += 1
    total += 1
    results = 
    print("Base: " + str(base_correct / total) + " Base final:" + str(base_final_correct/total) +  " Experts: " + str(expert_correct / total))


print("Base acc")
print(base_correct / total)

print("Base final acc")
print(base_final_correct / total)

print("Expert acc")
print(expert_correct / total)