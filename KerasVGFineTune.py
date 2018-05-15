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

def save_model():
    currentTime = int(time.time())
    model_name = str(currentTime) + 'Mother_Model.h5'
    mother_model.save(model_name)

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

mother_model = load_model('1526388233Model.h5')

mother_model.summary()

conv_model = mother_model.get_layer('model_1')

print_layer_trainable()

input_shape = conv_model.layers[0].output_shape[1:3]

datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.1,
      height_shift_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=False,
      fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1./255)
batch_size = 180

train_dir = '../tiny-imagenet-200/train'
test_dir = '../tiny-imagenet-200/val'

#train_dir = './knifey-spoony/train'
#test_dir = './knifey-spoony/test'

generator_train = datagen_train.flow_from_directory(
    directory=train_dir,
    target_size=input_shape,
    batch_size=batch_size,
    shuffle=True,
    save_to_dir=None)

generator_train_histogram = datagen_test.flow_from_directory(
    directory=train_dir,
    target_size=input_shape,
    batch_size=batch_size,
    shuffle=False)

generator_test = datagen_test.flow_from_directory(
    directory=test_dir,
    target_size=input_shape,
    batch_size=batch_size,
    shuffle=False)

steps_test = generator_test.n / batch_size
steps_train_histogram = generator_train_histogram.n / batch_size


steps_per_epoch = 100000 / batch_size
steps_per_epoch = 10
epochs = 1

main_fine_history = mother_model.fit_generator(generator=generator_train,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

main_fine_result = mother_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(main_fine_result[1]))

# saves model
save_model()