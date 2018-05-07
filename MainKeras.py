from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
from keras import utils
from keras import optimizers
import glob
from collections import deque
import numpy as np
from input_pipe import *
import re
from input_pipe import *
from keras.layers import Input


def print_Categories():
    # label_dict (folder_name, number)
    # class_description (number, category_name)
    for i in label_dict:
        folder_name = i
        number = label_dict[folder_name]
        print(folder_name + '\t' + str(number) + '\t' + category_description[number])


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
batch_size = 10

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

[label_dict, category_description] = build_label_dicts()

# prints folders, numbers and descriptions
print_Categories()

# containers holding file paths to images (jpeg filename with path, label)
images_train = load_filenames_labels('train')
images_val = load_filenames_labels('val')

# Create training queue
train_queue = deque()
for i, image_fuck in enumerate(images_train):
    if(i > 100):
        break
    train_queue.append(image_fuck)

# Create validation queue
val_queue = deque()
for i, image_fuck in enumerate(images_val):
    if (i > 100):
        break
    val_queue.append(image_fuck)

print("Number of training images: " + str(len(train_queue)))
print("Number of validation images: " + str(len(val_queue)))

x_train = []
y_train = []
x_val = []
y_val = []
# Reads first image from queue
while len(train_queue) > 0:
    item = train_queue.popleft()
    filename = item[0]
    label = item[1]
    img = image.load_img(filename, target_size=(64, 64))
    tempImg = image.img_to_array(img)
    tempImg = preprocess_input(tempImg)
    x_train.append(tempImg)
    y_train.append(label)

while len(val_queue) > 0:
    item = val_queue.popleft()
    filename = item[0]
    label = item[1]
    img = image.load_img(filename, target_size=(64, 64))
    tempImg = image.img_to_array(img)
    tempImg = preprocess_input(tempImg)
    x_val.append(tempImg)
    y_val.append(label)

# Training data
x_train = np.array(x_train)
y_train = utils.np_utils.to_categorical(y_train, 200)

# Validation data
x_val = np.array(x_val)
y_val = utils.np_utils.to_categorical(y_val, 200)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
    'tiny-imagenet-200/train',
    target_size=(64, 64), batch_size=batch_size, class_mode='categorical')


# train the model on the new data for a few epochs
predictionsModel = model.fit_generator(train_generator)
#predictionsModel = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1, validation_data=(x_val, y_val))




#
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.
'''


# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
'''