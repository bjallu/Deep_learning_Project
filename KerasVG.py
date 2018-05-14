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

def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()

model = VGG16(include_top=True, weights='imagenet')

def save_model(model_in):
    currentTime = int(time.time())
    model_name = str(currentTime) + 'Model.h5'
    model_in.save(model_name)

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def predict(image_path, verbose = 1):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)

    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    if(verbose != 0):
        # Plot the image.
        plt.imshow(img_resized)
        plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred, top=1000)

    # Print the predictions.
    if(verbose != 0):
        for code, name, score in pred_decoded:
            print("{0:>6.2%} : {1}".format(score, name))

    return pred_decoded

def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def example_errors(model_in):
    # The Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset it first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginning so we know exactly which input-images were used.
    generator_train_histogram.reset()

    # Predict the classes for all images in the test-set.
    y_pred = model_in.predict_generator(generator_train_histogram,
                                         steps=steps_train_histogram)

    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred, axis=1)

    # Plot examples of mis-classified images.
    # plot_example_errors(cls_pred)

    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)

def save_history(history):
    currentTime = int(time.time())
    filename = str(currentTime) + 'History'
    filename = open(filename + ".pickle", 'wb')
    pickle.dump(history, filename)
    filename.close()

def print_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_train_histogram,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # write the confusion matrix to file
    currentTime = int(time.time())
    filename = str(currentTime) + 'ConfusionMatrix.txt'

    thefile = open(filename, 'w+')
    for item in cm:
        for number in item:
            thefile.write("%s\t" % number)
        thefile.write("\n")

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        thefile.write("%s\n" % class_name)

    thefile.close()

    filenameNumpy = str(currentTime) + 'ConfusionMatrix.npy'
    np.save(filenameNumpy, cm)

def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images.
    images = load_images(image_paths=image_paths[0:9])

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

input_shape = model.layers[0].output_shape[1:3]

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
batch_size = 160

if True:
    save_to_dir = None
else:
    save_to_dir='augmented_images/'

train_dir = '../tiny-imagenet-200/train'
test_dir = '../tiny-imagenet-200/val'

#train_dir = './knifey-spoony/train'
#test_dir = './knifey-spoony/test'

generator_train = datagen_train.flow_from_directory(
    directory=train_dir,
    target_size=input_shape,
    batch_size=batch_size,
    shuffle=True,
    save_to_dir=save_to_dir)

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

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_train_histogram = generator_train_histogram.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())
num_classes = len(class_names)

# Load the first images from the train-set.
images = load_images(image_paths=image_paths_train[0:9])

# Get the true classes for those images.
cls_true = cls_train[0:9]

# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true, smooth=True)


model.summary()
transfer_layer = model.get_layer('block5_pool')

conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

# Start a new Keras Sequential model.
main_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
main_model.add(conv_model)
main_model.add(Flatten())
# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
main_model.add(Dense(1024, activation='relu'))
#main_model.add(Dropout(0.5))
main_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-4)
steps_per_epoch = 100000 / batch_size

loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy', 'top_k_categorical_accuracy']

print_layer_trainable()

conv_model.trainable = False
for layer in model.layers:
    layer.trainable = False

print_layer_trainable()

main_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

main_model.summary()

epochs = 1

main_history = main_model.fit_generator(generator=generator_train,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

main_result = main_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(main_result[1]))

conv_model.trainable = True

for layer in conv_model.layers:
    trainable = ('block5' in layer.name or 'block4' in layer.name)
    layer.trainable = trainable

print_layer_trainable()

optimizer_fine = Adam(lr=1e-7)

main_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

main_fine_history = main_model.fit_generator(generator=generator_train,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

main_fine_result = main_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(main_fine_result[1]))

# Confusion Matrix
example_errors(main_model)

# Saves history to file
save_model(main_model)
save_history(main_fine_history)