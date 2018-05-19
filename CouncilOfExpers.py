import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from input_pipe import *
from keras.models import load_model

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
np.save('BaseModelesults.npy', base_result)
del base_model

np.save('correctClasses.npy', np.asarray(generator_test.classes))


# base_final_result = base_model_final.predict_generator(generator_test, steps=steps_test, verbose=1)
# generator_test.reset()

council_resuts = np.zeros_like(base_result)

for i in range(34, number_of_experts):
    expertName = str(i) + "MichaelPhelps.h5"
    model = load_model(expertName)
    expert_result = model.predict_generator(generator_test, steps=steps_test, verbose=1)
    generator_test.reset()

    np.save(str(i) + 'ExpertResults.npy', expert_result)
    council_resuts += expert_result

    del model
    del expert_result
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
    true_p = generator_test.classes[i]

    if(true_p == base_p):
        base_correct += 1

    '''
    if(true_p == base_final_p):
        base_final_correct += 1
    '''
    if (true_p == expert_p):
        expert_correct += 1

    total += 1

    results = [base_correct/total, expert_correct/total]
    # results = [base_correct / total, base_final_correct / total, expert_correct / total, expert_correct_squared / total]
    print(results)
    total_results.append(results)

file = open('MichaelPhelps36.txt', 'w+')

for line in total_results:
    file.write(str(line[0]) + '\t' + str(line[1]) + '\n')

file.close()