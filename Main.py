from input_pipe import *
from collections import deque
import matplotlib.image as mpimg
import argparse


def print_Categories():
    # label_dict (folder_name, number)
    # class_description (number, category_name)
    for i in label_dict:
        folder_name = i
        number = label_dict[folder_name]
        print(folder_name + '\t' + str(number) + '\t' + category_description[number])


def create_dictionaries():
    # label_dict (folder_name, number)
    # class_description (number, category_name)
    folder_to_id = {}
    id_to_folder = {}
    for i in label_dict:
        folder_name = i
        number = label_dict[folder_name]
        id_to_folder[str(number)] = folder_name
        folder_to_id[folder_name] = number

    return [id_to_folder, folder_to_id]




if __name__ == '__main__':

    # loads the folders, numbers and descriptions into dictionaries
    [label_dict, category_description] = build_label_dicts()

    # prints folders, numbers and descriptions
    print_Categories()

    # containers holding file paths to images (jpeg filename with path, label)
    images_train = load_filenames_labels('train')
    images_val = load_filenames_labels('val')

    # Create training queue
    train_queue = deque()
    for i_train in images_train:
        train_queue.append(i_train)

    # Create validation queue
    val_queue = deque()
    for i_val in images_val:
        val_queue.append(i_val)

    print("Number of training images: " + str(len(train_queue)))
    print("Number of validation images: " + str(len(val_queue)))

    # Reads first image from queue
    [image, label] = read_image(train_queue, 'train')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """
    )

    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )

    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )



def imageClass(image_path):
    folder = image_path.rsplit('/', 1)[-1]
    folder = folder.rsplit('_', 1)[0]
    return [label_dict[folder], folder]


############### Start ###############
'''
[label_dict, category_description] = build_label_dicts()
# predict() uses only the pretrained model to
# predict categories based on Imagenet's 1000 category labels

for classifyImage in image_paths_train:
    [id, folder] = imageClass(classifyImage)
    imageNetPrediction = predict(image_path=classifyImage, verbose = 0)
    among200 = 0
    for predictions5 in imageNetPrediction:
        for p in predictions5:
            predictFolder = p[0]
            if(predictFolder in label_dict):
                if(folder == predictFolder):
                    print("Prediction in top: " + str(among200+1))
                    break
                among200 += 1

'''