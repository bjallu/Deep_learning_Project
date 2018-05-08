

from input_pipe import *
from collections import deque
import matplotlib.image as mpimg
import argparse
import os
import shutil


def print_Categories():
    # label_dict (folder_name, number)
    # class_description (number, category_name)
    id_dict = {}
    for i in label_dict:
        folder_name = i
        number = label_dict[folder_name]
        id_dict[str(number)] = folder_name
        print(folder_name + '\t' + str(number) + '\t' + category_description[number])
    return id_dict

if __name__ == '__main__':

    # loads the folders, numbers and descriptions into dictionaries
    [label_dict, category_description] = build_label_dicts()
    # containers holding file paths to images (jpeg filename with path, label)
    images_train = load_filenames_labels('train')
    images_val = load_filenames_labels('val')
    id_dict = print_Categories()

    path = '../tiny-imagenet-200/val/'
    for i in images_val:
        id = i[1]
        folder = id_dict[id]

        # if the folder does not exists
        path_to_folder = path + folder
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
            path_to_folder = path_to_folder + "/images"
            os.makedirs(path_to_folder)
        else:
            path_to_folder = path_to_folder + "/images"

        currentLocation = i[0]
        fileName = currentLocation.rsplit('/', 1)[-1]
        moveToLocation = path_to_folder + "/" + fileName
        if not os.path.exists(moveToLocation):
            shutil.move(currentLocation, moveToLocation)

