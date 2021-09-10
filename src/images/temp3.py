import copy
import itertools
import math
import os
import pprint
import random
import shutil
import time
from collections import defaultdict, Counter
from random import shuffle
from typing import Callable, Dict

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


from src.images.image_mutator import MutationFolder, Mutation, MutationType
from src.images.tester import Tester, plot_hist_as_line


def get_orig_pred(sut, image_file):
    image_file = get_image_name(image_file)
    image_file = image_file[image_file.find('_')+1:].replace('_edit.png', '_edit_prediction.png')
    return Tester.CITYSCAPES_DATA_ROOT + '/sut_gt_testing/' + sut + '/' + image_file


def mark_image(image, image_file, folder):
    img_to_mark = get_image_name(image_file)
    if folder[-1] != '/':
        folder += '/'
    with open(folder[:-1] + '.txt', 'a') as file:
        file.write('%s\n' % image_file)
    # shutil.copy2(image_file, folder + img_to_mark)
    cv2.imwrite(folder + img_to_mark, image)
    return folder


def show_image(img, image_file: str, count, key_handler: Dict[int, Callable[[str], str]]):
    smaller = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    # https://stackoverflow.com/questions/47234590/cv2-imshow-image-window-placement-is-outside-of-viewable-screen
    win_name = 'Image %d' % count
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 50, 50)
    cv2.imshow(win_name, smaller)
    key_pressed = -1
    while key_pressed not in key_handler:
        key_pressed = cv2.waitKey(0)
        cv2.destroyAllWindows()
    time.sleep(1)
    return key_handler[key_pressed](img, image_file)


def get_type_tuple(mutation_params_map):
    # print(mutation_params_map)
    # print(mutation_params_map['mutation_type'], mutation_params_map['semantic_label'])
    return mutation_params_map['mutation_type'], mutation_params_map['semantic_label']


def set_bar_text(bar, zero_height=0.8):
    height = bar.get_height()
    label = str(int(height))
    for _ in range(1 * (3 - len(label))):
        label = ' ' + label
    y = zero_height if height == 0 else height * 1.02  # mult becomes add in log scale
    plt.text(bar.get_x(), y, label, va='bottom')


def get_blurred_image(params):
    Tester.cityscapes_mutator.blurred = True
    if type(params) == tuple:
        params = {item[0]: item[1] for item in params}
    mut_type = Tester.cityscapes_mutator.get_mutation_type(params['mutation_type'].name)
    del params['mutation_type']
    return Tester.cityscapes_mutator.apply_mutation(mut_type, params).edit_image.image


def get_image_name(image):
    if type(image) == tuple:
        short = image[0]
    else:
        short = image
    if '/' in short:
        short = short[short.rfind('/')+1:]
    short = short.replace('_edit_prediction.png', '')
    return short


if __name__ == '__main__':
    # Tester.run_fuzzer(generate_only=True)
    Tester.initialize(load_mut_fols=False)
    mutation_type_map = defaultdict(lambda: 0)
    max_len = {
        (MutationType.ADD_OBJECT, 'car'): 50000,
        (MutationType.ADD_OBJECT, 'person'): 50000,
        (MutationType.CHANGE_COLOR, 'car'): 50000,
    }
    type_tuple_names = {
        (MutationType.ADD_OBJECT, 'car'): 'Add Car',
        (MutationType.ADD_OBJECT, 'person'): 'Add Person',
        (MutationType.CHANGE_COLOR, 'car'): 'Car Color',
    }
    name_list = []
    suts = [
        'nvidia-semantic-segmentation',
        'efficientps',
        'decouple_segnet',
        'nvidia-sdcnet',
        'hrnet'
    ]
    sut_names = {
        'nvidia-semantic-segmentation': 'NVIDIA SemSeg',
        'efficientps': 'EfficientPS',
        'decouple_segnet': 'DecoupleSegNet',
        'nvidia-sdcnet': 'SDCNet',
        'hrnet': 'HRNetV2+OCR'
    }
    false_positive_map = {}  # map from image name to folder
    false_positive_folder = Tester.working_directory + 'false_positive/'
    false_positive_folder_blurred = Tester.working_directory + 'false_positive_blurred/'
    true_positive_folder_blurred = Tester.working_directory + 'true_positive_blurred/'
    os.makedirs(false_positive_folder_blurred, exist_ok=True)
    os.makedirs(true_positive_folder_blurred, exist_ok=True)
    true_positive_folder = Tester.working_directory + 'true_positive/'
    maybe_positive_folder = Tester.working_directory + 'maybe_positive/'
    fp_fols = [false_positive_folder, true_positive_folder, maybe_positive_folder]
    for fp_fol in fp_fols:
        os.makedirs(fp_fol, exist_ok=True)
        if os.path.exists(fp_fol[:-1] + '.txt'):
            with open(fp_fol[:-1] + '.txt', 'r') as pos_file:
                for line in pos_file.readlines():
                    false_positive_map[get_image_name(line[:-1])] = fp_fol
    folders = []
    image_to_mutation = {}
    image_to_mutation_params = {}
    name_to_image_file = {}
    mutation_list = []
    mutation_set = set()
    dup_list = []
    for i in range(25):
        mut_fol = MutationFolder((Tester.working_directory + "/set_%d/") % i, perform_init=False)
        folders.append(mut_fol)
        cur_list = []
        with open(mut_fol.mutation_logs, 'r') as f:
            temp = eval('[' + f.read() + ']')
            for name, mutation_params in temp:
                type_tuple = get_type_tuple(mutation_params)
                if mutation_type_map[type_tuple] >= max_len[type_tuple]:
                    continue
                set_key = tuple(sorted(mutation_params.items()))
                image_to_mutation_params[name] = tuple(sorted(mutation_params.items()))
                mutation_list.append(set_key)
                prev_size = len(mutation_set)
                mutation_set.add(set_key)
                if len(mutation_set) == prev_size:  # the addition resulted in a duplicate
                    continue
                    # dup_list.append(name)
                mutation_type_map[type_tuple] += 1
                cur_list.append(name)
                image_to_mutation[name] = type_tuple
                name_to_image_file[name] = mut_fol.folder + name + '_edit.png'
                # name_list.append(name)
                mut_fol.mutation_map[name] = Mutation.from_params(name, mutation_params, mut_fol)
        # shuffle(cur_list)
        # name_list.extend(cur_list)
        num_per = len(cur_list) // 3
        for i in range(num_per):
            name_list.append(cur_list[i])
            name_list.append(cur_list[num_per + i])
            name_list.append(cur_list[2 * num_per + i])
    print([i for i in false_positive_map.items()][0])
    for image, folder in false_positive_map.items():
        image_name = image.replace('_edit.png', '')
        blurred = get_blurred_image(image_to_mutation_params[image_name])
        if folder == false_positive_folder or folder == maybe_positive_folder:
            cv2.imwrite(false_positive_folder_blurred + image, blurred)
        else:
            cv2.imwrite(true_positive_folder_blurred + image, blurred)

