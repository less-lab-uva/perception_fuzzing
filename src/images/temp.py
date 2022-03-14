"""Helper file for generating scripts used in the paper."""
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
    true_positive_folder = Tester.working_directory + 'true_positive/'
    maybe_positive_folder = Tester.working_directory + 'maybe_positive/'
    fp_fols = [false_positive_folder, true_positive_folder, maybe_positive_folder]
    for fp_fol in fp_fols:
        os.makedirs(fp_fol, exist_ok=True)
        if os.path.exists(fp_fol[:-1] + '.txt'):
            with open(fp_fol[:-1] + '.txt', 'r') as pos_file:
                for line in pos_file.readlines():
                    false_positive_map[get_image_name(line[:-1])] = fp_fol
    key_handler_map = {
        # W means positive (instead of up since arrow keys are platform dependent)
        ord('w'): lambda tmpimage, tmp: mark_image(tmpimage, tmp, true_positive_folder),
        # S means negative (instead of down)
        ord('s'): lambda tmpimage, tmp: mark_image(tmpimage, tmp, false_positive_folder),
        # D means "maybe" or "skip" (instead of right)
        ord('d'): lambda tmpimage, tmp: mark_image(tmpimage, tmp, maybe_positive_folder),
        ord('c'): lambda tmpimage, tmp: exit(),
    }
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
            name_list.append(cur_list[2*num_per + i])
    if len(mutation_set) != len(mutation_list):
        print('Total:', len(mutation_list))
        print('Without dups:', len(mutation_set))
        print('# dups:', len(mutation_list) - len(mutation_set))
        cur_count = 0
        dup_count = []
        type_count = {type_tuple: 0 for type_tuple in type_tuple_names.keys()}
        type_counts = {type_tuple: [] for type_tuple in type_tuple_names.keys()}
        for name in name_list:
            if name in dup_list:
                cur_count += 1
                type_count[image_to_mutation[name]] += 1
            for type_tuple in type_tuple_names.keys():
                type_counts[type_tuple].append(type_count[type_tuple])
            dup_count.append(cur_count)
        values = [dup_count]
        values.extend([type_counts[type_tuple] for type_tuple in type_tuple_names.keys()])
        labels = ['Total']
        labels.extend([name for name in type_tuple_names.values()])
        fig, ax1 = plt.subplots(figsize=np.array([6.4, 4.8])*1.2)
        for value, label in zip(values, labels):
            ax1.plot(np.array(value) / len(name_list), label=label)
        ax1.set_xlabel("Number of Tests Executed")
        ax1.set_ylabel("Percent of Total Tests")
        ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        ax2 = ax1.twinx()
        for value, label in zip(values, labels):
            ax2.plot(np.array(value), label=label)
            plt.text(len(name_list)*.95, value[-1]+5, str(value[-1]))
        ax2.set_ylabel("Number of Duplicates Found")
        plt.legend()
        plt.title('Duplicates found over time per mutation')
        plt.show()
        # exit()
    else:
        print('Found no duplicate mutations')
    shuffled_name_lists = [copy.deepcopy(name_list) for _ in range(5)]
    for ind in range(len(shuffled_name_lists)):
        shuffle(shuffled_name_lists[ind])
    temp = [len(mut_fol.mutation_map) for mut_fol in folders]
    print('Pulling from each folder:')
    print(temp)
    print('Total amount pulled:', sum(temp))
    folders.insert(0, MutationFolder(Tester.CITYSCAPES_DATA_ROOT + '/sut_gt_testing'))
    worst_images = {}
    worst_images_drop = {}
    worst_count = 5
    worst_images_for_counts = []
    score_on_training = {}
    results = defaultdict(lambda: {})
    running_total = 0
    MIN_DROP = 1
    SECONDARY_DROP = 5
    # MIN_DROP = 5
    # SECONDARY_DROP = 10
    # bins = None
    # bins = np.linspace(MIN_DROP, 4.5, 20)
    # bins = np.linspace(MIN_DROP, 25.5, 25)
    # bins = [0.5, 1, 10, 100]
    bins = [1, 5, 10, 100]
    orig_bins = bins
    hatches = ['/', '\\', 'o', 'x', '|']
    sut_gt_fol = MutationFolder(Tester.CITYSCAPES_DATA_ROOT + '/sut_gt_testing')
    for sut, result_dict in Tester.compute_cityscapes_metrics(sut_gt_fol, quiet=True).items():
        image_scores = result_dict["perImageScores"]
        data = [Tester.get_score(image) for image in image_scores.items()]
        score_on_training[sut] = {Tester.get_base_file(image[0]):
                                      Tester.get_score(image)
                                  for image in image_scores.items()}
    for index, mutation_folder in enumerate(folders):
        # running_total = 0
        for sut, result_dict in Tester.compute_cityscapes_metrics(mutation_folder, quiet=True).items():
            results[sut].update(result_dict["perImageScores"])
        worst_images[mutation_folder.short_name] = {}
        worst_images_drop[mutation_folder.short_name] = {}
    data_drop_sut = {}
    data_drop_sut_map = {}
    data_drop_sut_images_primary = {}
    data_drop_sut_images_primary_less_than_secondary = {}
    data_drop_sut_images_secondary = {}
    name_list_set = frozenset(name_list)
    failure_set = set([])
    for sut, result_dict in results.items():
        image_scores = result_dict
        # data = []
        data_drop_sut[sut] = []
        data_drop_sut_map[sut] = {}
        data_drop_sut_images_primary[sut] = []
        data_drop_sut_images_primary_less_than_secondary[sut] = []
        data_drop_sut_images_secondary[sut] = []
        for image in image_scores.items():
            image_name = get_image_name(image)
            if image_name not in name_list_set:
                continue
            score = Tester.get_score(image)
            # data.append(score)
            drop = (score_on_training[sut][Tester.get_base_file(image[0])] - score)
            data_drop_sut_map[sut][image_name] = drop
            if drop >= MIN_DROP:
                failure_set.add(image_name)
                data_drop_sut[sut].append(drop)
                data_drop_sut_images_primary[sut].append(image_name)
                if drop >= SECONDARY_DROP:
                    data_drop_sut_images_secondary[sut].append(image_name)
                else:
                    data_drop_sut_images_primary_less_than_secondary[sut].append(image_name)
        print('between primary and secondary', sut, len(data_drop_sut_images_primary_less_than_secondary[sut]))
        # data = [Tester.get_score(image) for image in image_scores.items() if get_image_name(image) in name_list]
        # data_drop = [score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image)
        #              for image in image_scores.items() if (score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image)) > MIN_DROP]
        # data_drop_sut[sut] = data_drop
        # item = [item for item in image_scores.items()][0]
        # data_drop_sut_images_primary[sut] = [get_image_name(image) for image in image_scores.items() if
        #                                      (score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image)) > MIN_DROP]
        # data_drop_sut_images_secondary[sut] = [get_image_name(image) for image in image_scores.items() if
        #                                        (score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image)) > SECONDARY_DROP]
        # TODO generate timing figure
        # data_with_images_drop = sorted([(image[0], Tester.get_score(image),
        #                             score_on_training[sut][Tester.get_base_file(image[0])],
        #                                  score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image))
        #                            for image in image_scores.items()],  # sort by drop from gt to us
        #                           key=lambda x: (x[3], -x[1]), reverse=True)
        # data_with_images = sorted(
        #     [(image[0], Tester.get_score(image),
        #       score_on_training[sut][Tester.get_base_file(image[0])],
        #       score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image))
        #      for image in image_scores.items()],  # sort by worst performance
        #     key=lambda x: x[1])
        # worst_images[mutation_folder.short_name][sut] = data_with_images[:worst_count]
        # worst_images_drop[mutation_folder.short_name][sut] = data_with_images_drop[:worst_count]
        # worst_images_for_counts.extend([Tester.get_base_file(item[0]) for item in data_with_images[:worst_count]])
        # n, bins = plot_hist_as_line(data_drop, sut, 40, bins)
        print('%s: %d' % (sut, len(data_drop_sut[sut])))
        # print(data_drop)
        # running_total += np.sum(n)
    # print({item[0]:item[1] for item in data_drop_sut_map['efficientps'].items() if item[1] >= orig_bins[0]})
    # exit()
    n, bins, _ = plt.hist([data_drop_sut[sut] for sut in suts], bins=bins,
             histtype='bar', alpha=1, stacked=False, label=[sut_names[sut] for sut in suts], log=True)
    print(n)
    total_tests = len(name_list) * len(suts)
    title = 'Inconsistencies found per SUT\n%d inconsistencies found from %d mutations' % (np.sum(n), len(failure_set))
    plt.title(title)
    plt.xlabel('Percentage point drop in % pixels correct')
    plt.ylabel('Number of Inconsistencies Found')
    plt.legend(loc='upper right')
    plt.show()
    x = np.arange(len(bins)-1)
    for i in range(len(suts)):
        bars = plt.bar(x+0.16*i, n[i], label=sut_names[suts[i]], width=.16, hatch=hatches[i], log=True)
        for bar in bars:
            set_bar_text(bar)
            # plt.text(x + 0.16 * i, int(val*1.1), str(val), fontweight='bold', ha='center')
    # plt.bar([x+0.16*i for i in range(len(suts))], n, label=suts, width=.16)
    plt.title(title)
    plt.ylabel('Number of Inconsistencies Found')
    labels = []
    for i in range(len(bins) - 2):
        if i == 1:
            labels.append('[%0.1f, %0.1f)\nSignificant' % (bins[i], bins[i + 1]))
        elif i == 0:
            labels.append('[%0.1f, %0.1f)\nModerate' % (bins[i], bins[i + 1]))
        else:
            labels.append('[%0.1f, %0.1f)' % (bins[i], bins[i+1]))
    labels.append('[%0.1f, 100]\nExtreme' % (bins[-2]))
    plt.xticks(x+.16*2, labels)
    plt.xlabel('Percentage point drop in % pixels correct')
    plt.legend(loc='upper right')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6.4, 5.2)  # increase height slightly to make axis label visible
    plt.show()
    # exit()
    x = np.arange(len(type_tuple_names))
    for i in range(len(suts)):
        counts = {}
        for key in type_tuple_names.keys():
            counts[key] = len([image for image in data_drop_sut_images_primary[suts[i]] if image_to_mutation[image] == key])
        bars = plt.bar(x+0.16*i, [counts[tup] for tup in type_tuple_names.keys()],
                label=sut_names[suts[i]], width=.16, hatch=hatches[i], log=True)
        for bar in bars:
            set_bar_text(bar, zero_height=8)
    # plt.bar([x+0.16*i for i in range(len(suts))], n, label=suts, width=.16)
    plt.title('Inconsistencies Found per Mutation')
    plt.ylabel('Number of Inconsistencies Found')
    plt.xticks(x+.16*2, [name for name in type_tuple_names.values()])
    plt.xlabel('Mutation Type')
    plt.legend(loc='upper right')
    plt.show()
    # exit()
    min_count = 0
    min_counts = []
    cur_min_sut = {sut: [0 for _ in range(len(shuffled_name_lists))] for sut in suts}
    min_count_sut = {sut: [[] for _ in range(len(shuffled_name_lists))] for sut in suts}
    cur_second_sut = {sut: [0 for _ in range(len(shuffled_name_lists))] for sut in suts}
    second_count_sut = {sut: [[] for _ in range(len(shuffled_name_lists))] for sut in suts}
    second_count = 0
    second_counts = []
    # data_drop_primary = []
    # data_drop_secondary = []
    # for sut in suts:
    #     data_drop_primary.extend(data_drop_sut_images_primary[sut])
    #     data_drop_secondary.extend(data_drop_sut_images_secondary[sut])
    # print('primary', len(data_drop_primary))
    # print('secondary', len(data_drop_secondary))
    for sut in suts:
        print(sut, 'primary', len(data_drop_sut_images_primary[sut]))
        print(sut, 'second', len(data_drop_sut_images_secondary[sut]))
    # shuffle(name_list)
    for ind, cur_shuffle in enumerate(shuffled_name_lists):
        mutation_set = {sut: set() for sut in suts}
        for name in cur_shuffle:
            for sut in suts:
                prev_size = len(mutation_set[sut])
                set_key = image_to_mutation_params[name]
                mutation_set[sut].add(set_key)
                if len(mutation_set[sut]) != prev_size:  # the addition did not result in a duplicate
                    if name in data_drop_sut_images_primary[sut]:
                        min_count += 1
                        cur_min_sut[sut][ind] += 1
                    if name in data_drop_sut_images_secondary[sut]:
                        second_count += 1
                        cur_second_sut[sut][ind] += 1
                min_count_sut[sut][ind].append(cur_min_sut[sut][ind])
                second_count_sut[sut][ind].append(cur_second_sut[sut][ind])
        # if name in data_drop_primary:
        #     min_count += 1
        # if name in data_drop_secondary:
        #     second_count += 1
        # min_count += data_drop_primary.count(name)
        # second_count += data_drop_secondary.count(name)
        min_counts.append(min_count)
        second_counts.append(second_count)
    # plt.plot(min_counts)
    # plt.plot(second_counts, '--')
    # plt.title('Failures found over time')
    # plt.xlabel('Number of Tests Executed')
    # plt.ylabel('Number of Failures Found')
    # plt.legend(['%.1f percentage point' % MIN_DROP, '%.1f percentage point' % SECONDARY_DROP])
    # plt.show()
    for index, sut in enumerate(suts):
        # min_count_sut[sut][min_count_sut[sut] == 0] = np.nan
        # second_count_sut[sut][second_count_sut[sut] == 0] = np.nan
        mins = np.array([np.mean([min_count_sut[sut][ind][j] for ind in range(len(shuffled_name_lists))]) for j in
                         range(len(name_list))])
        seconds = np.array([np.mean([second_count_sut[sut][ind][j] for ind in range(len(shuffled_name_lists))]) for j in
                            range(len(name_list))])
        mins[mins == 0] = np.nan
        seconds[seconds == 0] = np.nan
        plt.plot(mins, label=sut_names[sut], c='C%d' % index)
        plt.plot(seconds, '-.', c='C%d' % index)
    # legend1 = pyplot.legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
    plt.title('Inconsistencies found over time')
    # plt.yscale('log')
    ax = plt.gca()
    # Shrink current axis's height by 5% on the top
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])

    # Place the first legend
    num_cols = 3
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(flip(handles, num_cols), flip(labels, num_cols), loc='upper center', bbox_to_anchor=(0.5, 1.25),
                        ncol=num_cols)

    # Place the second legend
    legend2 = ax.legend(['%.1f percentage point' % MIN_DROP, '%.1f percentage point' % SECONDARY_DROP], loc=0)

    # Set the labels and limits
    ax.set_xlabel("Number of Tests Executed")
    ax.set_ylabel("Number of Inconsistencies Found")
    # ax.set_ylim([-0.05, 1.05])
    # Readd legend 1 before showing
    plt.gca().add_artist(legend1)
    plt.show()
    # exit()
    random.seed(8459741097717723050)
    Tester.cityscapes_mutator.blurred = True
    count = 1
    for sut in suts:
        extreme_counts = defaultdict(lambda: 0)
        sig_counts = defaultdict(lambda: 0)
        for image in data_drop_sut_images_secondary[sut]:
            image_file = name_to_image_file[image]
            blurred_image = get_blurred_image(image_to_mutation_params[image])
            extreme_folder = Tester.working_directory + ('/inconsistencies/ExtremeInconsistenciesAt10Plus/%s/' % sut)
            sig_folder = Tester.working_directory + ('/inconsistencies/SignificantInconsistenciesAt5_10/%s/' % sut)
            os.makedirs(extreme_folder, exist_ok=True)
            os.makedirs(sig_folder, exist_ok=True)
            gt_file = image_file.replace('mutations/', 'mutations_gt/').replace('_edit.png', '_mutation_gt.png')
            sut_perception = image_file.replace('mutations/', sut + '/').replace('_edit.png', '_edit_prediction.png')
            orig_perception = get_orig_pred(sut, image_file)
            # print(data_drop_sut_map[sut][image], orig_bins[-2], data_drop_sut_map[sut][image] >= orig_bins[-2])
            if data_drop_sut_map[sut][image] >= orig_bins[-2]:
                cv2.imwrite(extreme_folder + get_image_name(image_file), blurred_image)
                shutil.copy2(gt_file, extreme_folder + get_image_name(gt_file))
                shutil.copy2(sut_perception, extreme_folder + get_image_name(sut_perception) + '_edit_prediction.png')
                shutil.copy2(orig_perception, extreme_folder + get_image_name(orig_perception) + '_orig_prediction.png')
            else:
                cv2.imwrite(sig_folder + get_image_name(image_file), blurred_image)
                shutil.copy2(gt_file, sig_folder + get_image_name(gt_file))
                shutil.copy2(sut_perception, sig_folder + get_image_name(sut_perception) + '_edit_prediction.png')
                shutil.copy2(orig_perception, sig_folder + get_image_name(orig_perception) + '_orig_prediction.png')
            if get_image_name(image_file) in false_positive_map:
                which = false_positive_map[get_image_name(image_file)]
            else:
                which = show_image(blurred_image, image_file, count, key_handler_map)
                false_positive_map[get_image_name(image_file)] = which
            count += 1
            if data_drop_sut_map[sut][image] >= orig_bins[-2]:
                extreme_counts[which] += 1
            else:
                sig_counts[which] += 1
        print('Extreme counts for %s secondary bin' % sut)
        print(100-100*extreme_counts['/data/world_fuzzing/true_positive/'] / max(1, sum([val for val in extreme_counts.values()])),sum([val for val in extreme_counts.values()]), dict(extreme_counts))
        print('Significant counts for %s secondary bin' % sut)
        print(100-100*sig_counts['/data/world_fuzzing/true_positive/'] / max(1,sum([val for val in sig_counts.values()])),sum([val for val in sig_counts.values()]), dict(sig_counts))
    random_sample = 0.1
    for sut in suts:
        counts = defaultdict(lambda: 0)
        sample_size = int(round(len(data_drop_sut_images_primary_less_than_secondary[sut])) * random_sample)
        images = random.sample(data_drop_sut_images_primary_less_than_secondary[sut], sample_size)
        for image in images:
            image_file = name_to_image_file[image]
            gt_file = image_file.replace('mutations/', 'mutations_gt/').replace('_edit.png', '_mutation_gt.png')
            sut_perception = image_file.replace('mutations/', sut + '/').replace('_edit.png', '_edit_prediction.png')
            orig_perception = get_orig_pred(sut, image_file)
            blurred_image = get_blurred_image(image_to_mutation_params[image])
            reg_folder = Tester.working_directory + ('/inconsistencies/ModerateInconsistenciesAt1_5/%s/' % sut)
            os.makedirs(reg_folder, exist_ok=True)
            cv2.imwrite(reg_folder + get_image_name(image_file), blurred_image)
            shutil.copy2(gt_file, reg_folder + get_image_name(gt_file))
            shutil.copy2(sut_perception, reg_folder + get_image_name(sut_perception) + '_edit_prediction.png')
            shutil.copy2(orig_perception, reg_folder + get_image_name(orig_perception) + '_orig_prediction.png')
            if get_image_name(image_file) in false_positive_map:
                which = false_positive_map[get_image_name(image_file)]
            else:
                which = show_image(blurred_image, image_file, count, key_handler_map)
                false_positive_map[get_image_name(image_file)] = which
            count += 1
            counts[which] += 1
        print('Counts for %s primary bin' % sut)
        print(sum([val for val in counts.values()]), dict(counts))
        print(100-100*counts['/data/world_fuzzing/true_positive/'] / max(1,sum([val for val in counts.values()])),
              sum([val for val in counts.values()]), dict(counts))
    # print(worst_images)
    # print(worst_images_drop)
    # print(Counter(worst_images_for_counts))
    # pp = pprint.PrettyPrinter(indent=2, compact=True)
    # print('Worst:')
    # pp.pprint(worst_images)
    # print('Overall Worst:')
    # pp.pprint(Counter(worst_images_for_counts))
    # pp.pprint({mutation: Counter(list(itertools.chain(*[[Tester.get_base_file(item[0]) for item in lst] for lst in suts.values()]))) for mutation, suts in worst_images.items()})
    # print()
    # print('Worst Drop:')
    # pp.pprint(worst_images_drop)
    # pp.pprint(
    #     {mutation: Counter(list(itertools.chain(*[[Tester.get_base_file(item[0]) for item in lst] for lst in suts.values()])))
    #      for mutation, suts in worst_images_drop.items()})
