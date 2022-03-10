"""
Script that runs the timing tests described in Table 3 in Section 6.4 of the paper.
"""
import itertools
import tempfile
from collections import defaultdict, Counter
import time

import numpy as np
import matplotlib.pyplot as plt
import os

from image_mutator import MutationFolder, Mutation, MutationType
from tester import Tester

NUM_TO_GEN = 100
output_file = '/data/time_tests/time_output.txt'
os.makedirs('/data/time_tests/', exist_ok=True)

def pprint(output=''):
    print(output)
    with open(output_file, 'a+') as f:
        f.write('%s\n' % output)


def run_add_car(test_num):
    directory = '/data/time_tests/add_car_%d' % test_num
    Tester.initialize(working_directory=directory)
    start_time = time.time_ns()
    for _ in range(NUM_TO_GEN):
        Tester.cityscapes_mutator.apply_mutation(MutationType.ADD_OBJECT, {'semantic_label': 'car'})
    add_car_time = (time.time_ns() - start_time) / NUM_TO_GEN
    pprint('Test %d' % test_num)
    pprint('Time per Add Car: %f s' % (add_car_time / 1e9))
    pprint()


def run_add_person(test_num):
    directory = '/data/time_tests/add_person_%d' % test_num
    Tester.initialize(working_directory=directory)
    start_time = time.time_ns()
    for _ in range(NUM_TO_GEN):
        Tester.cityscapes_mutator.apply_mutation(MutationType.ADD_OBJECT, {'semantic_label': 'person'})
    add_person_time = (time.time_ns() - start_time) / NUM_TO_GEN
    pprint('Test %d' % test_num)
    pprint('Time per Add Person: %f s' % (add_person_time / 1e9))
    pprint()


def run_color_car(test_num):
    directory = '/data/time_tests/car_color_%d' % test_num
    Tester.initialize(working_directory=directory)
    start_time = time.time_ns()
    for _ in range(NUM_TO_GEN):
        Tester.cityscapes_mutator.apply_mutation(MutationType.CHANGE_COLOR, {'semantic_label': 'car'})
    color_car_time = (time.time_ns() - start_time) / NUM_TO_GEN
    pprint('Test %d' % test_num)
    pprint('Time per Color Car: %f s' % (color_car_time / 1e9))
    pprint()


def run_suts(test_num, index, mutation):
    directory = '/data/time_tests/sut_runs_%d_%d' % (test_num, index)
    Tester.initialize(working_directory=directory)
    Tester.run_fuzzer(folders_to_run=1, generate_only=True, num_per_mutation=NUM_TO_GEN, mutations_to_run=[mutation])
    mutation_folder = Tester.mutation_folders[0]
    # Tester.sut_manager.run_suts(Tester.mutation_folders[0])
    for sut in Tester.sut_manager.suts:
        pprint('---- Running SUT %s ----' % sut.name)
        start_time = time.time_ns()
        sut.run_semantic_seg(mutation_folder.folder, mutation_folder.get_sut_folder(sut.name))
        sut_time = (time.time_ns() - start_time) / NUM_TO_GEN
        pprint('Test %d' % test_num)
        pprint('Time per image for SUT %s on mutation %s: %f s' % (sut.name, str(mutation), sut_time / 1e9))
        pprint()


if __name__ == '__main__':
    mutations_to_run = [
        (MutationType.CHANGE_COLOR, {'semantic_label': 'car'}),
        (MutationType.ADD_OBJECT, {'semantic_label': 'person'}),
        (MutationType.ADD_OBJECT, {'semantic_label': 'car'})
    ]
    for test_num in range(10):
        pprint('Running test: %d' % (test_num + 1))
        run_add_car(test_num)
        run_add_person(test_num)
        run_color_car(test_num)
        for index, mutation in enumerate(mutations_to_run):
            run_suts(test_num, index, mutation)
