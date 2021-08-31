import itertools
import pprint
import tempfile
from collections import defaultdict, Counter
import time

import numpy as np
import matplotlib.pyplot as plt


from image_mutator import MutationFolder, Mutation, MutationType
from tester import Tester

if __name__ == '__main__':
    # Tester.run_fuzzer(generate_only=False)
    for test_num in range(5):
        print('Running test:', test_num + 1)
        NUM_TO_GEN = 1000
        with tempfile.TemporaryDirectory() as directory:
            Tester.initialize(working_directory=directory)
            start_time = time.time_ns()
            for _ in range(NUM_TO_GEN):
                Tester.cityscapes_mutator.apply_mutation(MutationType.ADD_OBJECT, {'semantic_label': 'car'})
            add_car_time = (time.time_ns() - start_time) / NUM_TO_GEN
            print('Time per Add Car: %f s' % (add_car_time / 1e9))
        with tempfile.TemporaryDirectory() as directory:
            Tester.initialize(working_directory=directory)
            start_time = time.time_ns()
            for _ in range(NUM_TO_GEN):
                Tester.cityscapes_mutator.apply_mutation(MutationType.ADD_OBJECT, {'semantic_label': 'person'})
            add_person_time = (time.time_ns() - start_time) / NUM_TO_GEN
            print('Time per Add Person: %f s' % (add_person_time / 1e9))
        with tempfile.TemporaryDirectory() as directory:
            Tester.initialize(working_directory=directory)
            start_time = time.time_ns()
            for _ in range(NUM_TO_GEN):
                Tester.cityscapes_mutator.apply_mutation(MutationType.CHANGE_COLOR, {'semantic_label': 'car'})
            color_car_time = (time.time_ns() - start_time) / NUM_TO_GEN
            print('Time per Color Car: %f s' % (color_car_time / 1e9))
        with tempfile.TemporaryDirectory() as directory:
            Tester.initialize(working_directory=directory)
            Tester.run_fuzzer(folders_to_run=1, generate_only=True, num_per_mutation=NUM_TO_GEN)
            mutation_folder = Tester.mutation_folders[0]
            # Tester.sut_manager.run_suts(Tester.mutation_folders[0])
            for sut in Tester.sut_manager.suts:
                print('---- Running SUT %s ----' % sut.name)
                start_time = time.time_ns()
                sut.run_semantic_seg(mutation_folder.folder, mutation_folder.get_sut_folder(sut.name))
                sut_time = (time.time_ns() - start_time) / NUM_TO_GEN
                print('Time per image for SUT %s: %f s' % (sut.name, sut_time / 1e9))
