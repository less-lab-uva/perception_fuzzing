import glob
import itertools
import math
import re
from collections import Counter
import copy
import pprint
import time
from shutil import copyfile

import cv2
import numpy as np
from skimage.metrics import structural_similarity

import traceback
from multiprocessing import Pool
from image_mutator import *
from src.sut_runner.sut_runner import SUTRunner
from src.sut_runner.decouple_segnet import DecoupleSegNet
from src.sut_runner.efficientps import EfficientPS
from src.sut_runner.hrnet import HRNet
from src.sut_runner.nvidia_sdcnet import NVIDIASDCNet
from src.sut_runner.nvidia_semantic_segmentation import NVIDIASemSeg
from src.sut_runner.sut_manager import SUTManager
from typing import List

current_file_path = Path(__file__)
sys.path.append(str(current_file_path.parent.parent.absolute()) + '/cityscapesScripts')
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import args as orig_cityscapes_eval_args


def save_paletted_image(old_file_path, new_file_path):
    Image(image=cv2.imread(old_file_path), image_file=new_file_path).save_paletted_image()


class Tester:
    CITYSCAPES_DATA_ROOT = None
    cityscapes_mutator = None
    BEST_SUT = None  # the SUT to be used for criteria of whether or not to include in mutation resources
    HIGH_DNC = []
    sut_list = None
    sut_manager = None
    pool_count = None
    _initialized = False
    SCORE_THRESHOLD = 95  # As a percent out of 100, this is the value used in the paper
    cityscapes_results = None
    __cityscapes_runs_folder = None
    working_directory = None
    mutation_folders = None

    @staticmethod
    def initialize(best_sut: SUTRunner=None, sut_list: List[SUTRunner]=None, working_directory=None, cityscapes_data_root=None,
                   pool_count=30, score_threshold=None, load_mut_fols=True):
        Tester._initialized = True
        Tester.working_directory = working_directory if working_directory is not None else os.environ('WORKING_DIR')
        if Tester.working_directory[-1] != '/':
            Tester.working_directory += '/'
        os.makedirs(Tester.working_directory, exist_ok=True)
        Tester.BEST_SUT = best_sut
        if best_sut not in sut_list:
            sut_list.insert(0, best_sut)
        Tester.sut_list = sut_list
        Tester.sut_manager = SUTManager(Tester.sut_list)
        Tester.pool_count = pool_count
        Tester.CITYSCAPES_DATA_ROOT = cityscapes_data_root if cityscapes_data_root is not None else os.environ('CITYSCAPES_DATA_ROOT')

        cityscapes_runs_folder = Tester.__get_cityscapes_runs_folder()
        all_paths_exist = True
        for sut in Tester.sut_list:
            if not os.path.exists(cityscapes_runs_folder.get_sut_raw_results(sut.name)):
                all_paths_exist = False
        if all_paths_exist:
            Tester.cityscapes_results = {}
            # define these so that the eval below has them in scope
            nan = np.nan
            float32 = np.float32
            for sut in Tester.sut_list:
                with open(cityscapes_runs_folder.get_sut_raw_results(sut.name)) as f:
                    Tester.cityscapes_results[sut.name] = eval(f.read())
        else:
            # this will run and return, saving for next time
            Tester.cityscapes_results = Tester.run_on_cityscapes_benchmark()
        if score_threshold is not None:
            Tester.SCORE_THRESHOLD = score_threshold
        good_files = [Tester.get_base_file(image)
                      for image, score in Tester.cityscapes_results[Tester.BEST_SUT.name]["perImageScores"].items()
                      if Tester.get_score(score) > Tester.SCORE_THRESHOLD]
        print('Found %d good files' % len(good_files))
        with open(Tester.working_directory + 'sut_gt_hist.txt', 'w') as f:
            f.write(str([(Tester.get_base_file(image), Tester.get_score(score))
                         for image, score in
                         Tester.cityscapes_results[Tester.BEST_SUT.name]["perImageScores"].items()]))

        Tester.cityscapes_mutator = CityscapesMutator(Tester.CITYSCAPES_DATA_ROOT, good_files)
        if load_mut_fols:
            Tester.mutation_folders = [MutationFolder(subdir)
                                       for subdir in glob.glob(Tester.working_directory + "/set_*/")]
            Tester.mutation_folders.sort(key=lambda x: int(re.search(r'.*set_(\d+)', x.base_folder).group(1)))

    @staticmethod
    def compute_scores_vs_baseline(mutation_folders: List[MutationFolder], min_drop=1):
        if not Tester._initialized:
            Tester.initialize()
        score_on_baseline = {}
        for sut, result_dict in Tester.cityscapes_results.items():
            image_scores = result_dict["perImageScores"]
            score_on_baseline[sut] = {Tester.get_base_file(image[0]):
                                          Tester.get_score(image[1])
                                      for image in image_scores.items()}
        data_with_images_drop = {}  # list of tuples: (image, score, baseline_score, baseline_score - score)
        data_with_images_drop_mutation = defaultdict(lambda:[])  # list of tuples: (image, score, baseline_score, baseline_score - score)
        data_with_images = {}
        for sut in Tester.sut_list:
            data_with_images_drop[sut.name] = []
            data_with_images[sut.name] = []
        mutation_map = {}
        for mut_fol in mutation_folders:
            mutation_map.update(mut_fol.mutation_map)
            # Tester.sut_manager.run_suts(mut_fol)
            results = Tester.compute_cityscapes_metrics(mut_fol)
            for sut, result_dict in results.items():
                image_scores = result_dict["perImageScores"]
                data_with_images_drop[sut].extend([(image[0], Tester.get_score(image[1]),
                                                        score_on_baseline[sut][Tester.get_base_file(image[0])],
                                                        score_on_baseline[sut][Tester.get_base_file(image[0])]
                                                        - Tester.get_score(image[1]))
                                                        for image in image_scores.items()])
                data_with_images[sut].extend([(image[0], Tester.get_score(image[1]),
                                                   score_on_baseline[sut][Tester.get_base_file(image[0])],
                                                   score_on_baseline[sut][Tester.get_base_file(image[0])]
                                                   - Tester.get_score(image[1]))
                                                   for image in image_scores.items()])
                for image in image_scores.items():
                    name = image[0]
                    name = name[name.rfind('/')+1:-len('_edit_prediction.png')]
                    mutation_params = mut_fol.mutation_map[name].params
                    mutation = (mutation_params['mutation_type'], mutation_params['semantic_label'])
                    data_with_images_drop_mutation[mutation].append((image[0], Tester.get_score(image[1]),
                                                        score_on_baseline[sut][Tester.get_base_file(image[0])],
                                                        score_on_baseline[sut][Tester.get_base_file(image[0])]
                                                        - Tester.get_score(image[1])))
        bins = np.linspace(0.5, 25.5, 25)
        bin_count = 40
        count_over_min = {}
        for sut in Tester.sut_list:
            data_with_images_drop[sut.name].sort(key=lambda x: (x[3], -x[1]), reverse=True)
            data_with_images[sut.name].sort(key=lambda x: x[1])
            over_min = [(item[0], item[3]) for item in data_with_images_drop[sut.name] if item[3] >= min_drop]
            print(sut.name, len(over_min))
            count_over_min[sut.name] = (len(over_min), over_min)
        plot_hist_as_line([[item[3] for item in data_with_images_drop_mutation[mutation]] for mutation in data_with_images_drop_mutation.keys()],
                          [str(mutation) for mutation in data_with_images_drop_mutation.keys()], bin_count, bins, log=False)
        plt.xlabel('Percentage point drop in % pixels correct')
        plt.ylabel('Log Count of Images')
        plt.legend(loc='upper right')
        plt.show()


    @staticmethod
    def get_next_mutation_folder(check_len):
        for mutation_folder in Tester.mutation_folders:
            if len(mutation_folder.mutation_map) < check_len:
                return mutation_folder
        mutation_folder = MutationFolder(Tester.working_directory + ('set_%d' % len(Tester.mutation_folders)))
        Tester.mutation_folders.append(mutation_folder)
        return mutation_folder

    @staticmethod
    def get_score(image):
        if type(image) == tuple:
            image = image[1]
        if image['nbNotIgnoredPixels'] == 0:
            return 0
        return 100.0 * (1.0 - image['nbCorrectPixels'] / image['nbNotIgnoredPixels'])

    @staticmethod
    def get_base_file(long_file: str):
        long_file = long_file[long_file.rfind('/') + 1:]
        if '-' in long_file:
            # strip uuid
            long_file = long_file[37:]
        long_file = long_file.replace('_edit_prediction.png', '')
        return long_file

    @staticmethod
    def create_images(folder: str, count, start_num, mutation_type: MutationType, arg_dict):
        """Generate the specified number of mutations"""
        # since we know that the folder being passed to this method has already been used to create
        # a MutationFolder before, we don't want to waste time performing the initialization
        mutation_folder = MutationFolder(folder, perform_init=False)
        i = start_num
        while i < start_num + count:
            try:
                mutation = Tester.cityscapes_mutator.apply_mutation(mutation_type, arg_dict)
                if mutation is not None:
                    mutation_folder.add_mutation(mutation)
                    mutation.save_images(free_mem=True)
                else:
                    i -= 1  # don't advance if we didn't get a new mutation
            except Exception as e:
                traceback.print_exc(e)
                pass
            i += 1
        return_map = mutation_folder.mutation_map
        del mutation_folder
        return return_map

    @staticmethod
    def create_fuzz_images(mutation_folder: MutationFolder, count, mutation_type: MutationType, arg_dict):
        """Generate mutated images using a thread pool for increased speed"""
        count_per = int(math.ceil(count / Tester.pool_count))
        results = []
        orig_count = count
        if Tester.pool_count == 1:
            mutation_folder.merge_folder(Tester.create_images(mutation_folder.base_folder,
                                                              count, 0, mutation_type, arg_dict))
        else:
            with Pool(Tester.pool_count) as pool:
                while count > 0:
                    res = pool.apply_async(Tester.create_images, (mutation_folder.base_folder, min(count, count_per),
                                                                  orig_count - count, mutation_type, arg_dict))
                    results.append(res)
                    count -= count_per
                for res in results:  # wait for all images to generate
                    mutation_folder.merge_folder(res.get())

    @staticmethod
    def execute_tests(mutation_folder: MutationFolder, mutation_type: MutationType, arg_dict,
                      num_tests=600):
        """
        Create the test cases of the specified type for the specified folder.
        :param mutation_folder: The MutationFolder to use
        :param mutation_type: The MutationType to instantiate
        :param arg_dict: The base arguments for any Mutations
        :param num_tests: The number of tests to create. Default 600
        :return:
        """
        if not Tester._initialized:
            Tester.initialize()
        start_time = time.time()
        Tester.create_fuzz_images(mutation_folder, num_tests, mutation_type=mutation_type, arg_dict=arg_dict)
        mutation_folder.record_mutations()
        end_time = time.time()
        total_time = end_time - start_time
        time_per = total_time / num_tests
        print('Generated %d mutations in %0.2f s (%0.2f s/im, ~%0.2f cpus/im)' %
              (num_tests, total_time, time_per, time_per * Tester.pool_count))
        # TODO add discriminator here or move it into the create_fuzz_images call

    @staticmethod
    def run_fuzzer(folders_to_run=20, generate_only=False, num_per_mutation=2000, mutations_to_run=None):
        """
        Runs the Fuzzer for the specified amount. Total number of mutations created is:
         num_per_mutation * len(mutations_to_run) * folders_to_run
        :param folders_to_run: The number of MutationFolders to generate. Default 20
        :param generate_only: If True, do not run the SUTs, only create the mutations. Default False
        :param num_per_mutation: The number of each mutation to run for MutationFolder.  Default 2000
        :param mutations_to_run: List of tuples of (MutationType, dict of parameters) pairs to run. Default:
            [
                (MutationType.CHANGE_COLOR, {'semantic_label': 'car'}),
                (MutationType.ADD_OBJECT, {'semantic_label': 'person'}),
                (MutationType.ADD_OBJECT, {'semantic_label': 'car'})
            ]
        :return: None
        """
        if mutations_to_run is None:
            mutations_to_run = [
                (MutationType.CHANGE_COLOR, {'semantic_label': 'car'}),
                (MutationType.ADD_OBJECT, {'semantic_label': 'person'}),
                (MutationType.ADD_OBJECT, {'semantic_label': 'car'})
            ]
        if not Tester._initialized:
            Tester.initialize()
        for folders_run in range(folders_to_run):
            mutation_folder = Tester.get_next_mutation_folder(num_per_mutation * len(mutations_to_run))
            print('Running fuzzing for %s' % mutation_folder.base_folder)
            num_already_run = len(mutation_folder.mutation_map)
            for mutation_type, arg_dict in mutations_to_run:
                if num_already_run >= num_per_mutation:
                    num_already_run = max(0, num_already_run - num_per_mutation)
                    continue
                Tester.execute_tests(mutation_folder, mutation_type, arg_dict, num_per_mutation - num_already_run)
            if not generate_only:
                Tester.sut_manager.run_suts(mutation_folder)
                Tester.compute_cityscapes_metrics(mutation_folder)

    @staticmethod
    def compute_cityscapes_metrics(mutation_folder: MutationFolder,
                                   exclude_high_dnc=False, quiet=True, force_recalc=False):
        """Compute the Cityscapes metric of SUT performance on the given MutationFolder.

        If metrics have already been calculated, do not recalculate unless force_recalc is True.

        If exclude_high_dnc is True, images on which the SUT predicted more than 200000 pixels (~14%) as DNC
          are removed from consideration. Default False.
        """
        results = {}
        if not exclude_high_dnc and not force_recalc:
            all_paths_exist = True
            for sut in Tester.sut_list:
                if not os.path.exists(mutation_folder.get_sut_raw_results(sut.name)):
                    all_paths_exist = False
            if all_paths_exist:
                # define these so that the eval below has them in scope
                nan = np.nan
                float32 = np.float32
                for sut in Tester.sut_list:
                    with open(mutation_folder.get_sut_raw_results(sut.name)) as f:
                        results[sut.name] = eval(f.read())
                return results
        cityscapes_eval_args = copy.copy(orig_cityscapes_eval_args)
        cityscapes_eval_args.evalPixelAccuracy = True
        cityscapes_eval_args.quiet = quiet
        black_pixel = [0, 0, 0]
        with Pool(Tester.pool_count) as pool:
            for sut in Tester.sut_list:
                print('--- Evaluating %s ---' % sut.name)
                pred_img_list = []
                gt_img_list = []
                skip_count = 0
                folder = mutation_folder.get_sut_folder(sut.name)
                for file in glob.glob(folder + '*edit_prediction.png'):
                    file_name = file[file.rfind('/') + 1:]
                    short_file = file_name[file_name.rfind('/') + 1:]
                    base_img = short_file[:-20]  # remove the _edit_prediction.png part
                    mutation_gt_file = mutation_folder.mutations_gt_folder +\
                                       base_img + '_mutation_gt.png'
                    if exclude_high_dnc:
                        if base_img in Tester.HIGH_DNC or (len(base_img) > 52 and base_img[37:] in Tester.HIGH_DNC):
                            skip_count += 1
                            continue
                        else:
                            gt_im = cv2.imread(mutation_gt_file)
                            dnc = np.count_nonzero(np.all(gt_im == black_pixel,axis=2))
                            if dnc > 200000:
                                Tester.HIGH_DNC.append(mutation_gt_file)
                                continue
                    pred_img_list.append(file)
                    gt_img_list.append(mutation_gt_file)
                print('Skipped %d, kept %d' % (skip_count, len(pred_img_list)))
                if Tester.pool_count == 1:
                    results[sut.name] = evaluateImgLists(pred_img_list, gt_img_list, cityscapes_eval_args)
                else:
                    results[sut.name] = pool.apply_async(evaluateImgLists,
                                                        (pred_img_list, gt_img_list, cityscapes_eval_args))
            if Tester.pool_count > 1:
                for sut in Tester.sut_list:
                    results[sut.name] = results[sut.name].get()
        with open(mutation_folder.human_results, 'w') as human_results:
            out = 'Exclude high DNC: ' + str(exclude_high_dnc)
            print(out)
            human_results.write(out + '\n')
            for sut in Tester.sut_list:
                out = '%s: %0.4f' % (sut.name, 100*results[sut.name]['averageScoreClasses'])
                print(out)
                human_results.write(out + '\n')
            human_results.write('All Results:\n')
            for sut in Tester.sut_list:
                human_results.write('SUT: %s\n' % sut.name)
                raw_str = str(results[sut.name])
                human_results.write(raw_str + '\n\n')
                with open(mutation_folder.get_sut_raw_results(sut.name), 'w') as raw_results:
                    raw_results.write(raw_str + '\n')
        return results

    @staticmethod
    def __get_cityscapes_runs_folder():
        """Location to save the SUT performance on the Cityscapes benchmark"""
        if Tester.__cityscapes_runs_folder is None:
            Tester.__cityscapes_runs_folder = MutationFolder(Tester.CITYSCAPES_DATA_ROOT + '/sut_gt_testing')
        return Tester.__cityscapes_runs_folder

    @staticmethod
    def run_on_cityscapes_benchmark(force_recalc=False):
        """This ensures that the SUTs have been run on the Cityscapes benchmark

        If there is no prior save data or force_recalc is True, the SUTs are run on the benchmark.

        Returns the output of compute_cityscapes_metrics run on the folder containing the benchmark results."""
        mutation_folder = Tester.__get_cityscapes_runs_folder()
        for camera_image in glob.glob(Tester.CITYSCAPES_DATA_ROOT +
                                      "/gtFine_trainvaltest/gtFine/leftImg8bit/**/*_leftImg8bit.png", recursive=True):
            if 'leftImg8bit/test/' in camera_image:
                continue
            short_file = camera_image[camera_image.rfind('/') + 1:-len('_leftImg8bit.png')]
            new_file = mutation_folder.folder + short_file + '_edit.png'
            if not os.path.exists(new_file):
                copyfile(camera_image, new_file)
        results = []
        with Pool(Tester.pool_count) as pool:
            for gt_image in glob.glob(Tester.CITYSCAPES_DATA_ROOT +
                                      "/gtFine_trainvaltest/gtFine/**/*_gtFine_color.png", recursive=True):
                if 'gtFine/test/' in gt_image:
                    continue
                short_file = gt_image[gt_image.rfind('/') + 1:-len('_gtFine_color.png')]
                new_file = mutation_folder.mutations_gt_folder + short_file + '_mutation_gt.png'
                if not os.path.exists(new_file) or force_recalc:
                    results.append(pool.apply_async(save_paletted_image, (gt_image, new_file)))
            for res in results:
                res.wait()
        Tester.sut_manager.run_suts(mutation_folder, force_recalc)
        return Tester.compute_cityscapes_metrics(mutation_folder)

    @staticmethod
    def visualize_diffs(sut_diffs):
        """Helper method to plot a histogram of the count of inconsistencies found by the SUTs"""
        bin_count = max([len(sut_diffs[sut.name].values()) for sut in Tester.sut_list]) // 10
        other_bins = None
        for sut in Tester.sut_list:
            diffs = sut_diffs[sut.name]
            if other_bins is None:
                _, other_bins, _ = plt.hist(diffs.values(), bins=bin_count, alpha=0.5, label=sut.name,
                                         histtype='step')
            else:
                plt.hist(diffs.values(), bins=other_bins, alpha=0.5, label=sut.name,
                         histtype='step')
        plt.xlabel("SUT Differences")
        plt.ylabel("Count")
        plt.title("SUT Differences")
        plt.legend(loc='upper right')
        plt.show()

    @staticmethod
    def compute_differences(truth, predicted, ignore_black=False):
        """This method is no longer used but was explored prior to adopting the Cityscapes difference metric."""
        # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
        truth_gray = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
        predicted_gray = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(truth_gray, predicted_gray, full=True)
        diff = (diff * 255).astype("uint8")
        diff = np.stack((diff,) * 3, axis=-1)
        diff_image = cv2.bitwise_xor(truth, predicted)
        diff_image[np.where((diff_image!=[0,0,0]).any(axis=2))] = [255, 255, 255]  # convert not black to white
        if ignore_black:
            diff_image[np.where((truth == [0, 0, 0]).any(axis=2))] = [0, 0, 0]  # convert black in truth to black in diff since we are ignoring
        num_pixels = np.count_nonzero(np.where((diff_image!=[0,0,0]).any(axis=2)))
        diff_image_pred = cv2.add(predicted, diff_image)
        return score, num_pixels, diff_image, diff_image_pred

    @staticmethod
    def draw_text(image, text_list, start_x=10, start_y=20):
        """Helper method to draw text on an image. Useful when investigating SUT/mutation performance"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.5
        fontColor = (255, 0, 255)
        lineType = 3
        for index, text in enumerate(text_list):
            cv2.putText(image,
                        text,
                        (start_x, start_y + 60 * (index + 1)),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    @staticmethod
    def visualize_folder(mutation_folder):
        """Helper method to visualize what parts of images were misclassified by an SUT from mutation"""
        orig_file_suffix = '_orig.png'
        orig_predicted_file_suffix = '_orig_prediction.png'
        edit_file_suffix = '_edit.png'
        edit_predicted_file_suffix = '_edit_prediction.png'
        differences = []
        for key, mutation in mutation_folder.mutation_map.items():
            base_file_name = key
            base_file = mutation_folder.folder + base_file_name
            mutation.update_orig_prediction(Image(image_file=base_file + orig_predicted_file_suffix, read_on_load=True))
            mutation.edit_prediction = Image(image_file=base_file + edit_predicted_file_suffix, read_on_load=True)
            top = cv2.hconcat([mutation.orig_image.image, mutation.edit_image.image])
            mid1, _ = Tester.diff_image_pair(base_file_name, mutation.orig_image.image, mutation.edit_image.image)
            mid = cv2.hconcat([mutation.orig_prediction_mutated.image, mutation.edit_prediction.image])
            bottom, diff_percent = Tester.diff_image_pair(base_file_name, mutation.orig_prediction_mutated.image, mutation.edit_prediction.image)
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid1, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(base_file + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    @staticmethod
    def visualize_plain_folder(folder):
        """Helper method to visualize what parts of images were misclassified by an SUT"""
        orig_file_suffix = '_orig.png'
        orig_predicted_file_suffix = '_orig_prediction.png'
        edit_file_suffix = '_edit.png'
        edit_predicted_file_suffix = '_edit_prediction.png'
        differences = []
        _, _, filenames = next(os.walk(folder))
        for file in filenames:
            base_file_name : str = file[:file.rfind('.png')]
            if not base_file_name.endswith('_orig_prediction'):
                continue
            base_file_name: str = file[:file.rfind('_orig_prediction.png')]
            edit_prediction_loc = folder + '/' + base_file_name + '_prediction.png'
            orig_prediction_loc = folder + '/' + base_file_name + '_orig_prediction.png'
            orig_image_loc = folder + '/' + base_file_name + '.png'
            print(edit_prediction_loc)
            print(orig_prediction_loc)
            print(orig_image_loc)
            edit_prediction = cv2.imread(edit_prediction_loc)
            orig_prediction = cv2.imread(orig_prediction_loc)
            orig_image = cv2.imread(orig_image_loc)
            top = cv2.hconcat([orig_image, orig_image])
            mid = cv2.hconcat([orig_prediction, edit_prediction])
            bottom, diff_percent = Tester.diff_image_pair(base_file_name, orig_prediction, edit_prediction, ignore_black=True)
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(folder + '/' + base_file_name + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    @staticmethod
    def diff_image_pair(base_file_name, orig_im, edit_im, ignore_black=False):
        """This method is unused. Prior to the adoption of the Cityscapes metric, it was explored as a diff metric."""
        score, num_pixels, blank_diff, diff = Tester.compute_differences(orig_im, edit_im, ignore_black)
        diff_percent = 100 * float(num_pixels) / (orig_im.shape[0] * orig_im.shape[1])
        # txt_image = np.zeros((orig_im.shape[0], orig_im.shape[1], 3), np.uint8)
        Tester.draw_text(blank_diff, [
            base_file_name,
            '# pixels diff: %d (%f%%)' % (num_pixels, diff_percent),
            'Diff Score (-1 to 1): %f' % score
        ])
        mid1 = cv2.hconcat([blank_diff, diff])
        return mid1, diff_percent

def print_distances(polys: List[SemanticPolygon]):
    i = 0
    while i < len(polys) - 1:
        j = i + 1
        while j < len(polys):
            min_dist = polys[i].min_distance(polys[j])
            print(min_dist)
            j += 1
        i += 1


def plot_hist_as_line(data, label, bin_count=None, bins=None, log=False):
    # https://stackoverflow.com/questions/27872723/is-there-a-clean-way-to-generate-a-line-histogram-chart-in-python
    n, calced_bins, _ = plt.hist(data, bins=bins if bins is not None else bin_count,
                                 histtype='bar', alpha=1, stacked=True, label=label, log=log)
    return n, calced_bins


if __name__ == '__main__':
    # TODO update these to point to the SUTs on your local machine
    best_sut_to_test = NVIDIASemSeg('/home/adwiii/git/nvidia/semantic-segmentation')
    suts_to_test = [
        best_sut_to_test,
        NVIDIASDCNet('/home/adwiii/git/nvidia/sdcnet/semantic-segmentation',
                     '/home/adwiii/git/nvidia/large_assets/sdcnet_weights/cityscapes_best.pth'),
        DecoupleSegNet('/home/adwiii/git/DecoupleSegNets'),
        EfficientPS('/home/adwiii/git/EfficientPS'),
        HRNet('/home/adwiii/git/HRNet-Semantic-Segmentation')
    ]
    Tester.initialize(best_sut_to_test, suts_to_test)
    Tester.run_fuzzer()