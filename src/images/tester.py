import itertools
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
from src.sut_runner.decouple_segnet import DecoupleSegNet
from src.sut_runner.efficientps import EfficientPS
from src.sut_runner.hrnet import HRNet
from src.sut_runner.nvidia_sdcnet import NVIDIASDCNet
from src.sut_runner.nvidia_semantic_segmentation import NVIDIASemSeg
from src.sut_runner.sut_manager import SUTManager

current_file_path = Path(__file__)
sys.path.append(str(current_file_path.parent.parent.absolute()) + '/cityscapesScripts')
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import args as orig_cityscapes_eval_args


def save_paletted_image(old_file_path, new_file_path):
    Image(image=cv2.imread(old_file_path), image_file=new_file_path).save_paletted_image()


class Tester:
    CITYSCAPES_DATA_ROOT = '/home/adwiii/data/cityscapes'
    cityscapes_mutator = None
    BEST_SUT = None  # the SUT to be used for criteria of whether or not to include in mutation resources
    HIGH_DNC = []
    sut_list = None
    sut_manager = None
    pool_count = None
    _initialized = False
    SCORE_THRESHOLD = 0.8

    @staticmethod
    def initialize(cityscapes_data_root=None, pool_count=30, score_threshold=None):
        Tester._initialized = True
        Tester.BEST_SUT = NVIDIASemSeg('/home/adwiii/git/nvidia/semantic-segmentation')
        Tester.sut_list = [
            Tester.BEST_SUT,
            NVIDIASDCNet('/home/adwiii/git/nvidia/sdcnet/semantic-segmentation',
                         '/home/adwiii/git/nvidia/large_assets/sdcnet_weights/cityscapes_best.pth'),
            DecoupleSegNet('/home/adwiii/git/DecoupleSegNets'),
            EfficientPS('/home/adwiii/git/EfficientPS'),
            HRNet('/home/adwiii/git/HRNet-Semantic-Segmentation')
        ]
        Tester.sut_manager = SUTManager(Tester.sut_list)
        Tester.pool_count = pool_count
        if cityscapes_data_root is not None:
            Tester.CITYSCAPES_DATA_ROOT = cityscapes_data_root

        cityscapes_runs_folder = Tester.__get_cityscapes_runs_folder()
        if os.path.exists(cityscapes_runs_folder.raw_results):
            with open(cityscapes_runs_folder.raw_results) as f:
                cityscapes_results = eval(f.read())
        else:
            cityscapes_results = Tester.run_on_cityscapes_benchmark()  # this will run and return, saving for next time
        if score_threshold is not None:
            Tester.SCORE_THRESHOLD = score_threshold
        good_files = [Tester.get_base_file(image[0])
                      for image in cityscapes_results[Tester.BEST_SUT.name]["perImageScores"]
                      if Tester.get_score(image) > Tester.SCORE_THRESHOLD]
        print(good_files)
        print('found %d good files' % len(good_files))
        Tester.cityscapes_mutator = CityscapesMutator(Tester.CITYSCAPES_DATA_ROOT, good_files)

    @staticmethod
    def get_score(image):
        return 100.0 * (1.0 - image[1]['nbCorrectPixels'] / image[1]['nbNotIgnoredPixels'])

    @staticmethod
    def get_base_file(long_file: str):
        long_file = long_file[long_file.rfind('/') + 1:]
        if '-' in long_file:
            # strip uuid
            long_file = long_file[37:]
        long_file = long_file.replace('_edit_prediction.png', '')
        return long_file

    @staticmethod
    def __create_images(folder: MutationFolder, count, start_num, mutation_type: MutationType, arg_dict):
        """Generate the specified number of mutations"""
        i = start_num
        mutations = []
        while i < start_num + count:
            try:
                mutation = Tester.cityscapes_mutator.apply_mutation(mutation_type, arg_dict)
                if mutation is not None:
                    mutations.append(mutation)
                    mutation.update_file_names(folder)
                    mutation.save_images()
                else:
                    i -= 1  # don't advance if we didn't get a new mutation
            except Exception as e:
                traceback.print_exc(e)
                pass
            i += 1
        return 0

    @staticmethod
    def __create_fuzz_images(mutation_folder: MutationFolder, count, mutation_type: MutationType, arg_dict):
        """Generate mutated images using a thread pool for increased speed"""
        count_per = int(count / Tester.pool_count)
        results = []
        orig_count = count
        if Tester.pool_count == 1:
            Tester.__create_images(mutation_folder, count, 0, mutation_type, arg_dict)
        else:
            with Pool(Tester.pool_count) as pool:
                while count > 0:
                    res = pool.apply_async(Tester.__create_images, (mutation_folder, min(count, count_per),
                                                                   orig_count - count, mutation_type, arg_dict))
                    results.append(res)
                    count -= count_per
                for res in results:  # wait for all images to generate
                    res.get()

    @staticmethod
    def execute_tests(mutation_folder: MutationFolder, mutation_type: MutationType, arg_dict,
                      num_tests=600, pool_count=30, compute_metrics=True):
        if not Tester._initialized:
            Tester.initialize()
        start_time = time.time()
        Tester.__create_fuzz_images(mutation_folder, num_tests, mutation_type=mutation_type, arg_dict=arg_dict)
        mutation_folder.record_mutations()
        end_time = time.time()
        total_time = end_time - start_time
        time_per = total_time / num_tests
        print('Generated %d mutations in %0.2f s (%0.2f s/im, ~%0.2f cpus/im)' % (num_tests, total_time,
                                                                                  time_per, time_per * pool_count))
        # TODO add discriminator here or move it into the create_fuzz_images call
        Tester.sut_manager.run_suts(mutation_folder)
        if compute_metrics:
            Tester.compute_cityscapes_metrics(mutation_folder)

    @staticmethod
    def compute_cityscapes_metrics(mutation_folder: MutationFolder,
                                   exclude_high_dnc=False, quiet=False, pool_count=28):
        cityscapes_eval_args = copy.copy(orig_cityscapes_eval_args)
        cityscapes_eval_args.evalPixelAccuracy = True
        cityscapes_eval_args.quiet = quiet
        results = {}
        black_pixel = [0, 0, 0]
        # dnc_count = []
        with Pool(pool_count) as pool:
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
                        if (base_img in Tester.HIGH_DNC or (len(base_img) > 52 and base_img[37:] in Tester.HIGH_DNC)):
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
                results[sut.name] = pool.apply_async(evaluateImgLists,
                                                     (pred_img_list, gt_img_list, cityscapes_eval_args))
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
                human_results.write(str(results[sut.name]) + '\n\n')
        with open(mutation_folder.raw_results, 'w') as raw_results:
            raw_results.write(str(results) + '\n')
        return results

    @staticmethod
    def __get_cityscapes_runs_folder():
        if Tester.__cityscapes_runs_folder is None:
            Tester.__cityscapes_runs_folder = MutationFolder(Tester.CITYSCAPES_DATA_ROOT + '/sut_gt_testing')
        return Tester.__cityscapes_runs_folder

    @staticmethod
    def run_on_cityscapes_benchmark():
        mutation_folder = Tester.__get_cityscapes_runs_folder()
        for camera_image in glob.glob(Tester.CITYSCAPES_DATA_ROOT +
                                      "/gtFine_trainvaltest/gtFine/leftImg8bit/**/*_leftImg8bit.png", recursive=True):
            short_file = camera_image[camera_image.rfind('/') + 1:-len('_leftImg8bit.png')]
            copyfile(camera_image, mutation_folder.folder + short_file + '_edit.png')
        results = []
        with Pool(Tester.pool_count) as pool:
            for gt_image in glob.glob(Tester.CITYSCAPES_DATA_ROOT +
                                      "/gtFine_trainvaltest/gtFine/**/*_gtFine_color.png", recursive=True):
                short_file = gt_image[gt_image.rfind('/') + 1:-len('_gtFine_color.png')]
                new_file = mutation_folder.mutations_gt_folder + short_file + '_mutation_gt.png'
                results.append(pool.apply_async(save_paletted_image, (gt_image, new_file)))
            for res in results:
                res.wait()
        Tester.sut_manager.run_suts(mutation_folder)
        return Tester.compute_cityscapes_metrics(mutation_folder)

    @staticmethod
    def visualize_diffs(sut_diffs):
        bin_count = max([len(sut_diffs[sut.name].values()) for sut in self.sut_list]) // 10
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
        # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
        truth_gray = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
        predicted_gray = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(truth_gray, predicted_gray, full=True)
        diff = (diff * 255).astype("uint8")
        diff = np.stack((diff,) * 3, axis=-1)
        # diff_image = np.copy(predicted)
        diff_image = cv2.bitwise_xor(truth, predicted)
        diff_image[np.where((diff_image!=[0,0,0]).any(axis=2))] = [255, 255, 255]  # convert not black to white
        if ignore_black:
            diff_image[np.where((truth == [0, 0, 0]).any(axis=2))] = [0, 0, 0]  # convert black in truth to black in diff since we are ignoring
        num_pixels = np.count_nonzero(np.where((diff_image!=[0,0,0]).any(axis=2)))
        diff_image_pred = cv2.add(predicted, diff_image)
        return score, num_pixels, diff_image, diff_image_pred

    @staticmethod
    def draw_text(image, text_list, start_x=10, start_y=20):
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
            # total = cv2.vconcat([top, mid1, mid, bottom])
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid1, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(base_file + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    @staticmethod
    def visualize_plain_folder(folder):
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
            # mid1, _ = self.diff_image_pair(base_file_name, orig_image, orig_image)
            mid = cv2.hconcat([orig_prediction, edit_prediction])
            bottom, diff_percent = Tester.diff_image_pair(base_file_name, orig_prediction, edit_prediction, ignore_black=True)
        #     # total = cv2.vconcat([top, mid1, mid, bottom])
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(folder + '/' + base_file_name + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    @staticmethod
    def diff_image_pair(base_file_name, orig_im, edit_im, ignore_black=False):
        score, num_pixels, blank_diff, diff = self.compute_differences(orig_im, edit_im, ignore_black)
        diff_percent = 100 * float(num_pixels) / (orig_im.shape[0] * orig_im.shape[1])
        # txt_image = np.zeros((orig_im.shape[0], orig_im.shape[1], 3), np.uint8)
        Tester.draw_text(blank_diff, [
            base_file_name,
            '# pixels diff: %d (%f%%)' % (num_pixels, diff_percent),
            'Diff Score (-1 to 1): %f' % score
        ])
        mid1 = cv2.hconcat([blank_diff, diff])
        return mid1, diff_percent

# TODO move the below functions into the class above
DATA_ROOT = '/home/adwiii/data/sets/nuimages'
nuim = NuImages(dataroot=DATA_ROOT, version='v1.0-mini', verbose=False, lazy=True)
nuim_mutator = NuScenesMutator(DATA_ROOT, 'v1.0-mini')


def print_distances(polys: List[SemanticPolygon]):
    i = 0
    while i < len(polys) - 1:
        j = i + 1
        while j < len(polys):
            min_dist = polys[i].min_distance(polys[j])
            print(min_dist)
            j += 1
        i += 1


def plot_hist_as_line(data, label, bin_count=None, bins=None):
    # https://stackoverflow.com/questions/27872723/is-there-a-clean-way-to-generate-a-line-histogram-chart-in-python
    n, calced_bins, _ = plt.hist(data, bins=bins if bins is not None else bin_count, histtype='bar', alpha=1, label=label, stacked=True)
    # bin_centers = 0.5 * (calced_bins[1:] + calced_bins[:-1])
    # plt.plot(bin_centers, n, label=label)  ## using bin_centers rather than edges
    return n, calced_bins


KEY_CLASSES = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'person', 'rider']
if __name__ == '__main__':
    tester = Tester()
    # mutation_folder = MutationFolder('/home/adwiii/git/perception_fuzzing/src/images/new_mutation_gt')
    # tester.compute_cityscapes_metrics(mutation_folder)
    base_folder = '/home/adwiii/git/perception_fuzzing/src/images/fri_'
    folders = []
    for local_folder, mutation_type, arg_dict in [
        ('person_color', MutationType.CHANGE_COLOR, {'semantic_label': 'person'}),
        ('car_color', MutationType.CHANGE_COLOR, {'semantic_label': 'car'}),
        ('add_person', MutationType.ADD_OBJECT, {'semantic_label': 'person'}),
        ('add_car', MutationType.ADD_OBJECT, {'semantic_label': 'car'}),
    ]:
        folder = base_folder + local_folder
        mutation_folder = MutationFolder(folder)
        # tester.execute_tests(mutation_folder, mutation_type=mutation_type, arg_dict=arg_dict, compute_metrics=False)
        folders.append(mutation_folder)
    folders.insert(0, MutationFolder(CITYSCAPES_DATA_ROOT + '/sut_gt_testing'))
    worst_images = {}
    worst_images_drop = {}
    worst_count = 5
    worst_images_for_counts = []
    score_on_training = {}
    for index, mutation_folder in enumerate(folders):
        bins = np.linspace(0.5, 4.5, 20)
        running_total = 0
        results = tester.compute_cityscapes_metrics(mutation_folder, quiet=True)
        worst_images[mutation_folder.short_name] = {}
        worst_images_drop[mutation_folder.short_name] = {}
        for sut, result_dict in results.items():
            image_scores = result_dict["perImageScores"]
            data = [Tester.get_score(image) for image in image_scores.items()]
            if index == 0:
                score_on_training[sut] = {Tester.get_base_file(image[0]):
                                              Tester.get_score(image)
                                          for image in image_scores.items()}
            data_drop = [score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image)
                         for image in image_scores.items() if (score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image)) > 0.5]
            data_with_images_drop = sorted([(image[0], Tester.get_score(image),
                                        score_on_training[sut][Tester.get_base_file(image[0])],
                                             score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image))
                                       for image in image_scores.items()],  # sort by drop from gt to us
                                      key=lambda x: (x[3], -x[1]), reverse=True)
            data_with_images = sorted(
                [(image[0], Tester.get_score(image),
                  score_on_training[sut][Tester.get_base_file(image[0])],
                  score_on_training[sut][Tester.get_base_file(image[0])] - Tester.get_score(image))
                 for image in image_scores.items()],  # sort by worst performance
                key=lambda x: x[1])
            worst_images[mutation_folder.short_name][sut] = data_with_images[:worst_count]
            worst_images_drop[mutation_folder.short_name][sut] = data_with_images_drop[:worst_count]
            worst_images_for_counts.extend([Tester.get_base_file(item[0]) for item in data_with_images[:worst_count]])
            n, bins = plot_hist_as_line(data_drop, sut, 40, bins)
            running_total += np.sum(n)
        plt.title('Hist of percentage point drop in %% pixel correct\nfor each image vs non-mutated img: %s\nTotal: %d' % (mutation_folder.short_name.replace('fri', ''), running_total))
        plt.xlabel('Percentage point drop in % pixels correct')
        plt.ylabel('Count of Images')
        plt.legend(loc='upper right')
        plt.show()
    print(worst_images)
    print(worst_images_drop)
    print(Counter(worst_images_for_counts))
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    print('Worst:')
    pp.pprint(worst_images)
    print('Overall Worst:')
    pp.pprint(Counter(worst_images_for_counts))
    pp.pprint({mutation: Counter(list(itertools.chain(*[[Tester.get_base_file(item[0]) for item in lst] for lst in suts.values()]))) for mutation, suts in worst_images.items()})
    print()
    print('Worst Drop:')
    pp.pprint(worst_images_drop)
    pp.pprint(
        {mutation: Counter(list(itertools.chain(*[[Tester.get_base_file(item[0]) for item in lst] for lst in suts.values()])))
         for mutation, suts in worst_images_drop.items()})

    exit()

    # tester.run_on_cityscapes_benchmark()

    # folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_mutations_10k/'
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_mutations_add_car/'
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_add_person/'
    folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_change_person_color/'
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/add_car_check_perspective_rotated/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    diff_folder = folder[:-1] + '_differences/'
    if not os.path.exists(diff_folder):
        os.mkdir(diff_folder)
    # cityscapes_mutator.aggregate_images(folder)
    # semantic_seg_runner.run_semantic_seg(folder)

    mutation_folder = MutationFolder(folder)
    # create_fuzz_images(mutation_folder, 100, pool_count=1)
    #
    # exit()
    # semantic_seg_runner.run_semantic_seg(folder)
    # exit()

    issue_count = defaultdict(lambda:0)
    issue_count_files = defaultdict(lambda:[])
    steering_diffs = []
    # for file in glob.glob('/home/adwiii/git/perception_fuzzing/src/images/cityscapes_gt_predictions/*_prediction.png'):
    for file in glob.glob(folder + '*edit_prediction.png'):
        print(file)
        try:
            file_name = file[file.rfind('/')+1:]
            # if file_name not in ['1255b027-7596-4fcf-912f-8b6734780f86_erfurt_000100_000019_edit_prediction.png', 'ff1d4db2-0cb8-4996-ac32-b525bf9eeed8_erfurt_000100_000019_edit_prediction.png', '7669be9a-2120-46a1-8aea-0a469932ab2b_bremen_000312_000019_edit_prediction.png', '55203bc1-545a-482c-b466-50256afcc304_erfurt_000100_000019_edit_prediction.png', '4c5d6fbf-5978-4ffd-a8eb-db8f3eb79adf_cologne_000059_000019_edit_prediction.png', 'ee683248-339e-4035-bf10-5ca0d984a84b_dusseldorf_000167_000019_edit_prediction.png', '60ddc2ca-549e-4538-8aad-5d0025c7e0e3_dusseldorf_000165_000019_edit_prediction.png', '6cddbf73-8c56-48fe-af9b-1a29eb8700ba_cologne_000059_000019_edit_prediction.png', '6dff1e2b-2022-409e-8827-23c635c8446a_dusseldorf_000167_000019_edit_prediction.png', 'f9887a77-1c34-478b-9e98-d1f4e2f37b7a_erfurt_000100_000019_edit_prediction.png', '4ef90064-cc16-4ab6-8952-cd540f818d12_cologne_000059_000019_edit_prediction.png']:
            #     continue
            # print(file_name)
            # if file_name not in ['7d5e1d74-2613-4122-8cc7-dc9683728300_krefeld_000000_030111_edit_prediction.png', '80bad570-6d37-40f4-aad8-1f6876f1c4c0_aachen_000129_000019_edit_prediction.png'] and \
            #     file_name not in ['b7d1dd45-6a1c-4573-8ef4-33e4f75433d1_dusseldorf_000167_000019_edit_prediction.png', 'c3093d20-2414-4d0f-aba3-54ca8f578b35_dusseldorf_000167_000019_edit_prediction.png', '08d13095-bb02-4141-a031-4ecf87015721_ulm_000049_000019_edit_prediction.png', 'c31bc55e-a1e9-42f6-bec5-bf5b8ca36415_hanover_000000_056457_edit_prediction.png', '82a8c4eb-c47f-4246-9b6f-0a5e8b9d1370_ulm_000008_000019_edit_prediction.png', '623874c8-067d-417a-a9e0-ce263c6be19e_hanover_000000_007897_edit_prediction.png']:
            #     continue
            # if file_name not in ['623874c8-067d-417a-a9e0-ce263c6be19e_hanover_000000_007897_edit_prediction.png']:
            #     continue
            # if file_name not in ['29848911-9773-4a31-a29e-5f739bfa990b_darmstadt_000060_000019_edit_prediction.png']:
            #     continue
            orig_pred_image_file = file.replace('edit_prediction', 'orig_prediction')
            orig_pred_image = Image(image_file=orig_pred_image_file, read_on_load=True)
            short_file = file_name[file_name.rfind('/')+1:]
            orig_pred_mutation_file = mutation_folder.pred_mutations_folder + short_file.replace('edit_prediction', 'mutation_prediction')
            if os.path.exists(orig_pred_mutation_file):
                orig_pred_mutation_image = Image(image_file=orig_pred_mutation_file, read_on_load=True)
                pred_mutation_mask = np.copy(orig_pred_mutation_image.image)
                # convert to black and white
                pred_mutation_mask[np.where((pred_mutation_mask != [0,0,0]).any(axis=2))] = [255, 255, 255]
                pred_mutation_mask = cv2.cvtColor(pred_mutation_mask, cv2.COLOR_BGR2GRAY)
                _, pred_mutation_mask = cv2.threshold(pred_mutation_mask, 40, 255, cv2.THRESH_BINARY)
                inv_mask = cv2.bitwise_not(pred_mutation_mask)
                orig_pred_image.image = cv2.bitwise_and(orig_pred_image.image, orig_pred_image.image, mask=inv_mask)
                # orig_pred_mutated = cv2.bitwise_or(orig_pred_image.image, orig_pred_mutation_image.image, mask=pred_mutation_mask)
                orig_pred_image.image = cv2.add(orig_pred_image.image, orig_pred_mutation_image.image)
            edit_pred_image = Image(image_file=file, read_on_load=True)
            steering_angles = steering_models.evaluate([orig_pred_image.image, edit_pred_image.image])
            print(steering_angles)
            steering_diff = abs(steering_angles[0] - steering_angles[1]) * 180 / np.pi
            # if 10 > steering_diff > 5:
            #     cv2.imshow('orig', orig_pred_image.image)
            #     cv2.imshow('edit', edit_pred_image.image)
            #     cv2.waitKey()
            # exit()
            steering_diffs.append((steering_diff, steering_angles[0] * 180 / np.pi, steering_angles[1] * 180 / np.pi, file))
            # orig_pred_semantics = ImageSemantics(orig_pred_image, CityscapesMutator.COLOR_TO_ID, KEY_CLASSES)
            # edit_pred_semantics = ImageSemantics(edit_pred_image, CityscapesMutator.COLOR_TO_ID, KEY_CLASSES)
            # orig_pred_semantics.compute_semantic_polygons()
            # edit_pred_semantics.compute_semantic_polygons()
            # cityscapes_short_file = file_name[file_name.find('_')+1:file_name.find('_edit')]
            # exclusion_zones: List[CityscapesPolygon] = cityscapes_mutator.get_ego_vehicle(cityscapes_mutator.get_file(cityscapes_short_file))
            # sem_diffs = SemanticDifferences(orig_pred_semantics, edit_pred_semantics, [poly.polygon for poly in exclusion_zones])
            # sem_diffs.calculate_semantic_difference()
            # issues = 0
            # for key in KEY_CLASSES:
            #     total = len(sem_diffs.only_in_orig[key]) + len(sem_diffs.only_in_edit[key])
            #     if total > 0:
            #         edit_pred_image_copy = Image(image=np.copy(edit_pred_image.image), image_file=diff_folder + file_name)
            #         orig_pred_image_copy = Image(image=np.copy(orig_pred_image.image), image_file=diff_folder + file_name.replace('edit', 'orig'))
            #         print('Found differences for %s in image %s' % (key, file_name))
            #         print('Only in gt:')
            #         for orig in sem_diffs.only_in_orig[key]:
            #             print(orig.center, orig.effective_area, len(orig.additions), orig.max_dim)
            #             orig_pred_image_copy.image = cv2.drawContours(orig_pred_image_copy.image, orig.get_inflated_polygon_list(), -1, (255, 255, 255))
            #             issues += 1
            #         print_distances(sem_diffs.only_in_orig[key])
            #         print('Only in predicted:')
            #         for edit in sem_diffs.only_in_edit[key]:
            #             print(edit.center, edit.effective_area, len(edit.additions), edit.max_dim)
            #             edit_pred_image_copy.image = cv2.drawContours(edit_pred_image_copy.image, edit.get_inflated_polygon_list(), -1, (255, 255, 255))
            #             issues += 1
            #         orig_pred_image_copy.save_image()
            #         edit_pred_image_copy.save_image()
            # # cv2.imshow('orig', orig_pred_semantics.image)
            # # cv2.imshow('edit', edit_pred_semantics.image)
            # # cv2.waitKey()
            # issue_count[issues] += 1
            # issue_count_files[issues].append(file_name)
        except:
            traceback.print_exc()
            exit()
            issue_count[-1] += 1
    for count, files in issue_count_files.items():
        if count == 0:
            continue
        print(count, files)
    print(issue_count)
    steering_diffs.sort(reverse=True)
    print('\n'.join([str(s) for s in steering_diffs]))
    plt.hist([item[0] for item in steering_diffs])
    plt.xlabel('Difference in steering angles (deg)')
    plt.ylabel('Count')
    title = folder[:-1]
    title = title[title.rfind('/'):]
    plt.title(title)
    plt.show()



# def check_differences_polygons_cityscapes():
#     cityscapes_color_to_id = {(0, 0, 0): 'static', (0, 74, 111): 'dynamic', (81, 0, 81): 'ground', (128, 64, 128): 'road', (232, 35, 244): 'sidewalk', (160, 170, 250): 'parking', (140, 150, 230): 'rail track', (70, 70, 70): 'building', (156, 102, 102): 'wall', (153, 153, 190): 'fence', (180, 165, 180): 'guard rail', (100, 100, 150): 'bridge', (90, 120, 150): 'tunnel', (153, 153, 153): 'polegroup', (30, 170, 250): 'traffic light', (0, 220, 220): 'traffic sign', (35, 142, 107): 'vegetation', (152, 251, 152): 'terrain', (180, 130, 70): 'sky', (60, 20, 220): 'person', (0, 0, 255): 'rider', (70, 0, 0): 'truck', (100, 60, 0): 'bus', (90, 0, 0): 'caravan', (110, 0, 0): 'trailer', (100, 80, 0): 'train', (230, 0, 0): 'motorcycle', (32, 11, 119): 'bicycle'}
#     for i in range(1, 2001):
#         try:
#             which = str(i)  #  TODO 19, resume at 61
#             orig_image = Image(image_file='/home/adwiii/git/perception_fuzzing/src/images/test_imgs/%s_orig_prediction.png' % which)
#             orig_image.load_image()
#             orig_semantics = ImageSemantics(orig_image, cityscapes_color_to_id)
#             orig_semantics.compute_semantic_polygons()
#             edit_image = Image(image_file='/home/adwiii/git/perception_fuzzing/src/images/test_imgs/%s_edit_prediction.png' % which)
#             edit_image.load_image()
#             edit_semantics = ImageSemantics(edit_image, cityscapes_color_to_id)
#             edit_semantics.compute_semantic_polygons()
#
#             sem_diffs = SemanticDifferences(orig_semantics, edit_semantics)
#             sem_diffs.calculate_semantic_difference()
#             # for key in sem_diffs.all_keys:
#             for key in ['car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']:
#                 total = len(sem_diffs.only_in_orig[key]) + len(sem_diffs.only_in_edit[key])
#                 if total > 0:
#                     print('Found differences for %s in image %s' % (key, which))
#                     print('Only in orig:')
#                     for orig in sem_diffs.only_in_orig[key]:
#                         print(orig.center, orig.effective_area)
#                     print('Only in edit:')
#                     for edit in sem_diffs.only_in_edit[key]:
#                         print(edit.center, edit.effective_area)
#                 # for pair in sem_diffs.matching_pairs[key]:
#                 #     print('Center Distance:', pair.get_center_distance())
#                 #     print('Area Difference:', pair.get_area_difference())
#         except:
#             pass


    # for sem_id in orig_semantics.semantic_maps.keys():
    #     print(sem_id, len(orig_semantics.semantic_maps[sem_id]), len(edit_semantics.semantic_maps[sem_id]))

    # folder = '/home/adwiii/git/SIMS/result/synthesis/transform_order_512'
    # semantic_seg_runner.run_semantic_seg(folder)
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/test_imgs4/'
    # if not os.path.exists(folder):
    #     os.mkdir(folder)
    # mutation_folder = MutationFolder(folder)
    # create_fuzz_images(mutation_folder, 200, pool_count=1)
    # print('Generated images, running semantic seg')
    # semantic_seg_runner.run_semantic_seg(mutation_folder.folder)
    # print('Run Semantic Seg, running analytics')
    # tester = Tester()
    # tester.visualize_plain_folder(folder)

