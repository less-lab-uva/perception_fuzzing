from skimage.metrics import structural_similarity

import traceback
from multiprocessing import Pool
from image_mutator import *

sys.path.append('/home/adwiii/git/pytorch-DAVE2/src/torchdave/')
from dave_model import *

class Tester:
    def __init__(self):
        pass

    def compute_differences(self, truth, predicted, ignore_black=False):
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

    def draw_text(self, image, text_list, start_x=10, start_y=20):
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

    def visualize_folder(self, mutation_folder):
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
            mid1, _ = self.diff_image_pair(base_file_name, mutation.orig_image.image, mutation.edit_image.image)
            mid = cv2.hconcat([mutation.orig_prediction_mutated.image, mutation.edit_prediction.image])
            bottom, diff_percent = self.diff_image_pair(base_file_name, mutation.orig_prediction_mutated.image, mutation.edit_prediction.image)
            # total = cv2.vconcat([top, mid1, mid, bottom])
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid1, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(base_file + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    def visualize_plain_folder(self, folder):
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
            bottom, diff_percent = self.diff_image_pair(base_file_name, orig_prediction, edit_prediction, ignore_black=True)
        #     # total = cv2.vconcat([top, mid1, mid, bottom])
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(folder + '/' + base_file_name + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    def diff_image_pair(self, base_file_name, orig_im, edit_im, ignore_black=False):
        score, num_pixels, blank_diff, diff = self.compute_differences(orig_im, edit_im, ignore_black)
        diff_percent = 100 * float(num_pixels) / (orig_im.shape[0] * orig_im.shape[1])
        # txt_image = np.zeros((orig_im.shape[0], orig_im.shape[1], 3), np.uint8)
        self.draw_text(blank_diff, [
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
CITYSCAPES_DATA_ROOT = '/home/adwiii/data/cityscapes'
cityscapes_mutator = CityscapesMutator(CITYSCAPES_DATA_ROOT)

def create_images(folder, count, start_num):
    i = start_num
    mutations = []
    while i < start_num + count:
        try:
            # mutation = nuim_mutator.change_car_color()
            # mutation = cityscapes_mutator.change_instance_color('car')
            # mutation = cityscapes_mutator.change_instance_color('person')
            mutation = cityscapes_mutator.add_instance('car', rotation=random.randint(1, 3)*90)
            # mutation = nuim_mutator.random_obj_to_random_image()
            if mutation is not None:
                mutations.append(mutation)
                mutation.update_file_names(folder.folder, folder.pred_mutations_folder)
                mutation.save_images()
            else:
                i -= 1  # don't advance if we didn't get a new mutation
        except Exception as e:
            traceback.print_exc(e)
            pass
        i += 1
    return 0


def create_fuzz_images(mutation_folder, count, pool_count=16):
    count_per = int(count / pool_count)
    results = []
    orig_count = count
    if pool_count == 1:
        create_images(mutation_folder, count, 0)
    else:
        # TODO below is currently broken due to pickling issues
        with Pool(pool_count) as pool:
            while count > 0:
                res = pool.apply_async(create_images, (mutation_folder, min(count, count_per), orig_count - count))
                results.append(res)
                count -= count_per
            for res in results:  # wait for all images to generate
                res.get()


def print_distances(polys: List[SemanticPolygon]):
    i = 0
    while i < len(polys) - 1:
        j = i + 1
        while j < len(polys):
            min_dist = polys[i].min_distance(polys[j])
            print(min_dist)
            j += 1
        i += 1


KEY_CLASSES = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'person', 'rider']
if __name__ == '__main__':
    steering_models = SteeringModel()
    steering_models.load_state_dict()
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

