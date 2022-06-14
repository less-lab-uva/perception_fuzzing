import glob
import os
import time
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

import PIL.ImagePalette
import cv2
import matplotlib.pyplot as plt
import sys
import scipy
import numpy as np
import random
import uuid
import tempfile  # used for ipc with deepfill
import json
from pathlib import Path
from PIL import Image as PILImage

from src.images.vanishing_point.vanishing_point_detection import get_vanishing_point

current_file_path = Path(__file__)
sys.path.append(str(current_file_path.parent.parent.absolute()) + '/cityscapesScripts')
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels     import name2label, trainId2label, id2label


class Scene:
    def __init__(self):
        pass


class Image:
    _flattening_array = np.array([1, 1 << 8, 1 << 16], dtype=np.uint32)
    _default_palette_rgb = [trainId2label[id].color if id in trainId2label else (0, 0, 0) for id in range(256)]
    _default_palette_bgr_flattened = {}
    for index, color in enumerate(_default_palette_rgb):
        dot_color = np.dot(np.array([color[2], color[1], color[0]]), _flattening_array)
        if dot_color not in _default_palette_bgr_flattened:
            _default_palette_bgr_flattened[dot_color] = index
    for label in id2label.values():
        dot_color = np.dot(np.array([label.color[2], label.color[1], label.color[0]]), _flattening_array)
        if dot_color not in _default_palette_bgr_flattened:
            _default_palette_bgr_flattened[dot_color] = 19
    _default_palette = [val for rgbs in _default_palette_rgb for val in rgbs]

    def __init__(self, image=None, image_file: str = None, read_on_load=False):
        self.image = image
        self.image_file = image_file
        if read_on_load and image_file is not None:
            self.load_image()

    def save_image(self):
        cv2.imwrite(self.image_file, self.image)

    def save_paletted_image(self):
        flattened = self.image.reshape((self.image.shape[0] * self.image.shape[1], 3))
        flattened = np.matmul(flattened, Image._flattening_array)
        paletted_image = np.array([Image._default_palette_bgr_flattened[flat_color] for flat_color in flattened],
                                  dtype=np.uint8)
        paletted_image = paletted_image.reshape((self.image.shape[0], self.image.shape[1]))
        paletted = PILImage.fromarray(paletted_image.astype(np.uint8)).convert('P')
        paletted.putpalette(Image._default_palette)
        paletted.save(self.image_file)

    def load_image(self):
        self.image = cv2.imread(self.image_file)

    def free_memory(self):
        self.image = None


class MutationFolder:
    def __init__(self, base_folder, perform_init=True):
        if base_folder[-1] != '/':
            base_folder += '/'
        self.short_name = base_folder[:-1]  # remove trailing slash
        self.short_name = self.short_name[self.short_name.rfind('/')+1:]  # remove leading path
        self.base_folder = base_folder
        self.folder = base_folder + 'mutations/'
        self.human_results = base_folder + 'results.txt'
        self.mutation_logs = base_folder + 'mutation_logs.txt'
        self.mutations_gt_folder = base_folder + 'mutations_gt/'
        self.mutation_map = {}
        if perform_init:
            os.makedirs(self.folder, exist_ok=True)
            os.makedirs(self.mutations_gt_folder, exist_ok=True)
            if os.path.exists(self.mutation_logs):
                self.read_mutations()

    def get_sut_folder(self, sut_name: str):
        return '%s%s/' % (self.base_folder, sut_name)

    def get_sut_raw_results(self, sut_name: str):
        return '%s%s_raw_results.txt' % (self.base_folder, sut_name)

    def add_mutation(self, mutation):
        self.mutation_map[mutation.name] = mutation
        mutation.update_file_names(self)

    def record_mutations(self):
        with open(self.mutation_logs, 'w') as f:
            for name, mutation in self.mutation_map.items():
                f.write("('%s', %s),\n" % (name, str(mutation.params)))

    def read_mutations(self):
        with open(self.mutation_logs, 'r') as f:
            self.mutations_from_file(f)

    def mutations_from_file(self, file):
        arr = eval('[' + file.read() + ']')
        for name, mutation_params in arr:
            self.mutation_map[name] = Mutation.from_params(name, mutation_params, self)

    @classmethod
    def mutation_folder_from_file(cls, base_folder, file):
        mut_fol = cls(base_folder, False)
        mut_fol.mutations_from_file(file)
        os.makedirs(mut_fol.folder, exist_ok=True)
        os.makedirs(mut_fol.mutations_gt_folder, exist_ok=True)
        return mut_fol

    def add_all(self, mutations):
        for mutation in mutations:
            self.add_mutation(mutation)

    def save_all_images(self):
        for mutation in self.mutation_map.values():
            mutation.save_images()

    def merge_folder(self, mutation_map):
        self.mutation_map.update(mutation_map)


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    # shoelace formula taken from
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class SemanticPolygon:
    def __init__(self, semantic_id, points, inflation: Optional[Tuple[float, float]] = None,
                 exclusions: Optional[List[np.ndarray]] = None, additions: Optional[List[np.ndarray]] = None):
        self.points = points
        self.semantic_id = semantic_id
        self.uuid = uuid.uuid4()
        self.exclusions = [] if exclusions is None else exclusions
        self.additions = [] if additions is None else additions
        self.exclusion_area = sum([polygon_area(points) for points in self.exclusions])
        self.addition_area = sum([polygon_area(points) for points in self.additions])
        self.total_area = polygon_area(self.points)
        self.effective_points = np.copy(points)
        if len(self.additions) > 0:
            addition_points = []
            for points in self.additions:
                addition_points.extend(points)
            self.effective_points = np.append(self.effective_points, addition_points, axis=0)
        self.polygon_list = [self.points]
        self.polygon_list.extend(self.additions)
        self.inflation = inflation
        self.inflated_points = None
        if self.inflation is not None:
            self.inflated_points = self.get_inflated_polygon_list()
        self.center = self.__calculate_center()
        self.effective_area = self.__calculate_effective_area()
        self.bounding_box = None
        self.width = 0
        self.height = 0
        self.max_dim = 0
        self.__calculate_bounding_box()

    def __calculate_effective_area(self):
        return self.total_area - self.exclusion_area + self.addition_area

    def __calculate_center(self):
        # TODO, consider exclusions?
        return np.sum(self.effective_points, 0) / self.effective_points.shape[0]

    def __calculate_bounding_box(self):
        mins = np.amin(self.effective_points, axis=0)
        maxs = np.amax(self.effective_points, axis=0)
        self.width = maxs[0] - mins[0]
        self.height = maxs[1] - mins[1]
        self.max_dim = max(self.width, self.height)
        self.bounding_box = np.array([[mins[0], mins[1]], [maxs[0], mins[1]], [maxs[0], maxs[1]], [mins[0], maxs[1]]])

    def add_exclusion(self, exclusion: np.ndarray):
        self.exclusions.append(exclusion)
        self.exclusion_area += polygon_area(exclusion)
        self.effective_area = self.__calculate_effective_area()

    def add_addition(self, addition: "SemanticPolygon"):
        self.additions.append(addition.points)
        self.polygon_list.extend(addition.polygon_list)
        self.total_area += addition.total_area
        self.addition_area += addition.addition_area
        self.effective_points = np.append(self.effective_points, addition.effective_points, axis=0)
        for exclusion in addition.exclusions:
            self.add_exclusion(exclusion)
        self.effective_area = self.__calculate_effective_area()
        self.center = self.__calculate_center()
        self.inflated_points = None  # clear the cache so that next time it is regenerated
        self.__calculate_bounding_box()

    def center_distance(self, other: "SemanticPolygon"):
        return np.hypot(*(self.center - other.center))

    def min_distance(self, other: "SemanticPolygon"):
        return np.amin(scipy.spatial.distance.cdist(self.effective_points, other.effective_points))

    def get_inflated_polygon_list(self, inflation: Optional[Tuple[float, float]] = None):
        if self.inflated_points is not None and (inflation is None or self.inflation == inflation):
            return self.inflated_points
        if inflation is None:
            inflation = self.inflation
        return [points * inflation for points in self.polygon_list]

    def __lt__(self, other):
        if self.center[0] < other.center[0]:
            return True
        if self.center[0] == other.center[0]:
            return self.center[1] < other.center[1]
        else:
            return False


def is_parent(index, hierarchy):
    count = 0
    while hierarchy[index][-1] != -1:
        count += 1
        index = hierarchy[index][-1]
    return (count % 2) == 0  # if we had to go up an even number of times then this is a prent


class ImageSemantics:
    MIN_DIST_TO_MERGE_FRAC = 0.5  # merge if within half of either object's size
    MIN_DIST_TO_MERGE = 15
    def __init__(self, image: Image, color_to_ids=None, key_classes=None):
        self.reduction = (4, 4)
        self.image = cv2.resize(image.image, (image.image.shape[1] // self.reduction[1], image.image.shape[0] // self.reduction[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        self.polygons = []
        if color_to_ids is None:
            self.color_to_ids = {}
            self.calculate_semantic_ids()
        else:
            self.color_to_ids = color_to_ids
        self.key_classes = key_classes if key_classes is not None else self.color_to_ids.keys()
        self.semantic_maps: Dict[Any, List[SemanticPolygon]] = defaultdict(lambda: [])
        self.calculate_semantic_ids()

    def calculate_semantic_ids(self):
        if len(self.color_to_ids) > 0:
            return
        temp = np.reshape(self.image, (-1, 3))  # I cannot figure out how to get unique to operate on the original image, so reshape into 1d
        semantic_ids = np.unique(temp, axis=0)  # for the image this is the distinct colors, those can be used as keys
        # Lists cannot be used as keys, so we need to convert them to tuples
        for index, (b, g, r) in enumerate(semantic_ids):
            self.color_to_ids[(b, g, r)] = index

    def compute_semantic_polygons(self):
        for color, semantic_id in self.color_to_ids.items():
            if semantic_id not in self.key_classes:
                continue
            color_mask = np.copy(self.image)
            # convert to black and white
            color_mask[np.where((color_mask == color).all(axis=2))] = [255, 255, 255]
            color_mask[np.where((color_mask != [255, 255, 255]).any(axis=2))] = [0, 0, 0]
            # convert to gray scale for cv2 find contours
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
            unblurred = color_mask
            color_mask = cv2.blur(color_mask, (7, 7))
            pre_thresh = color_mask
            _, color_mask = cv2.threshold(color_mask, 40, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE simplifies series of points into lines, but this makes calculating the center, etc harder
            if len(contours) > 0:
                hierarchy = hierarchy[0]  # for some reason it is wrapped as a 1-item array?
                hierarchy_id_to_sem_poly = {}
                current_polys: List[SemanticPolygon] = []
                # create SemanticPolygons for all parent polygons in the contour
                for index, (polygon, (next_poly, prev_poly, first_child, parent)) in enumerate(zip(contours, hierarchy)):
                    if not is_parent(index, hierarchy):
                        continue  # this polygon has a parent, so we need to keep track of it with that
                    polygon = np.reshape(polygon, (-1, 2))
                    semantic_polygon = SemanticPolygon(semantic_id, polygon, inflation=self.reduction)
                    current_polys.append(semantic_polygon)
                    hierarchy_id_to_sem_poly[index] = semantic_polygon
                # fill in the child polygons as exclusion zones in the parent SemanticPolygon
                for index, (polygon, (next_poly, prev_poly, first_child, parent)) in enumerate(zip(contours, hierarchy)):
                    if is_parent(index, hierarchy):
                        continue  # we only want to fill in ones with parents
                    polygon = np.reshape(polygon, (-1, 2))
                    parent_poly = hierarchy_id_to_sem_poly[parent]
                    parent_poly.add_exclusion(polygon)
                # merge any polygons that are too close to each other
                current_polys.sort()
                i = 0
                while i < len(current_polys):
                    j = 0
                    while i < len(current_polys) and j < len(current_polys):
                        if j == i:
                            j += 1
                            continue
                        min_dist = current_polys[i].min_distance(current_polys[j])
                        if min_dist <= max(ImageSemantics.MIN_DIST_TO_MERGE,
                                           ImageSemantics.MIN_DIST_TO_MERGE_FRAC *
                                           min(current_polys[i].max_dim, current_polys[j].max_dim)):
                            current_polys[i].add_addition(current_polys.pop(j))
                            j = 0  # we merged with one, so this one's size has changed. Start over
                        else:
                            # do not increment if we deleted one
                            j += 1
                    i += 1
                for poly in current_polys:
                    self.polygons.append(poly)
                    self.semantic_maps[semantic_id].append(poly)


class MatchedPolygonPair:
    def __init__(self, orig: SemanticPolygon, edit: SemanticPolygon):
        self.orig = orig
        self.edit = edit

    def get_area_difference(self):
        return self.orig.effective_area - self.edit.effective_area

    def get_center_distance(self):
        return self.orig.center_distance(self.edit)

    def get_polygon_of_difference(self):
        return None


def polygon_contains(container_list, points):
    # TODO check actual containment instead of bounding box containment
    bool = True
    for container in container_list:
        mins = np.amin(container, axis=0)
        maxs = np.amax(container, axis=0)
        bool &= np.all(np.logical_and(mins[0] <= points[:, 0], points[:,0] <= maxs[0])) and np.all(np.logical_and(mins[1] <= points[:, 1], points[:,1] <= maxs[1]))
    return bool


def polygon_intersects(container_list, points):
    # TODO check actual containment instead of bounding box containment
    bool = True
    for container in container_list:
        mins = np.amin(container, axis=0)
        maxs = np.amax(container, axis=0)
        bool |= np.any(np.logical_and(mins[0] <= points[:, 0], points[:,0] <= maxs[0])) and np.any(np.logical_and(mins[1] <= points[:, 1], points[:,1] <= maxs[1]))
    return bool


class SemanticDifferences:
    # MIN_DIST_TO_MERGE = 5
    MAX_DIST_FOR_SIM = 50  # completely arbitrary, but if the center of the object we are looking at is more than 50 then we should say they aren't related
    # completely arbitrary, but if an area is small (<this px^2) then we will just ignore it.
    # There are lots of places where 1px is labelled in the ground truth and that is not reasonable to test on imo
    MAX_NEGLIGIBLE_AREA = 50  # make both 10 for //4
    MAX_PERCENT_AREA_ERROR = 0.25  # if the two shapes differ in area by more than this we will call them diff

    def __init__(self, orig_sem: ImageSemantics, edit_sem: ImageSemantics, exclusion_zones: Optional[List[List[List[float]]]] = None):
        self.orig_sem = orig_sem
        self.edit_sem = edit_sem
        self.all_keys = list(self.orig_sem.semantic_maps.keys())
        self.all_keys.extend(self.edit_sem.semantic_maps.keys())
        self.all_keys = list(set(self.all_keys))  # remove duplicates
        self.matching_pairs: Dict[Any, List[MatchedPolygonPair]] = defaultdict(lambda: [])
        self.only_in_orig: Dict[Any, List[SemanticPolygon]] = defaultdict(lambda: [])
        self.only_in_edit: Dict[Any, List[SemanticPolygon]] = defaultdict(lambda: [])
        self.exclusion_zones = []
        if exclusion_zones is not None:
            for poly in exclusion_zones:
                condensed = np.array([[point[0] / self.orig_sem.reduction[0], point[1] / self.orig_sem.reduction[1]] for point in poly])
                self.exclusion_zones.append(condensed)

    def calculate_semantic_difference(self):
        for sem_id in self.all_keys:
            # orig_count = len(self.orig_sem.semantic_maps[sem_id])
            # edit_count = len(self.edit_sem.semantic_maps[sem_id])
            # if orig_count != edit_count:
            #     print('Semantic counts differ for class %s. Orig: %d, Edit: %d' % (sem_id, orig_count, edit_count))
            orig_list = list(self.orig_sem.semantic_maps[sem_id])
            edit_list = list(self.edit_sem.semantic_maps[sem_id])
            pair_wise_distances: Dict[SemanticPolygon, List[Tuple[float, SemanticPolygon]]] = defaultdict(lambda: [])
            for orig in orig_list:
                for edit in edit_list:
                    pair_wise_distances[orig].append((orig.center_distance(edit), edit))
                pair_wise_distances[orig].sort()
            looking = len(orig_list) > 0
            orig_cannot_match = []
            while looking:
                found_match = False
                current = orig_list[0]
                diffs_list = pair_wise_distances[current]
                found = None
                cur_best_idx = 0
                best_dist = np.inf
                while found not in edit_list and cur_best_idx < len(diffs_list):
                    best_dist, found = diffs_list[cur_best_idx]
                    cur_best_idx += 1
                # print(current.center)
                # print('Dist:', best_dist)
                if best_dist < SemanticDifferences.MAX_DIST_FOR_SIM:
                    if np.array_equal(current.points, found.points):
                        found_match = True
                    else:
                        # If we want to add additional logic based on, e.g., size, do so here.
                        # found_match = True
                        # if (current.effective_area < SemanticDifferences.MAX_NEGLIGIBLE_AREA or found.effective_area < SemanticDifferences.MAX_NEGLIGIBLE_AREA) and abs(current.effective_area - found.effective_area) < SemanticDifferences.MAX_NEGLIGIBLE_AREA:
                        #     found_match = True
                        if found.effective_area == 0 or current.effective_area == 0:
                            found_match = found.effective_area == 0 and current.effective_area == 0
                        else:
                            if min(current.effective_area / found.effective_area, found.effective_area / current.effective_area) > (1 - SemanticDifferences.MAX_PERCENT_AREA_ERROR):
                                found_match = True
                else:
                    found_match = False  # redundant but keeping it here to be clear

                if found_match:
                    self.matching_pairs[sem_id].append(MatchedPolygonPair(current, found))
                    # remove them from their respective lists
                    del orig_list[0]
                    edit_list.remove(found)
                else:
                    # remove the orig from its list and add to the unmatchable list
                    orig_cannot_match.append(orig_list.pop(0))
                if len(orig_list) == 0 or len(edit_list) == 0:
                    looking = False

            # Filter out the ones that are too small
            orig_cannot_match = [sem for sem in orig_cannot_match if sem.effective_area > SemanticDifferences.MAX_NEGLIGIBLE_AREA and not polygon_contains(self.exclusion_zones, sem.effective_points)]
            edit_list = [sem for sem in edit_list if sem.effective_area > SemanticDifferences.MAX_NEGLIGIBLE_AREA and not polygon_contains(self.exclusion_zones, sem.effective_points)]
            self.only_in_orig[sem_id].extend(orig_cannot_match)
            self.only_in_edit[sem_id].extend(edit_list)


class MutationType(Enum):
    CHANGE_COLOR = 'CHANGE_COLOR'
    ADD_OBJECT = 'ADD_OBJECT'

    def __str__(self):
        return 'MutationType.' + self.name

    def __repr__(self):
        return 'MutationType.' + self.name


class Mutation:
    def __init__(self, mutation_type: MutationType, orig_image: Image, edit_image: Image, mutation_gt: Image, name=None,
                 params: dict = None, unique_name=True):
        self.name = str(uuid.uuid4()) if unique_name else ''
        if name is not None:
            if unique_name:
                self.name += '_'  # add spacer if we already added a uuid above
            self.name += name
        self.orig_image = orig_image
        self.edit_image = edit_image
        self.mutation_gt = mutation_gt
        self.params = params
        self.params['mutation_type'] = mutation_type

    @classmethod
    def from_params(cls, name, mutation_params: dict, mutation_folder: MutationFolder):
        mutation = cls(mutation_type=mutation_params['mutation_type'], orig_image=Image(),
                       edit_image=Image(), mutation_gt=Image(), name=name, params=mutation_params, unique_name=False)
        mutation.update_file_names(mutation_folder)
        return mutation

    def update_file_names(self, mutation_folder: MutationFolder):
        for image, postfix in [(self.orig_image, '_orig.png'),
                               (self.edit_image, '_edit.png')]:
            if image is not None:
                image.image_file = mutation_folder.folder + self.name + postfix
        if mutation_folder.mutations_gt_folder is not None and self.mutation_gt is not None:
            self.mutation_gt.image_file = mutation_folder.mutations_gt_folder + self.name + '_mutation_gt.png'

    def save_images(self, save_orig=False, free_mem=False):
        if save_orig:
            self.orig_image.save_image()
        self.edit_image.save_image()
        if self.mutation_gt is not None:
            self.mutation_gt.save_paletted_image()
        if free_mem:
            self.orig_image.free_memory()
            self.edit_image.free_memory()
            self.mutation_gt.free_memory()


class CityscapesGroundTruth:
    def __init__(self, real_world, json_labels, labeled_image):
        self.real_world = real_world
        self.json_labels = json_labels
        self.labeled_image = labeled_image


class CityscapesPolygon:
    def __init__(self, json_file: str, polygon, label, poly_id, order):
        self.json_file = json_file
        self.polygon = polygon
        self.label = label
        self.poly_id = poly_id
        self.order = order

    def get_gt_semantics_image(self):
        semantic_file = self.json_file.replace('_polygons.json', '_color.png')
        return cv2.imread(semantic_file)


class CityscapesMutator:
    IMAGE_WIDTH = 2048
    IMAGE_HEIGHT = 1024
    COLOR_TO_ID = {(label.color[2], label.color[1], label.color[0]): label.name
                   for label in name2label.values() if label.id != -1 and label.trainId != 255}

    def __init__(self, data_root, good_files=None, blurred=False):
        self.data_root = data_root
        self.mutate = ImageMutator()
        self.label_mapping: Dict[str, List[CityscapesPolygon]] = defaultdict(lambda: [])
        self.file_mapping: Dict[str, List[CityscapesPolygon]] = defaultdict(lambda: [])
        self.poly_id_mapping: Dict[str, CityscapesPolygon] = {}
        self.short_file_mapping: Dict[str, str] = {}
        self.good_files = good_files
        self.blurred = blurred
        self.build_image_dict()

    def get_mutation_type(self, name):
        return MutationType[name]

    def apply_mutation(self, mutation_type: MutationType, arg_dict):
        if mutation_type == MutationType.CHANGE_COLOR:
            return self.change_instance_color(**arg_dict)
        elif mutation_type == MutationType.ADD_OBJECT:
            return self.add_instance(**arg_dict)
        else:
            raise NotImplementedError('No existing implementation for ' + str(mutation_type))

    def get_file(self, short_file):
        return self.short_file_mapping[short_file]

    def build_image_dict(self):
        for file in glob.glob(self.data_root + "/gtFine_trainvaltest/gtFine/**/*.json", recursive=True):
            short_file = file[file.rfind('/')+1:-len('_gtFine_polygons.json')]
            if self.good_files is not None and short_file not in self.good_files:
                continue
            self.short_file_mapping[short_file] = file
            with open(file) as f:
                json_file = json.load(f)
                if 'objects' in json_file:
                    object_list = json_file['objects']
                    for index, semantic_label in enumerate(object_list):
                        if 'deleted' in semantic_label:
                            continue
                        label = semantic_label['label']
                        if (label not in name2label) and label.endswith('group'):
                            label = label[:-len('group')]
                        if label not in name2label or name2label[label].id == -1:
                            continue  # either we can't find it or it is marked to be not added
                        polygon = np.array(semantic_label['polygon'])
                        poly_id = file[file.rfind('/')+1:-21] + '_' + str(index)
                        city_poly = CityscapesPolygon(file, polygon, label, poly_id, index)
                        self.label_mapping[label].append(city_poly)
                        self.file_mapping[file].append(city_poly)
                        self.poly_id_mapping[poly_id] = city_poly

    def get_ego_vehicle(self, file):
        return self.get_polys_for_image(file, ['ego vehicle'])

    def get_polys_for_image(self, file, types=None):
        if types is None:
            return self.file_mapping[file]
        else:
            return [poly for poly in self.file_mapping[file] if poly.label in types]

    def aggregate_images(self, dest_folder):
        for file in self.file_mapping.keys():
            img = self.load_image(file)
            name = file[file.rfind('/') + 1:-21]
            cv2.imwrite(dest_folder + '/' + name + '.png', img)

    def get_random_instance(self, label='car') -> CityscapesPolygon:
        obj_list = self.label_mapping[label]
        return random.choice(obj_list)

    def load_image(self, json_file: str):
        file = json_file.replace('gtFine/', 'gtFine/leftImg8bit/').replace('gtFine_polygons.json', 'leftImg8bit.png')
        if self.blurred:
            # replace the path so that we load blurred images only
            file = file.replace('gtFine_trainvaltest/gtFine/leftImg8bit', 'leftImg8bit_blurred/leftImg8bit_blurred/')\
                       .replace('leftImg8bit.png', 'leftImg8bit_blurred.jpg')
        img = cv2.imread(file)
        return img

    def add_instance(self, semantic_label='car', bg_id=None, add_id=None, rotation=None):
        if bg_id is None:
            road_mask = None
            while road_mask is None:
                road_poly = self.get_random_instance('road')
                road_mask = self.get_instance_mask(road_poly)
            bg_id = road_poly.poly_id
        else:
            road_poly = self.poly_id_mapping[bg_id]
            road_mask = self.get_instance_mask(road_poly)
            if road_mask is None:
                return None
        road_img = self.load_image(road_poly.json_file)
        road_vanishing_point = get_vanishing_point(road_img)
        if add_id is not None:
            car_poly = self.poly_id_mapping[add_id]
            instance_mask = self.get_instance_mask(car_poly)
            thresh = cv2.threshold(instance_mask, 1, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            car_polygon = np.reshape(contours[0], (-1, 2))
            mins = np.amin(car_polygon, axis=0)
            maxs = np.amax(car_polygon, axis=0)
            random_car_image = self.load_image(car_poly.json_file)
        else:
            # find a car that we can put on this road
            count = 10
            found = False
            while count > 0 and not found:
                car_poly = self.get_random_instance(semantic_label)
                add_id = car_poly.poly_id
                instance_mask, orig_car_poly, other_polys = self.get_instance_mask(car_poly, return_other_polys=True)
                # if the car is too small or not contiguous then skip it
                thresh = cv2.threshold(instance_mask, 1, 255, cv2.THRESH_BINARY)[1]
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if instance_mask is None or len(contours) != 1 or\
                        (semantic_label == 'car' and
                         not self.mutate.is_larger_than(instance_mask, (100, 100), both=False)) or\
                        (semantic_label == 'person' and
                         not self.mutate.is_larger_than(instance_mask, (50, 50), both=False)):
                    continue
                count -= 1
                overlap = cv2.bitwise_and(other_polys, other_polys, mask=orig_car_poly)
                if np.count_nonzero(overlap) > 0 and semantic_label == 'car':
                    # as a precaution, don't use cars that are occluded
                    continue
                car_polygon = np.reshape(contours[0], (-1, 2))
                random_car_image = self.load_image(car_poly.json_file)
                mins = np.amin(car_polygon, axis=0)
                maxs = np.amax(car_polygon, axis=0)
                # if the mask is white here then we are on the road
                if road_mask[max(0, min(maxs[1], road_mask.shape[0])), max(0, min(mins[0], road_mask.shape[1]))] == 255:
                    # The below commented out if statement checks
                    # to see if the lower left of the car bounding box is in the road, then we will check other cars
                    # NOTE: this was not used in the study because it was determined to be overly aggressive at
                    # preventing mutations that would otherwise be acceptable. This is because pixel overlap is not
                    # sufficient for determining physical overlap.
                    # if not polygon_intersects([poly.polygon for poly in self.get_polys_for_image(road_poly, ['car'])], car_poly.polygon):

                    # check that the underlying area has a similar brightness to the car that we are going to insert
                    if not self.mutate.is_lighting_similar(road_img, random_car_image, instance_mask):
                        # then check that the vanishing point of the car image and the road image are the same
                        # if they are the same then it is more likely that the images can be combined
                        car_vanishing_point = get_vanishing_point(random_car_image)
                        if car_vanishing_point == road_vanishing_point:
                            found = True
            if not found:
                return None
        lower_left = (mins[0], maxs[1])
        param_map = {
            'bg_id': bg_id,
            'add_id': add_id,
            'semantic_label': semantic_label
        }
        name = road_poly.json_file[road_poly.json_file.rfind('/') + 1:-21]
        img = self.mutate.add_object(random_car_image, instance_mask, road_img, lower_left, add_shadow=False, rotation=rotation)
        mutation_gt = road_poly.get_gt_semantics_image()
        rgb = name2label[semantic_label].color
        mutation_gt = cv2.fillPoly(mutation_gt, pts=[car_polygon], color=(rgb[2], rgb[1], rgb[0]))
        mutation = Mutation(MutationType.ADD_OBJECT, Image(road_img), Image(img), name=name,
                            mutation_gt=Image(mutation_gt), params=param_map)
        return mutation

    def change_instance_color(self, semantic_label='car', poly_id=None, color_shift=None, dec_lit=None, inc_sat=None):
        city_poly = self.get_random_instance(semantic_label) if poly_id is None else self.poly_id_mapping[poly_id]
        poly_id = city_poly.poly_id
        random_car_file = city_poly.json_file
        name = random_car_file[random_car_file.rfind('/')+1:-21]
        random_car_image = self.load_image(random_car_file)
        random_car_mask = self.get_instance_mask(city_poly)
        x, y, w, h = self.mutate.mask_bounding_rect(random_car_mask)
        if w < 50 or h < 50:
            return None
        random_car_isolated = self.mutate.get_isolated(random_car_image, random_car_mask)
        random_car_colored, color_shift, dec_lit, inc_sat = self.mutate.change_car_color(random_car_isolated,
                                                                                         color_shift, dec_lit, inc_sat)
        img = self.mutate.add_isolated_object(random_car_colored, random_car_image, (x, y+h), random_car_mask)
        param_map = {
            'poly_id': poly_id,
            'color_shift': color_shift,
            'dec_lit': dec_lit,
            'inc_sat': inc_sat,
            'semantic_label': semantic_label
        }
        mutation = Mutation(MutationType.CHANGE_COLOR, Image(random_car_image), Image(img), name=name,
                            mutation_gt=Image(city_poly.get_gt_semantics_image()), params=param_map)
        return mutation

    def get_instance_mask(self, city_poly: CityscapesPolygon, filter_in_front=True, color=None, return_other_polys=False):
        polygon = city_poly.polygon
        color = color if color is not None else 255
        blank = np.zeros((CityscapesMutator.IMAGE_HEIGHT, CityscapesMutator.IMAGE_WIDTH, 1), np.uint8)
        orig_poly = cv2.fillPoly(blank, pts=[polygon], color=color)
        mask = np.copy(orig_poly)
        if filter_in_front:
            other_polys = np.zeros((CityscapesMutator.IMAGE_HEIGHT, CityscapesMutator.IMAGE_WIDTH, 1), np.uint8)
            other_list = self.file_mapping[city_poly.json_file]
            for other_poly in other_list:
                if other_poly.order <= city_poly.order:  # skip ourselves and anyone behind us
                    continue
                other_polys = cv2.fillPoly(other_polys, pts=[other_poly.polygon], color=color)
            # if we are occluded by anything in front of us, take that out of our poly
            mask[np.where((other_polys == 255).all(axis=2))] = 0
            if np.count_nonzero(mask) == 0:
                if return_other_polys:
                    return None, None, None
                return None  # the object we are looking for is entirely occluded, no use in continuing
        if return_other_polys:
            return mask, orig_poly, other_polys
        return mask


class ImageMutator:

    def __init__(self):
        pass

    def __translate_affine(self, tx, ty):
        return np.float32([[1, 0, tx], [0, 1, ty]])

    def __normalize_mask(self, mask):
        if len(mask.shape) == 3 and mask.shape[-1] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # convert mask to grayscale
        return mask

    def is_contiguous(self, mask):
        mask = self.__normalize_mask(mask)
        thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) == 1

    def is_larger_than(self, mask, dim, both=True):
        mask = self.__normalize_mask(mask)
        x, y, w, h = self.mask_bounding_rect(mask)
        if both:
            return w > dim[0] and h > dim[1]
        else:
            return w > dim[0] or h > dim[1]

    def get_isolated(self, img, mask):
        mask = self.__normalize_mask(mask)
        return cv2.bitwise_and(img, img, mask=mask)  # isolate just the part we want to add

    def change_car_color(self, isolated, color_shift=None, dec_lit=None, inc_sat=None):
        # TODO isolate the parts of the car that can change color (i.e. not the windows, etc.)
        return self.change_color(isolated, color_shift, dec_lit, inc_sat)

    def change_color(self, isolated, color_shift=None, dec_lit=None, inc_sat=None):
        # Current problems:
        # If the car is already very low V (black) then changing the hue doesn't help much
        # If the car is already very low S (white) then changing the hue doesn't help much
        # BUT, we need to be careful bc the low V (black) is partially what helps us to determine where windows are for cars,
        # if we start handling black differently, then we need
        # adapted from: https://stackoverflow.com/questions/62648862/how-can-i-change-the-hue-of-a-certain-area-with-opencv-python
        cropped_mask = self.__crop_mask(isolated)
        cropped_mask_hls = cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2HLS)
        ch, cl, cs = cv2.split(cropped_mask_hls)
        avg_light = np.mean(cl[cl > 0])
        color_shift = color_shift if color_shift is not None else random.randint(45, 135)
        hsv = cv2.cvtColor(isolated, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hsv)
        # shift the hue to change the color
        h += color_shift
        h %= 180
        # If the object is white, increase the saturation to add color
        lit_thresh = 100
        sat_thresh = 50
        # for handling white cars
        if avg_light > 100:
            dec_lit = dec_lit if dec_lit is not None else random.randint(20, 50)
            l[l >= lit_thresh] -= dec_lit
            inc_sat = inc_sat if inc_sat is not None else random.randint(10, 40)
            locs = np.where((l >= lit_thresh) & (s <= sat_thresh))
            s[locs] += inc_sat
        else:
            pass
        hls = cv2.merge([h, l, s])
        bgr_new = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
        return bgr_new, color_shift, dec_lit, inc_sat

    def add_object(self, src, src_mask, dest, dest_loc, add_shadow=False, rotation=None):
        """
        src_mask is a whited out section representing the object to remove from src
        dest_loc is the position of the lower left of the image as a tuple (x, y)
        """
        src_mask = self.__normalize_mask(src_mask)
        isolated_addition = self.get_isolated(src, src_mask)
        if rotation is not None:
            moments = cv2.moments(src_mask, True)
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            h, w = src_mask.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
            src_mask = cv2.warpAffine(src_mask, rot_mat, (w, h))
            isolated_addition = cv2.warpAffine(isolated_addition, rot_mat, (w, h))
        if add_shadow:
            # TODO shadow needs to handle rotation param and object orientation
            shadow_factor = 0.65
            height, width = dest.shape[:2]
            src_mask_x, src_mask_y, src_mask_width, src_mask_height = self.mask_bounding_rect(src_mask)
            to_move_y = dest_loc[1] - (src_mask_y + src_mask_height)
            to_move_x = dest_loc[0] - src_mask_x
            translate_affine = self.__translate_affine(to_move_x, to_move_y)
            src_mask_desc_loc = cv2.warpAffine(src_mask, translate_affine, (width, height))
            # we will use the mask of where the car is going to go and apply a blur, then use that blur to apply shadow
            # blurring adapted from https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size))
            kernel /= np.sum(kernel)
            blurred = cv2.filter2D(src_mask_desc_loc, -1, kernel)
            blurred[blurred > 0] = 255
            # convert to convex hull, adapted from https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/
            contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hulls = [cv2.convexHull(contour, False) for contour in contours]
            for hull in hulls:
                blurred = cv2.fillPoly(blurred, pts=[hull], color=255)
            blurred[:int(src_mask_y+src_mask_height * 0.8), :] = 0  # TODO: convert to tire height?
            dest_hls = cv2.cvtColor(dest, cv2.COLOR_BGR2HLS)
            # image_HLS[:, :, 1][mask[:, :, 0] == 255] = image_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.5
            dest_hls[:, :, 1][blurred > 0] = dest_hls[:, :, 1][blurred > 0] * shadow_factor
            dest_shadowed = cv2.cvtColor(dest_hls, cv2.COLOR_HLS2BGR)
            dest = dest_shadowed
        result = self.add_isolated_object(isolated_addition, dest, dest_loc, src_mask)
        cv2.waitKey()

        return result

    def add_isolated_object(self, isolated_addition, dest, dest_loc, src_mask):
        src_mask = self.__normalize_mask(src_mask)
        height, width = dest.shape[:2]
        src_mask_x, src_mask_y, src_mask_width, src_mask_height = self.mask_bounding_rect(src_mask)
        remove_from_dest_mask_src = cv2.bitwise_not(src_mask)  # we need to black out the part we are about to add to
        to_move_y = dest_loc[1] - (src_mask_y + src_mask_height)
        to_move_x = dest_loc[0] - src_mask_x
        translate_affine = self.__translate_affine(to_move_x, to_move_y)
        isolated_addition_dest = cv2.warpAffine(isolated_addition, translate_affine, (width, height))
        # bordervalue needs to be white since we want to fill with +mask
        remove_from_dest_mask_dest = cv2.warpAffine(remove_from_dest_mask_src, translate_affine,
                                                    (width, height), borderValue=(255, 255, 255))
        emptied_dest = cv2.bitwise_and(dest, dest, mask=remove_from_dest_mask_dest)
        added_dest = cv2.add(emptied_dest, isolated_addition_dest)
        return added_dest

    def mask_bounding_rect(self, mask):
        if mask is None:
            return 0, 0, 0, 0
        # Adapted from https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        mask = self.__normalize_mask(mask)
        thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h

    def __crop_mask(self, mask):
        # Adapted from https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
        x, y, w, h = self.mask_bounding_rect(mask)
        cropped = mask[y:y+h, x:x+w]
        return cropped


    def remove_object(self, image, object_id : int):
        with tempfile.NamedTemporaryFile() as temp:

            temp.flush()

    def __call_deepfill(self, image_file, mask_file):
        pass

    def is_lighting_similar(self, road_img, random_car_image, instance_mask):
        road_hsv = cv2.cvtColor(road_img, cv2.COLOR_BGR2HSV)
        car_hsv = cv2.cvtColor(random_car_image, cv2.COLOR_BGR2HSV)
        road_isolated = self.get_isolated(road_hsv, instance_mask)
        car_isolated = self.get_isolated(car_hsv, instance_mask)
        instance_mask = np.squeeze(instance_mask)
        road_values = road_isolated[:, :, -1]
        car_values = car_isolated[:, :, -1]
        road_values = road_values[np.where(instance_mask == 255)]
        car_values = car_values[np.where(instance_mask == 255)]
        road_median_value = np.median(road_values)
        car_median_value = np.median(car_values)
        return abs(road_median_value - car_median_value) < 5
