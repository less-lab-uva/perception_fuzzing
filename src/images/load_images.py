from nuimages import NuImages
from nuimages.utils.utils import annotation_name, mask_decode, get_font, name_to_index_mapping
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import uuid
from image_mutator import *
from multiprocessing import Pool
# np.set_printoptions(threshold=sys.maxsize)

DATA_ROOT = '/home/adwiii/data/sets/nuimages'
nuim = NuImages(dataroot=DATA_ROOT, version='v1.0-mini', verbose=False, lazy=True)
nuim_mutator = NuScenesMutator(DATA_ROOT, 'v1.0-mini')

# DATA_ROOT_FULL = '/home/adwiii/data/sets/nuscenes_full'
# nuim = NuImages(dataroot=DATA_ROOT_FULL, version='v1.0-trainval', verbose=False, lazy=True)
# nuim_mutator = NuScenesMutator(DATA_ROOT_FULL, 'v1.0-trainval')

# sample_idx = 17
# sample = nuim.sample[sample_idx]
# print(nuim.list_anns(sample['token']))
# for sample in nuim.sample:
#     print(nuim.list_anns(sample['token']))
# exit()
# key_camera_token = sample['key_camera_token']
# semantic_mask, instance_mask = nuim.get_segmentation(key_camera_token)
# cat_to_index = name_to_index_mapping(nuim.category)
# print(cat_to_index)
# plt.figure(figsize=(32, 9))
#
# plt.subplot(1, 2, 1)
# semantic_mask = np.stack((semantic_mask,)*3, axis=-1)
# print(cat_to_index['vehicle.car'])
# print(np.unique(semantic_mask))
# semantic_mask[semantic_mask != cat_to_index['vehicle.car']] = 0
# plt.imshow(semantic_mask)
# plt.show()
# plt.subplot(1, 2, 2)

# instance_mask[instance_mask != 11] = 0
# instance_mask[instance_mask == 11] = 255

# print('sample: ')
# print(sample)
# print('---')
# print(nuim.list_categories(sample['token']))
# print('object_anns, surfance_anns')
# object_anns, surface_anns = nuim.list_anns(sample['token'])
# print('======')
# print(object_anns)
# exit()
# sample_data = nuim.get('sample_data', key_camera_token)
# print(sample_data)
# file = DATA_ROOT_FULL + '/' + sample_data['filename']
# print(file)
# img = cv2.imread(file)


# print(np.shape(img))
# print(np.shape(instance_mask))
# just_car = cv2.bitwise_and(img, img, mask=instance_mask)
# just_car = np.multiply(img, instance_mask)
# plt.imshow(just_car)
# cv2.imwrite("only_car.jpeg", just_car)
# cv2.imshow(just_car)
# plt.colorbar()


mutate = ImageMutator()
# while True:
#     random_road = random.choice(nuim_mutator.get_instances(None, 'flat.driveable_surface'))
#     random_road_sample = nuim_mutator.sample_for_surface(random_road)
#     img = nuim_mutator.load_image(random_road_sample)
#     car_loc = nuim_mutator.get_random_road_location(random_road_sample)
#     print(car_loc)
#     # instance_mask = nuim_mutator.get_instance_mask(nuim_mutator.get_random_instance(sample['token'], 'vehicle.car'))
#     # print(nuim_mutator.get_instances(None, 'vehicle.car'))
#     # print(len(nuim_mutator.get_instances(None, 'vehicle.car')))
#     random_car = random.choice(nuim_mutator.get_instances(None, 'vehicle.car'))
#     # print(random_car)
#     random_car_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car))
#     instance_mask = nuim_mutator.get_instance_mask(random_car)
#     while not mutate.is_contiguous(instance_mask) or not mutate.is_larger_than(instance_mask, (100, 100), False):
#         random_car = random.choice(nuim_mutator.get_instances(None, 'vehicle.car'))
#         random_car_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car))
#         instance_mask = nuim_mutator.get_instance_mask(random_car)
#     cv2.imshow('random_car_image', random_car_image)
#     added_car = mutate.add_object(random_car_image, instance_mask, img, car_loc)
#     cv2.imshow('added_car', added_car)
#     cv2.waitKey()

# while True:
#     # # Get random two random images and move one object from first image to second, placing on the road
#     # random_road = random.choice(nuim_mutator.get_instances(None, 'flat.driveable_surface'))
#     # random_road_sample = nuim_mutator.sample_for_surface(random_road)
#     # img = nuim_mutator.load_image(random_road_sample)
#     # car_loc = nuim_mutator.get_random_road_location(random_road_sample)
#     # print(car_loc)
#     # # instance_mask = nuim_mutator.get_instance_mask(nuim_mutator.get_random_instance(sample['token'], 'vehicle.car'))
#     # # print(nuim_mutator.get_instances(None, 'vehicle.car'))
#     # # print(len(nuim_mutator.get_instances(None, 'vehicle.car')))
#     # # obj_list = ['human.pedestrian.adult', 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility']
#     # obj_list = ['movable_object.barrier']
#     # random_car = random.choice(nuim_mutator.get_instances(None, obj_list))
#     # # print(random_car)
#     # random_car_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car))
#     # instance_mask = nuim_mutator.get_instance_mask(random_car)
#     # while instance_mask is None or not mutate.is_contiguous(instance_mask) or not mutate.is_larger_than(instance_mask, (100, 100), both=False):
#     #     random_car = random.choice(nuim_mutator.get_instances(None, obj_list))
#     #     random_car_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car))
#     #     instance_mask = nuim_mutator.get_instance_mask(random_car)
#     # cv2.imshow('random_car_image', random_car_image)
#     # added_car = mutate.add_object(random_car_image, instance_mask, img, car_loc)
#     # cv2.imshow('added_car', added_car)
#     # cv2.waitKey()
#
#     # Get random two random images containing cars and put one car overtop of a smaller car from A to B
#     obj_list = ['vehicle.car']
#     random_car_src = random.choice(nuim_mutator.get_instances(None, obj_list))
#     # print(random_car)
#     random_car_src_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car_src))
#     instance_src_mask = nuim_mutator.get_instance_mask(random_car_src)
#     xs, ys, ws, hs = mutate.mask_bounding_rect(instance_src_mask)
#
#     random_car = random.choice(nuim_mutator.get_instances(None, obj_list))
#     # print(random_car)
#     random_car_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car))
#     instance_mask = nuim_mutator.get_instance_mask(random_car)
#     x,y,w,h = mutate.mask_bounding_rect(instance_mask)
#     try_count = 100
#     while try_count > 0 and instance_mask is None or not mutate.is_contiguous(instance_mask) or not mutate.is_larger_than(instance_mask, (w, h), both=True):
#         random_car = random.choice(nuim_mutator.get_instances(None, obj_list))
#         random_car_image = nuim_mutator.load_image(nuim_mutator.sample_for_object(random_car))
#         instance_mask = nuim_mutator.get_instance_mask(random_car)
#         try_count -= 1
#     if try_count == 0:
#         continue
#     cv2.imshow('src_image', random_car_image)
#     added_car = mutate.add_object(random_car_image, instance_mask, random_car_src_image, (xs, ys+hs))
#     cv2.imshow('dest_image', random_car_src_image)
#     cv2.imshow('added_car', added_car)
#     cv2.waitKey()

        # raise e
    # nuim_mutator.random_obj_to_random_image()





# np.set_printoptions(threshold=sys.maxsize)
# print(semantic_mask)
# print(nuim.list_categories(sample_tokens=[key_camera_token]))
# object_tokens, surface_tokens = nuim.list_anns(sample['token'])
# print(object_tokens)
# print(surface_tokens)

# plt.show()