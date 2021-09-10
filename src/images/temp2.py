import numpy as np
from src.images.image_mutator import MutationFolder
from src.images.tester import Tester

def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    # shoelace formula taken from
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

if __name__ == '__main__':
    Tester.initialize(working_directory='/data/temp/', pool_count=1)
    # areas = []
    # for citypoly in Tester.cityscapes_mutator.label_mapping['car']:
    #     areas.append(polygon_area(citypoly.polygon))
    # areas = np.array(areas)
    # print('num cars', len(areas))
    # print(np.mean(areas))
    # print(np.median(areas))
    # print('num above 1pp', np.count_nonzero(areas > (1024*2048*0.01)))
    # exit()
    Tester.cityscapes_mutator.blurred = True
    i = 0
    while True:
        # , bg_id='zurich_000085_000019_1', 'add_id': 'stuttgart_000193_000019_98'
        # {'bg_id': 'zurich_000085_000019_1', 'add_id': 'stuttgart_000193_000019_98', 'semantic_label': 'car',
         # 'mutation_type': MutationType.ADD_OBJECT}
        # temp = Tester.cityscapes_mutator.add_instance('car', bg_id='zurich_000085_000019_1')
        temp = Tester.cityscapes_mutator.change_instance_color('car', poly_id='cologne_000004_000019_26')
        i += 1
        print(temp if temp is None else temp.params)
