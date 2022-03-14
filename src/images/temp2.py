"""Helper script for generating specific instances of a mutation. Used for generating images for the paper."""
import numpy as np
from src.images.image_mutator import MutationFolder
from src.images.tester import Tester

if __name__ == '__main__':
    Tester.initialize(working_directory='/data/temp/', pool_count=1)
    Tester.cityscapes_mutator.blurred = True
    i = 0
    while True:
        temp = Tester.cityscapes_mutator.change_instance_color('car', poly_id='cologne_000004_000019_26')
        i += 1
        print(temp if temp is None else temp.params)
