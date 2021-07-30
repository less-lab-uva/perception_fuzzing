from typing import List

from src.images.image_mutator import MutationFolder
from src.sut_runner.sut_runner import SUTRunner


class SUTManager:
    def __init__(self, suts: List[SUTRunner]):
        self.suts = suts

    def run_suts(self, mutation_folder: MutationFolder):
        for sut in self.suts:
            print('---- Running SUT %s ----' % sut.name)
            sut.run_semantic_seg(mutation_folder.folder, mutation_folder.get_sut_folder(sut.name))
