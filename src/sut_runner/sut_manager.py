import os
from typing import List

from src.images.image_mutator import MutationFolder
from src.sut_runner.sut_runner import SUTRunner


class SUTManager:
    def __init__(self, suts: List[SUTRunner]):
        self.suts = suts

    def run_suts(self, mutation_folder: MutationFolder, force_recalc=False):
        print('Running SUTs for', mutation_folder.base_folder)
        if force_recalc:
            suts_to_run = [sut.name for sut in self.suts]
        else:
            suts_to_run = []
            for sut in self.suts:
                if not os.path.exists(mutation_folder.get_sut_raw_results(sut.name)):
                    suts_to_run.append(sut.name)
        for sut in self.suts:
            if sut.name not in suts_to_run:
                print('---- Skipping SUT %s ----' % sut.name)
            else:
                print('---- Running SUT %s ----' % sut.name)
                sut.run_semantic_seg(mutation_folder.folder, mutation_folder.get_sut_folder(sut.name))
