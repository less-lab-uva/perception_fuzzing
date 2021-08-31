from src.images.image_mutator import MutationFolder
from src.images.tester import Tester

if __name__ == '__main__':
    Tester.initialize(working_directory='/data/disc_training/', pool_count=25)
    Tester.run_fuzzer(folders_to_run=5, generate_only=True)
