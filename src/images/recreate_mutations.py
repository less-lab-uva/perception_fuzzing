from pathlib import Path
from image_mutator import *
from tester import *
import tqdm

if __name__ == '__main__':
    cur_dir = Path(__file__).parent
    project_root = cur_dir.parent.parent
    sets = [i for i in project_root.glob('study_data/all_mutation_logs/*.txt')]  # find mutation files from study
    sets.sort(key=lambda x: int(x.name[x.name.rfind('_')+1:-4]))  # sort numerically for consistency
    print('Found %d sets, recreating images from study' % len(sets))
    WORKING_DIR = os.getenv('WORKING_DIR')
    print('Saving results to folders in %s' % WORKING_DIR)
    Tester.initialize()
    for mutation_set in sets:
        set_num = int(mutation_set.name[mutation_set.name.rfind('_') + 1:-4])
        folder_path = WORKING_DIR + 'set_%d' % set_num
        print('Recreating mutations for set %d, saving to %s' % (set_num, folder_path))
        with mutation_set.open() as mutation_file:
            mutation_folder = MutationFolder.mutation_folder_from_file(folder_path, mutation_file)
        print('Found %d mutations to recreate' % len(mutation_folder.mutation_map))
        for mutation_name, mutation in tqdm.tqdm(mutation_folder.mutation_map.items()):
            mutation_type = mutation.params['mutation_type']
            params = dict(mutation.params)
            del params['mutation_type']
            recreated_mutation = Tester.cityscapes_mutator.apply_mutation(mutation_type, params)
            recreated_mutation.name = mutation_name  # set the name to be the same for cross-referencing
            if mutation is not None:
                recreated_mutation.update_file_names(mutation_folder)
                recreated_mutation.save_images(free_mem=True)


