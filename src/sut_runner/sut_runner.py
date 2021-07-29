import os

class SUTRunner:
    HOME_DIR = '/home/adwiii/'
    TEMP_DIR = HOME_DIR + 'tmp/'
    if not os.path.isdir(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    POST_FIX = '_prediction.png'

    def __init__(self, sut_name):
        self.sut_name = sut_name

    def normalize_folders(self, folder, dest_folder=None):
        if not folder[-1] == '/':  # normalize the folder_to_check name to have a trailing slash
            folder += '/'
        if dest_folder is None:
            dest_folder = folder
        elif not dest_folder[-1] == '/':
            dest_folder += '/'
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        return folder, dest_folder

    def _run_semantic_seg(self, folder, dest_folder):
        """Run the SUT on the specified input folder, then copy the results to the dest folder"""
        pass

    def run_semantic_seg(self, folder, dest_folder=None):
        folder, dest_folder = self.normalize_folders(folder, dest_folder)
        self._run_semantic_seg(folder, dest_folder)

