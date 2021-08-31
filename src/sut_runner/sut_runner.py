import os
import subprocess


def normalize_folders(folder, dest_folder=None):
    if not folder[-1] == '/':  # normalize the folder_to_check name to have a trailing slash
        folder += '/'
    if dest_folder is None:
        dest_folder = folder
    elif not dest_folder[-1] == '/':
        dest_folder += '/'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    return folder, dest_folder


class SUTRunner:
    HOME_DIR = '/data/'
    EXTRA_DIR = '/home/adwiii/'
    DOCKER_VOLUME_STR = '-v "' + HOME_DIR + ':' + HOME_DIR + '" -v "' + EXTRA_DIR + ':' + EXTRA_DIR + '"'
    TEMP_DIR = HOME_DIR + 'tmp/'
    if not os.path.isdir(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    POST_FIX = '_prediction.png'

    def __init__(self, sut_name):
        self.name = sut_name

    def _run_semantic_seg(self, folder, dest_folder, verbose=False):
        """Run the SUT on the specified input folder, then copy the results to the dest folder"""
        pass

    def run_semantic_seg(self, folder, dest_folder=None):
        folder, dest_folder = normalize_folders(folder, dest_folder)
        self._run_semantic_seg(folder, dest_folder)

    @staticmethod
    def _run_docker(command, verbose=False):
        if verbose:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
                print(line.decode())
        else:
            process = subprocess.Popen(command, shell=True)
        process.wait()

