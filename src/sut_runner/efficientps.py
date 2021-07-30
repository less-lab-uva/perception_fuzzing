import tempfile
import subprocess
import os
import glob
import uuid
from shutil import copyfile

from src.sut_runner.sut_runner import SUTRunner


def copy_output(dest_folder, temp_name):
    for file in glob.glob(temp_name + '/**/*.png'):
        short_file = file[file.rfind('/'):-4] + SUTRunner.POST_FIX  # -4 to remove .png
        copyfile(file, dest_folder + short_file)


class EfficientPS(SUTRunner):

    def __init__(self, eps_home, snapshot_path=None):
        super().__init__('efficientps')
        self.EPS_HOME = eps_home
        if snapshot_path is None:
            self.SNAPSHOT_PATH = self.EPS_HOME + '/checkpoints/efficientPS_cityscapes/model/model.pth'
        else:
            self.SNAPSHOT_PATH = snapshot_path

        self.base_command = 'nvidia-docker run --ipc=host -v "%s:%s" efficientps-semantic-segmentation ' \
                            '"pip3 install git+https://github.com/mapillary/inplace_abn.git && ' \
                            'cd %s/efficientNet && python setup.py develop && cd .. && python setup.py develop && ' \
                            'python tools/cityscapes_save_predictions.py ./configs/efficientPS_singlegpu_sample.py ' \
                            '%s INPUT_DIR OUTPUT_DIR && chown -R $(id -u):$(id -g) OUTPUT_DIR"' \
                            % (SUTRunner.HOME_DIR, SUTRunner.HOME_DIR, self.EPS_HOME, self.SNAPSHOT_PATH)

    def _run_semantic_seg(self, folder, dest_folder, verbose=False):
        # temp dir will be automatically cleaned up on exit of the with statement
        with tempfile.TemporaryDirectory(dir=SUTRunner.TEMP_DIR) as input_folder:
            # EPS expects the folder structure to be of the form:
            # input dir
            #    |- folder1
            #    |- folder2
            #        |- image1.png
            # ...
            # In order to make this happen, create a temp folder and then symlink
            # input dir
            #     |- symlink
            #        |- images
            symlink = input_folder + '/' + str(uuid.uuid4())
            os.symlink(folder, symlink)
            with tempfile.TemporaryDirectory(dir=SUTRunner.TEMP_DIR) as temp_folder:
                command = self.base_command.replace('INPUT_DIR', input_folder).replace('OUTPUT_DIR', temp_folder)
                SUTRunner._run_docker(command, verbose)
                copy_output(dest_folder, temp_folder)
                os.unlink(symlink)


if __name__ == '__main__':
    EfficientPS('/home/adwiii/git/EfficientPS').run_semantic_seg('/home/adwiii/git/perception_fuzzing/src/images/add_car_check_perspective', SUTRunner.TEMP_DIR + '/eps_out')
