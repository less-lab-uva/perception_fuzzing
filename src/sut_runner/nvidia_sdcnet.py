import tempfile
import subprocess
import os
import glob
from shutil import copyfile

from src.sut_runner.sut_runner import SUTRunner


def copy_output(dest_folder, temp_name):
    file_prefix = 'color_mask_'
    for file in glob.glob(temp_name + '/' + file_prefix + '*.png'):
        short_file = file[file.rfind(file_prefix)+len(file_prefix):-4] + SUTRunner.POST_FIX  # -4 to remove .png
        copyfile(file, dest_folder + short_file)


class NVIDIASDCNet(SUTRunner):
    """Class for running NVIDIASDCNet. Actual code for docker container from:
        https://github.com/less-lab-uva/semantic-segmentation/tree/sdcnet
    """
    def __init__(self, sdcnet_home, snapshot_path=None):
        super().__init__('nvidia-sdcnet')
        self.SDCNET_HOME = sdcnet_home
        self.SNAPSHOT_PATH = snapshot_path

        self.base_command = 'nvidia-docker run --ipc=host %s --user "$(id -u):$(id -g)"' \
                            ' nvidia-sdcnet bash -c "cd %s && python demo_folder.py --demo-folder INPUT_DIR' \
                            ' --snapshot %s --save-dir OUTPUT_DIR --color-mask-only"' \
                            % (SUTRunner.DOCKER_VOLUME_STR, self.SDCNET_HOME, self.SNAPSHOT_PATH)

    def _run_semantic_seg(self, folder, dest_folder, verbose=False):
        # temp dir will be automatically cleaned up on exit of the with statement
        with tempfile.TemporaryDirectory(dir=SUTRunner.TEMP_DIR) as temp_folder:
            command = self.base_command.replace('INPUT_DIR', folder).replace('OUTPUT_DIR', temp_folder)
            SUTRunner._run_docker(command, verbose)
            copy_output(dest_folder, temp_folder)
