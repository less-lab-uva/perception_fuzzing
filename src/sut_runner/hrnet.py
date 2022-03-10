import shutil
import tempfile
import subprocess
import os
import glob
import uuid
from shutil import copyfile

from src.sut_runner.sut_runner import SUTRunner


def copy_output(dest_folder, temp_name):
    for file in glob.glob(temp_name + '/cityscapes'
                                      '/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484'
                                      '/test_results/*.png'):
        short_file = file[file.rfind('/'):-4] + SUTRunner.POST_FIX  # -4 to remove .png
        copyfile(file, dest_folder + short_file)


class HRNet(SUTRunner):
    """Class for running EfficientPS. Actual code for docker container from:
        https://github.com/less-lab-uva/HRNet-Semantic-Segmentation
    """
    def __init__(self, hrnet_home, snapshot_path=None):
        super().__init__('hrnet')
        self.HRNet_HOME = hrnet_home
        if snapshot_path is None:
            self.SNAPSHOT_PATH = self.HRNet_HOME + '/checkpoints/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth'
        else:
            self.SNAPSHOT_PATH = snapshot_path
        self.LIST_BASE_PATH = self.HRNet_HOME + '/data/'
        self.LIST_PATH = 'list/cityscapes'
        self.INPUT_BASE_PATH = self.HRNet_HOME + '/data/cityscapes'
        self.OUTPUT_PATH = self.HRNet_HOME + '/output/cityscapes'

        self.base_command = 'nvidia-docker run --ipc=host %s --user "$(id -u):$(id -g)" ' \
                            'hrnet-semantic-segmentation bash -c "cd %s && python tools/test.py ' \
                            '--cfg experiments/cityscapes/' \
                            'seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml ' \
                            'DATASET.TEST_SET INPUT_LIST OUTPUT_DIR OUTPUT_DIR_TO_REPLACE ' \
                            'TEST.MODEL_FILE %s TEST.FLIP_TEST False"' \
                            % (SUTRunner.DOCKER_VOLUME_STR, self.HRNet_HOME, self.SNAPSHOT_PATH)

    def _run_semantic_seg(self, folder, dest_folder, verbose=False):
        # temp dir will be automatically cleaned up on exit of the with statement
        with tempfile.TemporaryDirectory(dir=self.INPUT_BASE_PATH) as temp_input_folder:
            temp_input_folder_name = str(temp_input_folder)
            dest_file_list = []
            for file in os.listdir(folder):
                if file[-4:] != '.png':
                    continue
                new_name = '%s/%s' % (temp_input_folder_name, file)
                dest_file_list.append(new_name)
                shutil.copy2(folder + file, new_name)
            with tempfile.NamedTemporaryFile(suffix='_test.lst', dir=self.LIST_BASE_PATH + self.LIST_PATH) as temp:
                temp.write(str.encode('\n'.join(dest_file_list)))
                temp.flush()
                input_list = self.LIST_PATH + temp.name[temp.name.rfind('/'):]
                with tempfile.TemporaryDirectory(dir=self.HRNet_HOME) as temp_folder:
                    folder_name = str(temp_folder)
                    folder_name = folder_name[folder_name.rfind('/')+1:]
                    command = self.base_command.replace('INPUT_LIST',
                                                         input_list).replace('OUTPUT_DIR_TO_REPLACE', folder_name)
                    SUTRunner._run_docker(command, verbose)
                    copy_output(dest_folder, str(temp_folder))


if __name__ == '__main__':
    HRNet('/home/adwiii/git/HRNet-Semantic-Segmentation').run_semantic_seg('/home/adwiii/git/perception_fuzzing/src/images/add_car_check_perspective', SUTRunner.TEMP_DIR + '/hrnet_out')
