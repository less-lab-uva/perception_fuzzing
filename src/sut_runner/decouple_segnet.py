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


class DecoupleSegNet(SUTRunner):

    def __init__(self, decouple_segnet_home, snapshot_path=None):
        super().__init__('decouple_segnet')
        self.DECOUPLE_SEGNET_HOME = decouple_segnet_home
        if snapshot_path is None:
            self.SNAPSHOT_PATH = self.DECOUPLE_SEGNET_HOME + '/checkpoints/GFFNet_wider_resnet.pth'
        else:
            self.SNAPSHOT_PATH = snapshot_path

        self.base_command = 'nvidia-docker run --ipc=host -v "%s:%s" --user "$(id -u):$(id -g)" ' \
                            'decouple-seg-nets-semantic-segmentation bash -c "cd %s && ' \
                            'python demo/demo_folder_decouple.py --demo_folder INPUT_DIR ' \
                            '--snapshot %s --save_dir OUTPUT_DIR ' \
                            '--arch network.gffnets.DeepWV3PlusGFFNet --color-mask-only"' \
                            % (SUTRunner.HOME_DIR, SUTRunner.HOME_DIR, self.DECOUPLE_SEGNET_HOME, self.SNAPSHOT_PATH)

    def _run_semantic_seg(self, folder, dest_folder):
        # temp dir will be automatically cleaned up on exit of the with statement
        with tempfile.TemporaryDirectory(dir=SUTRunner.TEMP_DIR) as temp_folder:
            command = self.base_command.replace('INPUT_DIR', folder).replace('OUTPUT_DIR', temp_folder)
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
                print(line.decode())
            process.wait()
            copy_output(dest_folder, temp_folder)


if __name__ == '__main__':
    DecoupleSegNet('/home/adwiii/git/DecoupleSegNets').run_semantic_seg('/home/adwiii/git/perception_fuzzing/src/images/add_car_check_perspective', SUTRunner.TEMP_DIR + '/decouple_segnet_out')
