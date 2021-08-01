import tempfile
import subprocess
import os
from shutil import copyfile, rmtree

import traceback

from src.sut_runner.sut_runner import SUTRunner


class NVIDIASemSeg(SUTRunner):
    base_string = '''
    # Run Evaluation and Dump Images on Cityscapes with a pretrained model
    
    CMD: "python -m torch.distributed.launch --nproc_per_node=1 train.py"
    
    HPARAMS: [
      {
       dataset: cityscapes,
       cv: 0,
       syncbn: true,
       apex: true,
       fp16: true,
       bs_val: 1,
       eval: folder,
       eval_folder: 'IMAGE_FOLDER',
       dump_assets: true,
       dump_all_images: true,
       n_scales: "0.5,1.0,2.0",
       snapshot: "ASSETS_PATH/seg_weights/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth",
       arch: ocrnet.HRNet_Mscale,
       result_dir: LOGDIR,
      },
    ]
    '''

    def __init__(self, semseg_home):
        super().__init__("nvidia-semantic-segmentation")
        self.SEMANTIC_SEG_HOME = semseg_home

        self.base_command = 'nvidia-docker run --ipc=host -v "%s:%s" --user "$(id -u):$(id -g)" ' \
                            'nvidia-semantic-segmentation bash -c "cd %s && python -m runx.runx YML_FILE -i"' \
                            % (SUTRunner.HOME_DIR, SUTRunner.HOME_DIR, self.SEMANTIC_SEG_HOME)

    def _run_semantic_seg(self, folder, dest_folder, verbose=False):
        runx_string = NVIDIASemSeg.base_string.replace('IMAGE_FOLDER', folder)
        print(runx_string)
        with tempfile.NamedTemporaryFile(suffix='.yml', dir=SUTRunner.TEMP_DIR) as temp:
            temp.write(str.encode(runx_string))  # tempfile needs bytes for some reason
            temp.flush()
            command = self.base_command.replace('YML_FILE', temp.name)
            SUTRunner._run_docker(command, verbose)
            # output files are in
            # SEMANTIC_SEG_HOME/logs/temp.name/(only one folder_to_check, but name is random)/best_images
            self.copy_output(dest_folder, temp.name)

    def copy_output(self, folder, temp_name):
        if '.yml' in temp_name:
            temp_file_name = temp_name[temp_name.rfind('/') + 1:temp_name.rfind('.yml')]
        else:
            temp_file_name = temp_name
        if folder[-1] != '/':
            folder += '/'
        base_folder = self.SEMANTIC_SEG_HOME + '/logs/' + temp_file_name + '/'
        random_folder = base_folder + '/' + os.listdir(base_folder)[0]
        images_folder = random_folder + '/best_images/'
        for file in os.listdir(images_folder):
            if '_prediction.png' not in file:
                continue
            predicted_file = images_folder + file
            orig_folder = folder + file
            try:
                copyfile(predicted_file, orig_folder)
            except:
                traceback.print_exc()
                exit()
        rmtree(random_folder)


if __name__ == '__main__':
    NVIDIASemSeg('/home/adwiii/git/nvidia/semantic-segmentation').run_semantic_seg('/home/adwiii/git/perception_fuzzing/src/images/add_car_check_perspective', SUTRunner.TEMP_DIR + '/nvidiasemseg_out')
    # run_semantic_seg('/home/adwiii/Downloads/Ch2_002/output/center', '/home/adwiii/Downloads/Ch2_002/output/center_semantic')
    copy_output('/home/adwiii/Downloads/Ch2_002/output/center_semantic', 'tmp2c56t94d')


