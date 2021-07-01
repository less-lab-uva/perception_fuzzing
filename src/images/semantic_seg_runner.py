import tempfile
import subprocess
import os
import glob
from shutil import copyfile, rmtree

import traceback

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

HOME_DIR = '/home/adwiii/'
TEMP_DIR = HOME_DIR + 'tmp/'
if not os.path.isdir(TEMP_DIR):
    os.mkdir(TEMP_DIR)
SEMANTIC_SEG_HOME = '/home/adwiii/git/nvidia/semantic-segmentation'

base_command = 'nvidia-docker run --ipc=host -v "%s:%s" --user "$(id -u):$(id -g)" nvidia-semantic-segmentation bash -c "cd %s && python -m runx.runx YML_FILE -i"' % (HOME_DIR, HOME_DIR, SEMANTIC_SEG_HOME)

def run_semantic_seg(folder, dest_folder=None):
    if not folder[-1] == '/':  # normalize the folder_to_check name to have a trailing slash
        folder += '/'
    if dest_folder is None:
        dest_folder = folder
    runx_string = base_string.replace('IMAGE_FOLDER', folder)
    print(runx_string)
    with tempfile.NamedTemporaryFile(suffix='.yml', dir=TEMP_DIR) as temp:
        # print(temp.name)
        temp.write(str.encode(runx_string))  # tempfile needs bytes for some reason
        temp.flush()
        command = base_command.replace('YML_FILE', temp.name)
        # print(command)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
            print(line.decode())
        process.wait()
        # output files are in SEMANTIC_SEG_HOME/logs/temp.name/(only one folder_to_check, but name is random)/best_images
        copy_output(dest_folder, temp.name)


def copy_output(folder, temp_name):
    if '.yml' in temp_name:
        temp_file_name = temp_name[temp_name.rfind('/') + 1:temp_name.rfind('.yml')]
    else:
        temp_file_name = temp_name
    if folder[-1] != '/':
        folder += '/'
    base_folder = SEMANTIC_SEG_HOME + '/logs/' + temp_file_name + '/'
    print(base_folder)
    random_folder = base_folder + '/' + os.listdir(base_folder)[0]
    images_folder = random_folder + '/best_images/'
    print(random_folder)
    print(images_folder)
    for file in os.listdir(images_folder):
        if '_prediction.png' not in file:
            continue
        # the prediction is of the form NAME_prediction.png
        file_name = file[:file.rfind('.')]
        # predicted_file = images_folder + file_name + '_prediction.png'
        predicted_file = images_folder + file
        orig_folder = folder + file_name + '_prediction.png'
        try:
            copyfile(predicted_file, orig_folder)
        except:
            traceback.print_exc()
            exit()
            print('could not copy %s' % predicted_file)
    # rmtree(random_folder)


if __name__ == '__main__':
    # run_semantic_seg('/home/adwiii/Downloads/Ch2_002/output/center', '/home/adwiii/Downloads/Ch2_002/output/center_semantic')
    copy_output('/home/adwiii/Downloads/Ch2_002/output/center_semantic', 'tmp2c56t94d')


