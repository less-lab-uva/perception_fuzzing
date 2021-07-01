from shutil import copyfile
from os import walk

src_folder = '/home/adwiii/git/SIMS/testdata/label_refine_512'
dst_folder = '/home/adwiii/git/SIMS/result/synthesis/transform_order_512'


_, _, filenames = next(walk(src_folder))
for file in filenames:
    base_file = file[:file.rfind('.')]
    copyfile(src_folder + '/' + file, dst_folder + '/' + base_file + '_orig_prediction.png')
