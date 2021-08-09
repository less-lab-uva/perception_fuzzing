import itertools
from collections import Counter
import copy
import pprint
import time
from shutil import copyfile

import cv2
import numpy as np
from skimage.metrics import structural_similarity

import traceback
from multiprocessing import Pool
from image_mutator import *
from src.sut_runner.decouple_segnet import DecoupleSegNet
from src.sut_runner.efficientps import EfficientPS
from src.sut_runner.hrnet import HRNet
from src.sut_runner.nvidia_sdcnet import NVIDIASDCNet
from src.sut_runner.nvidia_semantic_segmentation import NVIDIASemSeg
from src.sut_runner.sut_manager import SUTManager

current_file_path = Path(__file__)
sys.path.append(str(current_file_path.parent.parent.absolute()) + '/cityscapesScripts')
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import evaluateImgLists
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import args as orig_cityscapes_eval_args


CITYSCAPES_DATA_ROOT = '/home/adwiii/data/cityscapes'
cityscapes_mutator = CityscapesMutator(CITYSCAPES_DATA_ROOT)

HIGH_DNC = ['strasbourg_000001_031683', 'strasbourg_000000_004248', 'strasbourg_000001_016681', 'strasbourg_000001_032660', 'strasbourg_000001_010162', 'strasbourg_000001_061685', 'strasbourg_000001_042235', 'krefeld_000000_013766', 'monchengladbach_000000_010733', 'strasbourg_000001_020904', 'bremen_000097_000019', 'strasbourg_000000_025907', 'bochum_000000_009951', 'monchengladbach_000000_018114', 'hamburg_000000_100300', 'bochum_000000_018195', 'hamburg_000000_023472', 'bochum_000000_028297', 'krefeld_000000_000442', 'monchengladbach_000000_026908', 'strasbourg_000001_015220', 'stuttgart_000182_000019', 'hanover_000000_045188', 'hamburg_000000_077927', 'hanover_000000_034935', 'strasbourg_000000_000751', 'krefeld_000000_034686', 'bochum_000000_031477', 'strasbourg_000001_037645', 'hamburg_000000_037161', 'hamburg_000000_066988', 'hamburg_000000_077144', 'strasbourg_000001_026355', 'hanover_000000_043653', 'ulm_000037_000019', 'hamburg_000000_092850', 'strasbourg_000000_002183', 'bochum_000000_031922', 'hamburg_000000_085321', 'hamburg_000000_013577', 'strasbourg_000000_014101', 'strasbourg_000001_030269', 'strasbourg_000000_008784', 'hamburg_000000_098616', 'hamburg_000000_038729', 'hanover_000000_000164', 'hamburg_000000_101724', 'hanover_000000_053437', 'strasbourg_000001_065572', 'hanover_000000_042770', 'zurich_000060_000019', 'krefeld_000000_021222', 'hamburg_000000_004985', 'strasbourg_000001_014033', 'hamburg_000000_029676', 'strasbourg_000001_007148', 'strasbourg_000001_017844', 'hamburg_000000_020563', 'strasbourg_000001_024701', 'krefeld_000000_030400', 'bochum_000000_020673', 'strasbourg_000000_015602', 'strasbourg_000001_055934', 'krefeld_000000_015687', 'hamburg_000000_065983', 'krefeld_000000_017342', 'hamburg_000000_046078', 'strasbourg_000001_028852', 'ulm_000019_000019', 'strasbourg_000001_039446', 'krefeld_000000_033478', 'bochum_000000_021393', 'hanover_000000_046572', 'strasbourg_000001_000508', 'krefeld_000000_034389', 'cologne_000026_000019', 'hamburg_000000_029378', 'hanover_000000_041493', 'strasbourg_000001_005219', 'hamburg_000000_084746', 'bremen_000057_000019', 'bochum_000000_014803', 'hamburg_000000_077756', 'strasbourg_000001_043080', 'hamburg_000000_085073', 'hamburg_000000_064825', 'bochum_000000_013209', 'hanover_000000_057710', 'krefeld_000000_010160', 'monchengladbach_000000_004580', 'strasbourg_000001_009097', 'hamburg_000000_000042', 'hanover_000000_012675', 'monchengladbach_000000_030662', 'bochum_000000_006746', 'strasbourg_000001_051661', 'monchengladbach_000000_023375', 'krefeld_000000_014673', 'bochum_000000_015645', 'strasbourg_000001_047336', 'hamburg_000000_039420', 'hamburg_000000_021961', 'strasbourg_000001_032315', 'hamburg_000000_024251', 'dusseldorf_000019_000019', 'krefeld_000000_036299', 'hamburg_000000_030279', 'hamburg_000000_042885', 'monchengladbach_000000_001068', 'hanover_000000_020655', 'strasbourg_000001_002644', 'hanover_000000_027282', 'hamburg_000000_060215', 'hamburg_000000_096063', 'hanover_000000_031144', 'strasbourg_000000_007813', 'strasbourg_000000_016436', 'krefeld_000000_011483', 'hamburg_000000_047390', 'hanover_000000_025437', 'dusseldorf_000028_000019', 'monchengladbach_000000_036139', 'bochum_000000_015321', 'strasbourg_000001_034633', 'strasbourg_000001_005666', 'hanover_000000_048765', 'strasbourg_000001_049399', 'hanover_000000_023881', 'strasbourg_000000_006264', 'bochum_000000_006026', 'tubingen_000028_000019', 'monchengladbach_000000_002972', 'strasbourg_000000_030122', 'strasbourg_000001_037776', 'hanover_000000_052729', 'bochum_000000_034936', 'monchengladbach_000000_014685', 'krefeld_000000_027596', 'hamburg_000000_055039', 'hanover_000000_056457', 'hanover_000000_019116', 'krefeld_000000_032614', 'strasbourg_000001_014258', 'hanover_000000_029455', 'monchengladbach_000000_020596', 'strasbourg_000001_054639', 'monchengladbach_000000_023489', 'strasbourg_000001_008310', 'strasbourg_000001_024152', 'hanover_000000_027390', 'strasbourg_000001_063385', 'darmstadt_000072_000019', 'bochum_000000_023648', 'hanover_000000_016558', 'tubingen_000078_000019', 'hanover_000000_037298', 'strasbourg_000001_015605', 'strasbourg_000001_051574', 'bochum_000000_026056', 'strasbourg_000001_038281', 'hanover_000000_048508', 'strasbourg_000000_014503', 'monchengladbach_000000_022361', 'bochum_000000_010700', 'krefeld_000000_015494', 'strasbourg_000000_013223', 'hamburg_000000_046566', 'krefeld_000000_026580', 'monchengladbach_000000_009690', 'hamburg_000000_094185', 'hanover_000000_034720', 'monchengladbach_000000_035650', 'bremen_000304_000019', 'hanover_000000_038927', 'monchengladbach_000000_009191', 'strasbourg_000001_043748', 'hamburg_000000_054220', 'strasbourg_000001_059433', 'monchengladbach_000000_018720', 'strasbourg_000001_063808', 'bochum_000000_038150', 'hamburg_000000_036527', 'cologne_000016_000019', 'krefeld_000000_005503', 'strasbourg_000000_018358', 'bremen_000056_000019', 'strasbourg_000000_026575', 'monchengladbach_000000_006169', 'bochum_000000_011255', 'hamburg_000000_033506', 'erfurt_000080_000019', 'monchengladbach_000000_005686', 'hamburg_000000_093325', 'strasbourg_000001_058105', 'hamburg_000000_016691', 'bochum_000000_004748', 'strasbourg_000001_005289', 'hanover_000000_019282', 'strasbourg_000001_010755', 'hanover_000000_011471', 'strasbourg_000000_004112', 'hanover_000000_000381', 'strasbourg_000000_024945', 'cologne_000011_000019', 'hamburg_000000_059339', 'strasbourg_000001_035562', 'tubingen_000083_000019', 'strasbourg_000001_035689', 'strasbourg_000000_029400', 'bochum_000000_029721', 'hanover_000000_032351', 'krefeld_000000_021000', 'strasbourg_000001_031116', 'hamburg_000000_037279', 'strasbourg_000001_010445', 'hanover_000000_048379', 'krefeld_000000_017042', 'hanover_000000_028202', 'hanover_000000_031856', 'hanover_000000_002357', 'strasbourg_000000_023064', 'strasbourg_000001_003991', 'strasbourg_000001_057517', 'krefeld_000000_013139', 'strasbourg_000000_030017', 'hamburg_000000_093572', 'krefeld_000000_007325', 'erfurt_000000_000019', 'strasbourg_000001_064224', 'hanover_000000_007342', 'krefeld_000000_030560', 'strasbourg_000001_014629', 'strasbourg_000001_031272', 'strasbourg_000000_030435', 'hanover_000000_039021', 'bochum_000000_007651', 'strasbourg_000001_052544', 'hamburg_000000_027304', 'stuttgart_000029_000019', 'hanover_000000_024719', 'bochum_000000_000885', 'krefeld_000000_000108', 'krefeld_000000_027954', 'bochum_000000_031152', 'bochum_000000_004229', 'cologne_000081_000019', 'bremen_000235_000019', 'hanover_000000_010403', 'strasbourg_000000_029051', 'strasbourg_000000_028556', 'bremen_000055_000019', 'erfurt_000049_000019', 'monchengladbach_000000_021104', 'hamburg_000000_087216', 'monchengladbach_000000_015126', 'krefeld_000000_010329', 'monchengladbach_000000_000076', 'krefeld_000000_010653', 'hamburg_000000_066424', 'krefeld_000000_001908', 'strasbourg_000001_041215', 'strasbourg_000001_018432', 'bochum_000000_027057', 'monchengladbach_000000_012376', 'hanover_000000_009420', 'krefeld_000000_002083', 'strasbourg_000000_015764', 'hamburg_000000_086499', 'krefeld_000000_026919', 'dusseldorf_000067_000019', 'hamburg_000000_070444', 'strasbourg_000001_006562', 'strasbourg_000000_007441', 'strasbourg_000000_033062', 'hanover_000000_016038', 'monchengladbach_000000_015561', 'darmstadt_000069_000019', 'cologne_000111_000019', 'monchengladbach_000001_000537', 'bochum_000000_037223', 'monchengladbach_000001_000054', 'strasbourg_000001_034494', 'hamburg_000000_103541', 'hanover_000000_021337', 'hamburg_000000_057678', 'strasbourg_000001_013767', 'krefeld_000000_019697', 'strasbourg_000000_008603', 'cologne_000114_000019', 'monchengladbach_000000_022748', 'tubingen_000053_000019', 'strasbourg_000001_011990', 'monchengladbach_000000_019142', 'weimar_000054_000019', 'hanover_000000_003411', 'bochum_000000_024343', 'strasbourg_000001_054275', 'krefeld_000000_020334', 'strasbourg_000000_035255', 'monchengladbach_000000_018445', 'hanover_000000_046646', 'hanover_000000_019456', 'hamburg_000000_083696', 'hanover_000000_050228', 'strasbourg_000000_022489', 'monchengladbach_000000_018575', 'hamburg_000000_054029', 'krefeld_000000_029050', 'strasbourg_000000_007727', 'hanover_000000_006922', 'hamburg_000000_006322', 'krefeld_000000_000316', 'strasbourg_000000_005249', 'hamburg_000000_091155', 'hanover_000000_012347', 'hanover_000000_008200', 'weimar_000099_000019', 'strasbourg_000000_029481', 'strasbourg_000000_030706', 'bochum_000000_027699', 'cologne_000000_000019', 'hanover_000000_052512', 'strasbourg_000001_008576', 'strasbourg_000001_042309', 'bochum_000000_024524', 'monchengladbach_000000_018294', 'hamburg_000000_068916', 'monchengladbach_000001_001531', 'strasbourg_000000_014416', 'strasbourg_000001_016481', 'hanover_000000_045446', 'strasbourg_000001_009795', 'hamburg_000000_060586', 'strasbourg_000000_023854', 'bochum_000000_021070', 'hamburg_000000_032906', 'monchengladbach_000000_034302', 'strasbourg_000001_049977', 'aachen_000019_000019', 'hamburg_000000_061468', 'hamburg_000000_099109', 'krefeld_000000_030221', 'strasbourg_000000_025268', 'monchengladbach_000000_006518', 'krefeld_000000_008305', 'monchengladbach_000000_017950', 'strasbourg_000001_057811', 'strasbourg_000001_003489', 'hanover_000000_035606', 'stuttgart_000188_000019', 'krefeld_000000_023698', 'strasbourg_000000_032346', 'hamburg_000000_039546', 'hamburg_000000_058591', 'bochum_000000_023174', 'hamburg_000000_102574', 'hamburg_000000_105123', 'strasbourg_000001_047955', 'hanover_000000_046954', 'strasbourg_000000_033129', 'hanover_000000_009004', 'darmstadt_000063_000019', 'monchengladbach_000000_024964', 'hamburg_000000_076392', 'strasbourg_000001_052430', 'hamburg_000000_003904', 'strasbourg_000001_062691', 'strasbourg_000000_014931', 'bochum_000000_037829', 'hamburg_000000_088054', 'strasbourg_000000_009619', 'hanover_000000_040221', 'bochum_000000_007150', 'strasbourg_000001_057129', 'bochum_000000_022210', 'hanover_000000_041610', 'hanover_000000_001620', 'hamburg_000000_065604', 'bochum_000000_003005', 'hanover_000000_046732', 'krefeld_000000_030701', 'hanover_000000_051059', 'krefeld_000000_026269', 'bochum_000000_008448', 'strasbourg_000001_001901', 'krefeld_000000_014146', 'krefeld_000000_027075', 'hanover_000000_004646', 'hamburg_000000_045437', 'hanover_000000_042255', 'strasbourg_000001_006153', 'hamburg_000000_081299', 'bochum_000000_002562', 'strasbourg_000001_031427', 'bochum_000000_001519', 'strasbourg_000001_042434', 'hanover_000000_054965', 'strasbourg_000000_029577', 'hanover_000000_055124', 'strasbourg_000000_029281', 'hamburg_000000_047157', 'strasbourg_000000_017081', 'monchengladbach_000000_012672', 'hamburg_000000_053486', 'strasbourg_000001_059675', 'monchengladbach_000000_009615', 'strasbourg_000001_036697', 'krefeld_000000_029704', 'hamburg_000000_046510', 'hamburg_000000_103856', 'hamburg_000000_047220', 'strasbourg_000000_026998', 'hanover_000000_011170', 'strasbourg_000001_047702', 'bochum_000000_005936', 'bremen_000158_000019', 'bochum_000000_015038', 'strasbourg_000000_033838', 'monchengladbach_000000_021663', 'hanover_000000_004230', 'jena_000042_000019', 'monchengladbach_000000_031005', 'hamburg_000000_001613', 'strasbourg_000000_029839', 'strasbourg_000000_025772', 'stuttgart_000113_000019', 'strasbourg_000001_051877', 'hanover_000000_051152', 'strasbourg_000001_001072', 'strasbourg_000001_030539', 'strasbourg_000001_040564', 'hamburg_000000_044996', 'strasbourg_000001_001722', 'bremen_000197_000019', 'hamburg_000000_025986', 'hanover_000000_047629', 'strasbourg_000001_000778', 'strasbourg_000001_065214', 'strasbourg_000001_021951', 'hamburg_000000_076966', 'hamburg_000000_070334', 'hanover_000000_014319', 'tubingen_000038_000019', 'hanover_000000_044195', 'hamburg_000000_098862', 'hamburg_000000_032719', 'hamburg_000000_038511', 'strasbourg_000000_029020', 'hamburg_000000_008494', 'monchengladbach_000000_028216', 'hamburg_000000_031971', 'hanover_000000_049465', 'hamburg_000000_034049', 'hanover_000000_026743', 'hamburg_000000_007737', 'strasbourg_000000_017593', 'strasbourg_000001_007524', 'hanover_000000_056142', 'bochum_000000_033331', 'strasbourg_000000_016311', 'bochum_000000_013705', 'strasbourg_000000_010372', 'strasbourg_000001_016376', 'strasbourg_000000_011225', 'strasbourg_000000_035942', 'strasbourg_000000_013863', 'krefeld_000000_025812', 'bremen_000011_000019', 'strasbourg_000001_039374', 'monchengladbach_000001_002229', 'bochum_000000_016758', 'jena_000051_000019', 'strasbourg_000000_004660', 'hamburg_000000_025802', 'strasbourg_000001_047755', 'bochum_000000_020776', 'hamburg_000000_048494', 'strasbourg_000001_056142', 'stuttgart_000068_000019', 'hamburg_000000_062371', 'hamburg_000000_069177', 'strasbourg_000000_021651', 'hamburg_000000_044251', 'hamburg_000000_028056', 'hamburg_000000_028608', 'hamburg_000000_019892', 'strasbourg_000001_037090', 'bremen_000023_000019', 'hamburg_000000_005639', 'aachen_000008_000019', 'dusseldorf_000064_000019', 'strasbourg_000001_048121', 'monchengladbach_000000_029526', 'hamburg_000000_082187', 'bochum_000000_036606', 'strasbourg_000000_026882', 'bochum_000000_014658', 'hanover_000000_023975', 'strasbourg_000000_033425', 'cologne_000032_000019', 'hanover_000000_040793', 'hanover_000000_040133', 'monchengladbach_000001_002353', 'hamburg_000000_078407', 'hamburg_000000_099902', 'hamburg_000000_090742', 'krefeld_000000_028378', 'hamburg_000000_097447', 'strasbourg_000000_031223', 'aachen_000109_000019', 'strasbourg_000001_057191', 'strasbourg_000000_031067', 'bochum_000000_031687', 'hamburg_000000_053563', 'hanover_000000_046398', 'hanover_000000_039470', 'krefeld_000000_023338', 'hanover_000000_043822', 'krefeld_000000_020873', 'strasbourg_000001_012956', 'strasbourg_000001_056330', 'strasbourg_000000_015506', 'strasbourg_000001_004745', 'monchengladbach_000000_013228', 'monchengladbach_000000_029240', 'strasbourg_000000_002553', 'monchengladbach_000000_024243', 'hanover_000000_037039', 'monchengladbach_000000_031360', 'hamburg_000000_057816', 'krefeld_000000_014886', 'bochum_000000_005537', 'hamburg_000000_042505', 'monchengladbach_000000_015685', 'hanover_000000_040294', 'hanover_000000_053604', 'strasbourg_000000_013574', 'hanover_000000_013205', 'strasbourg_000001_037906', 'strasbourg_000000_034040', 'hanover_000000_013814', 'hamburg_000000_044400', 'monchengladbach_000000_027628', 'strasbourg_000001_000710', 'strasbourg_000001_039231', 'strasbourg_000000_031323', 'strasbourg_000000_028912', 'strasbourg_000001_045481', 'hamburg_000000_037036', 'strasbourg_000000_035008', 'bremen_000054_000019', 'hamburg_000000_016447', 'hamburg_000000_098061', 'monchengladbach_000000_005876', 'hamburg_000000_073999', 'monchengladbach_000000_028563', 'strasbourg_000001_033027', 'hamburg_000000_093787', 'strasbourg_000001_018155', 'bochum_000000_024717', 'cologne_000036_000019', 'hanover_000000_049005', 'strasbourg_000001_013914', 'strasbourg_000000_010816', 'hanover_000000_035491', 'hamburg_000000_074545', 'weimar_000104_000019', 'strasbourg_000001_007657', 'strasbourg_000001_036480', 'strasbourg_000001_003676', 'strasbourg_000001_018872', 'cologne_000083_000019', 'strasbourg_000001_060173', 'bochum_000000_007950', 'strasbourg_000001_053579', 'hanover_000000_042581', 'bochum_000000_022414', 'hamburg_000000_054850', 'erfurt_000060_000019', 'monchengladbach_000000_020856', 'hanover_000000_045004', 'monchengladbach_000000_015928', 'monchengladbach_000000_007695', 'monchengladbach_000000_011383', 'hamburg_000000_022524', 'monchengladbach_000000_007098', 'hamburg_000000_049558', 'krefeld_000000_030111', 'hanover_000000_030276', 'hanover_000000_032210', 'monchengladbach_000000_025215', 'strasbourg_000000_030324', 'monchengladbach_000000_023052', 'strasbourg_000001_020956', 'bochum_000000_001828', 'strasbourg_000001_052497', 'hanover_000000_026183', 'hamburg_000000_080674', 'krefeld_000000_016863', 'strasbourg_000000_027233', 'hamburg_000000_091038', 'krefeld_000000_000926', 'hamburg_000000_094717', 'hanover_000000_050398', 'krefeld_000000_032845', 'strasbourg_000001_030839', 'strasbourg_000000_018874', 'hanover_000000_020089', 'strasbourg_000000_017159', 'krefeld_000000_011655', 'hamburg_000000_046619', 'hanover_000000_034015', 'bremen_000082_000019', 'hamburg_000000_052904', 'hanover_000000_040456', 'krefeld_000000_020624', 'hanover_000000_018213', 'erfurt_000064_000019', 'strasbourg_000000_013322', 'strasbourg_000001_009618', 'krefeld_000000_019791', 'hamburg_000000_062039', 'hamburg_000000_008221', 'hanover_000000_019938', 'hamburg_000000_063698', 'hamburg_000000_103367', 'hamburg_000000_045704', 'ulm_000053_000019', 'bochum_000000_000600', 'krefeld_000000_023510', 'hamburg_000000_050160', 'strasbourg_000001_052050', 'hanover_000000_005970', 'strasbourg_000000_021231', 'hamburg_000000_014940', 'hamburg_000000_085982', 'hamburg_000000_091900', 'monchengladbach_000000_000383', 'hamburg_000000_062710', 'hamburg_000000_088627', 'strasbourg_000001_033448', 'strasbourg_000001_046324', 'krefeld_000000_028638', 'strasbourg_000001_013266', 'hanover_000000_002140', 'strasbourg_000001_062362', 'hanover_000000_034347', 'strasbourg_000001_025833', 'krefeld_000000_009926', 'hanover_000000_023276', 'strasbourg_000001_035276', 'hamburg_000000_098400', 'hanover_000000_024441', 'hamburg_000000_041667', 'bochum_000000_024855', 'hamburg_000000_001106', 'strasbourg_000001_034375', 'hamburg_000000_036427', 'krefeld_000000_018747', 'strasbourg_000001_052979', 'hamburg_000000_055414', 'bochum_000000_026634', 'hamburg_000000_061790', 'hanover_000000_034141', 'bochum_000000_004032', 'strasbourg_000001_051448', 'strasbourg_000000_020432', 'strasbourg_000000_004383', 'strasbourg_000001_056857', 'monchengladbach_000000_020303', 'bochum_000000_028764', 'strasbourg_000001_006916', 'hamburg_000000_053086', 'strasbourg_000001_002519', 'hanover_000000_006355', 'krefeld_000000_017489', 'bochum_000000_038022', 'strasbourg_000000_027156', 'hanover_000000_049269', 'krefeld_000000_020033', 'hamburg_000000_083586', 'hamburg_000000_087822', 'krefeld_000000_008584', 'strasbourg_000000_025491', 'hanover_000000_001173', 'strasbourg_000001_064393', 'hanover_000000_005175', 'strasbourg_000000_005995', 'hamburg_000000_035568', 'hamburg_000000_038915', 'bochum_000000_019188', 'hanover_000000_026804', 'hanover_000000_058189', 'krefeld_000000_023143', 'strasbourg_000000_026741', 'strasbourg_000001_045135', 'hamburg_000000_079376', 'hanover_000000_022645', 'bochum_000000_024196', 'strasbourg_000001_030120', 'hamburg_000000_103186', 'strasbourg_000000_006106', 'hamburg_000000_073549', 'strasbourg_000001_044219', 'krefeld_000000_032390', 'strasbourg_000001_052840', 'monchengladbach_000000_007851', 'hamburg_000000_027857', 'strasbourg_000000_014584', 'strasbourg_000001_061472', 'bochum_000000_017216', 'hamburg_000000_019760', 'strasbourg_000001_047619', 'hamburg_000000_068693', 'hamburg_000000_074425', 'strasbourg_000001_004260', 'hamburg_000000_047057', 'strasbourg_000001_024379', 'strasbourg_000001_017675', 'strasbourg_000001_022560', 'strasbourg_000001_033925', 'krefeld_000000_015116', 'monchengladbach_000000_023856', 'hanover_000000_030889', 'strasbourg_000000_034652', 'bochum_000000_010562', 'strasbourg_000001_061285', 'hanover_000000_044085', 'hanover_000000_045657', 'monchengladbach_000000_002255', 'hamburg_000000_038446', 'krefeld_000000_024921', 'hamburg_000000_074139', 'strasbourg_000001_029696', 'hanover_000000_032681', 'monchengladbach_000000_028883', 'krefeld_000000_015868', 'hamburg_000000_105464', 'strasbourg_000000_019229', 'hanover_000000_036562', 'stuttgart_000102_000019', 'hamburg_000000_077642', 'strasbourg_000000_033747', 'strasbourg_000000_031602', 'hamburg_000000_060907', 'strasbourg_000000_030941', 'bochum_000000_002293', 'strasbourg_000001_022151', 'hanover_000000_030546', 'krefeld_000000_008239', 'hamburg_000000_052122', 'hamburg_000000_057487', 'strasbourg_000001_017540', 'hamburg_000000_053776', 'strasbourg_000001_029178', 'hamburg_000000_065055', 'monchengladbach_000000_019901', 'hanover_000000_029325', 'hamburg_000000_061048', 'hanover_000000_009128', 'strasbourg_000000_017283', 'strasbourg_000000_035713', 'strasbourg_000000_029339', 'krefeld_000000_020933', 'hamburg_000000_071016', 'strasbourg_000001_062542', 'hamburg_000000_040021', 'hamburg_000000_059720', 'monchengladbach_000000_003442', 'hanover_000000_024276', 'hanover_000000_051842', 'strasbourg_000000_016247', 'strasbourg_000001_030997', 'strasbourg_000000_006995', 'krefeld_000000_018866', 'bochum_000000_023435', 'hanover_000000_014537', 'hamburg_000000_105724', 'bremen_000178_000019', 'bochum_000000_033531', 'aachen_000117_000019', 'strasbourg_000001_034923', 'strasbourg_000001_019247', 'hamburg_000000_062964', 'hanover_000000_011971', 'krefeld_000000_016342', 'strasbourg_000000_013654', 'hamburg_000000_080169', 'hamburg_000000_067587', 'strasbourg_000000_019891', 'strasbourg_000000_003846', 'monchengladbach_000000_034930', 'strasbourg_000001_052297', 'hamburg_000000_084865', 'strasbourg_000000_018616', 'krefeld_000000_003096', 'hanover_000000_027998', 'hamburg_000000_064269', 'dusseldorf_000057_000019', 'strasbourg_000001_022363', 'hamburg_000000_066706', 'strasbourg_000001_022836', 'hanover_000000_044344', 'hamburg_000000_002095', 'hamburg_000000_002338', 'strasbourg_000001_011617', 'strasbourg_000001_043886', 'hanover_000000_023239', 'ulm_000015_000019', 'monchengladbach_000000_034621', 'strasbourg_000001_040981', 'hamburg_000000_043944', 'hanover_000000_014919', 'monchengladbach_000000_009930', 'erfurt_000063_000019', 'strasbourg_000001_061384', 'hamburg_000000_039264', 'hamburg_000000_036003', 'hamburg_000000_085413', 'strasbourg_000001_049776', 'hamburg_000000_069289', 'hanover_000000_035768', 'hamburg_000000_090398', 'hanover_000000_027007', 'hanover_000000_056800', 'strasbourg_000000_028628', 'bochum_000000_011711', 'strasbourg_000000_012934', 'hanover_000000_055592', 'hamburg_000000_056229', 'strasbourg_000001_004106', 'hanover_000000_054276', 'hamburg_000000_102379', 'strasbourg_000001_053976', 'hamburg_000000_079657', 'monchengladbach_000000_026305', 'strasbourg_000001_039558', 'dusseldorf_000031_000019', 'bochum_000000_017453', 'strasbourg_000001_042558', 'hamburg_000000_051855', 'strasbourg_000001_000113', 'hanover_000000_032559', 'krefeld_000000_035398', 'hamburg_000000_003488', 'hamburg_000000_047108', 'strasbourg_000000_003632', 'hamburg_000000_028439', 'monchengladbach_000000_001294', 'strasbourg_000001_039114', 'bochum_000000_033056', 'stuttgart_000019_000019', 'strasbourg_000001_060821', 'strasbourg_000001_026606', 'krefeld_000000_006686', 'strasbourg_000001_051134', 'stuttgart_000181_000019', 'strasbourg_000000_029915', 'bochum_000000_025746', 'hamburg_000000_067799', 'bochum_000000_035958', 'strasbourg_000001_028379', 'hamburg_000000_014030', 'bochum_000000_001097', 'ulm_000074_000019', 'monchengladbach_000000_010860', 'bochum_000000_003245', 'dusseldorf_000068_000019', 'krefeld_000000_025434', 'strasbourg_000001_003159', 'hamburg_000000_030953', 'hanover_000000_051271', 'hamburg_000000_080878', 'strasbourg_000001_051934', 'bochum_000000_009554', 'dusseldorf_000075_000019', 'hanover_000000_030346', 'krefeld_000000_024604', 'bochum_000000_027951', 'strasbourg_000001_017469', 'strasbourg_000000_026611', 'monchengladbach_000000_033683', 'dusseldorf_000194_000019', 'strasbourg_000001_015974', 'strasbourg_000000_024179', 'strasbourg_000000_005912', 'bremen_000073_000019', 'hanover_000000_027481', 'strasbourg_000000_001278', 'monchengladbach_000000_002478', 'hamburg_000000_029144', 'krefeld_000000_005252', 'strasbourg_000000_018153', 'monchengladbach_000000_010280', 'krefeld_000000_012505', 'ulm_000013_000019', 'hamburg_000000_073758', 'hanover_000000_045841', 'strasbourg_000001_006386', 'hanover_000000_027561', 'hamburg_000000_104857', 'hamburg_000000_073314', 'strasbourg_000000_025089', 'strasbourg_000000_009110', 'strasbourg_000000_006483', 'hamburg_000000_055894', 'monchengladbach_000000_033454', 'strasbourg_000000_012070', 'strasbourg_000001_051317', 'monchengladbach_000001_000876', 'strasbourg_000001_052198', 'hanover_000000_043102', 'strasbourg_000001_010640', 'hamburg_000000_048750', 'strasbourg_000000_017044', 'strasbourg_000001_002354', 'hanover_000000_013094', 'hanover_000000_052649', 'hanover_000000_056361', 'dusseldorf_000052_000019', 'hanover_000000_042992', 'stuttgart_000171_000019', 'bremen_000264_000019', 'hamburg_000000_105296', 'hamburg_000000_053886', 'strasbourg_000000_034387', 'stuttgart_000189_000019', 'hamburg_000000_071942', 'krefeld_000000_013257', 'bochum_000000_020899', 'krefeld_000000_003707', 'hamburg_000000_088939', 'krefeld_000000_004447', 'strasbourg_000000_014066', 'monchengladbach_000000_035083', 'strasbourg_000000_014235', 'bochum_000000_032169', 'krefeld_000000_022162', 'strasbourg_000001_055273', 'bochum_000000_021606', 'hanover_000000_051536', 'tubingen_000079_000019', 'strasbourg_000000_019355', 'strasbourg_000001_055698', 'strasbourg_000000_028822', 'krefeld_000000_034156', 'hanover_000000_018546', 'strasbourg_000000_008677', 'strasbourg_000000_029729', 'hamburg_000000_089491', 'krefeld_000000_031257', 'hamburg_000000_056508', 'cologne_000122_000019', 'hanover_000000_057532', 'strasbourg_000000_000295', 'strasbourg_000001_011775', 'hanover_000000_014713', 'hanover_000000_026014', 'hamburg_000000_011641', 'monchengladbach_000000_019500', 'strasbourg_000000_032186', 'strasbourg_000000_020653', 'strasbourg_000001_040620', 'hamburg_000000_069096', 'hamburg_000000_020211', 'hamburg_000000_048960', 'krefeld_000000_019125', 'stuttgart_000100_000019', 'strasbourg_000001_005876', 'stuttgart_000059_000019', 'bremen_000211_000019', 'strasbourg_000001_009246', 'strasbourg_000001_050098', 'strasbourg_000001_008771', 'bochum_000000_006484', 'hanover_000000_048274', 'hamburg_000000_078842', 'bochum_000000_016591', 'strasbourg_000001_058373', 'hanover_000000_026356', 'strasbourg_000001_039703', 'hamburg_000000_037741', 'krefeld_000000_009574', 'monchengladbach_000001_001936', 'bochum_000000_016125', 'hamburg_000000_069417', 'hamburg_000000_006192', 'hanover_000000_043550', 'strasbourg_000000_019617', 'hanover_000000_003224', 'hamburg_000000_080438', 'hamburg_000000_103075', 'hanover_000000_019672', 'bochum_000000_021325', 'hanover_000000_030781', 'krefeld_000000_012353', 'hanover_000000_038773', 'hamburg_000000_054555', 'monchengladbach_000000_019682', 'strasbourg_000001_029980', 'krefeld_000000_004608', 'hanover_000000_018800', 'bochum_000000_021479', 'strasbourg_000000_000065', 'strasbourg_000001_019698', 'hanover_000000_040051', 'strasbourg_000000_006621', 'krefeld_000000_006274', 'strasbourg_000001_048605', 'hanover_000000_053027', 'hanover_000000_052887', 'hamburg_000000_000629', 'bochum_000000_030913', 'hanover_000000_015849', 'hanover_000000_023614', 'strasbourg_000001_023271', 'hanover_000000_047870', 'cologne_000008_000019', 'dusseldorf_000111_000019', 'hamburg_000000_104428', 'hanover_000000_037516', 'strasbourg_000000_023694', 'hamburg_000000_071150', 'hamburg_000000_019373', 'hamburg_000000_067223', 'hanover_000000_043236', 'strasbourg_000000_027771', 'krefeld_000000_009404', 'hamburg_000000_106102', 'hanover_000000_005732', 'hamburg_000000_016928', 'hanover_000000_034560', 'strasbourg_000000_019050', 'hanover_000000_055800', 'krefeld_000000_018004', 'hamburg_000000_097086', 'hamburg_000000_088783', 'hanover_000000_029404', 'strasbourg_000001_023515', 'hanover_000000_056601', 'hanover_000000_036051', 'strasbourg_000001_042869', 'hanover_000000_004752', 'hamburg_000000_046872', 'strasbourg_000000_035571', 'hamburg_000000_015350', 'bochum_000000_008804', 'strasbourg_000001_060061', 'strasbourg_000001_036937', 'strasbourg_000000_022067', 'hanover_000000_003853', 'monchengladbach_000000_030010', 'hamburg_000000_073672', 'hanover_000000_055937', 'ulm_000075_000019', 'hanover_000000_005599', 'hamburg_000000_095561', 'monchengladbach_000000_013352', 'strasbourg_000000_017450', 'strasbourg_000001_059914', 'monchengladbach_000000_024637', 'bochum_000000_025833', 'strasbourg_000000_017761', 'hanover_000000_008017', 'hamburg_000000_021353', 'hanover_000000_007780', 'strasbourg_000000_016024', 'erfurt_000086_000019', 'hanover_000000_005288', 'hanover_000000_027766', 'strasbourg_000001_002081', 'monchengladbach_000000_026602', 'krefeld_000000_003937', 'hamburg_000000_032460', 'hamburg_000000_063403', 'hanover_000000_025335', 'hanover_000000_000712', 'strasbourg_000000_036016', 'hamburg_000000_073389', 'krefeld_000000_021814', 'hanover_000000_044622', 'hamburg_000000_096624', 'strasbourg_000001_004983', 'hanover_000000_041232', 'hamburg_000000_044747', 'strasbourg_000001_007864', 'hamburg_000000_088197', 'bochum_000000_003674', 'krefeld_000000_024362', 'hamburg_000000_026675', 'strasbourg_000001_045880', 'hanover_000000_017041', 'bochum_000000_008162', 'bochum_000000_037039', 'strasbourg_000001_040761', 'cologne_000113_000019', 'hanover_000000_029043', 'strasbourg_000001_025426', 'strasbourg_000001_002949', 'bremen_000026_000019', 'hamburg_000000_085645', 'strasbourg_000000_015131', 'hanover_000000_042382', 'hamburg_000000_065843', 'strasbourg_000000_034097', 'krefeld_000000_035124', 'monchengladbach_000000_010505', 'monchengladbach_000000_032540', 'hamburg_000000_032266', 'strasbourg_000001_009471', 'strasbourg_000001_036232', 'hanover_000000_002458', 'hamburg_000000_071675', 'dusseldorf_000132_000019', 'hamburg_000000_099368', 'hamburg_000000_048138', 'bochum_000000_029203', 'hamburg_000000_082301', 'hanover_000000_010553', 'strasbourg_000000_013944', 'bochum_000000_015880', 'strasbourg_000000_029179', 'hanover_000000_028460', 'strasbourg_000001_049143', 'strasbourg_000001_027097', 'aachen_000075_000019', 'strasbourg_000001_002216', 'strasbourg_000001_026106', 'hamburg_000000_045908', 'strasbourg_000000_032962', 'bremen_000017_000019', 'strasbourg_000001_055860', 'krefeld_000000_001566', 'bochum_000000_000313', 'strasbourg_000000_025351', 'hanover_000000_024136', 'strasbourg_000001_057930', 'hanover_000000_015587', 'hanover_000000_052013', 'hamburg_000000_088983', 'hamburg_000000_089696', 'monchengladbach_000000_031623', 'strasbourg_000001_026856', 'hanover_000000_024989', 'monchengladbach_000000_035364', 'strasbourg_000001_016253', 'strasbourg_000000_010049', 'bochum_000000_023040', 'bochum_000000_016260', 'hamburg_000000_067338', 'bochum_000000_033714', 'hanover_000000_033457', 'krefeld_000000_021553', 'strasbourg_000000_026316', 'cologne_000120_000019', 'hamburg_000000_018878', 'krefeld_000000_018514', 'strasbourg_000001_009333', 'monchengladbach_000000_035718', 'strasbourg_000001_031582', 'bochum_000000_014332', 'hanover_000000_038855', 'hamburg_000000_018592', 'cologne_000015_000019', 'cologne_000053_000019', 'strasbourg_000000_014743', 'hamburg_000000_086636', 'aachen_000162_000019', 'strasbourg_000001_053222', 'hamburg_000000_078579', 'hanover_000000_029769', 'krefeld_000000_034231', 'cologne_000064_000019', 'hanover_000000_047499', 'hanover_000000_046200', 'strasbourg_000000_028240', 'strasbourg_000000_011880', 'hamburg_000000_092476', 'strasbourg_000001_018742', 'strasbourg_000001_001449', 'hamburg_000000_074267', 'hamburg_000000_074694', 'bremen_000254_000019', 'strasbourg_000000_004951', 'strasbourg_000001_030725', 'hanover_000000_027650', 'stuttgart_000017_000019', 'krefeld_000000_024276', 'strasbourg_000001_058954', 'monchengladbach_000000_005138', 'aachen_000090_000019', 'strasbourg_000001_031976', 'monchengladbach_000000_026006', 'monchengladbach_000001_000168', 'hanover_000000_007897']

def create_images(folder: MutationFolder, count, start_num, mutation_type: MutationType, arg_dict):
    """Generate the specified number of mutations"""
    # TODO pass what mutation to do as argument
    i = start_num
    mutations = []
    while i < start_num + count:
        try:
            mutation = cityscapes_mutator.apply_mutation(mutation_type, arg_dict)
            if mutation is not None:
                mutations.append(mutation)
                mutation.update_file_names(folder)
                mutation.save_images()
            else:
                i -= 1  # don't advance if we didn't get a new mutation
        except Exception as e:
            traceback.print_exc(e)
            pass
        i += 1
    return 0


def create_fuzz_images(mutation_folder: MutationFolder, count, mutation_type: MutationType, arg_dict, pool_count=10):
    """Generate mutated images using a thread pool for increased speed"""
    count_per = int(count / pool_count)
    results = []
    orig_count = count
    if pool_count == 1:
        create_images(mutation_folder, count, 0, mutation_type, arg_dict)
    else:
        with Pool(pool_count) as pool:
            while count > 0:
                res = pool.apply_async(create_images, (mutation_folder, min(count, count_per), orig_count - count,
                                                       mutation_type, arg_dict))
                results.append(res)
                count -= count_per
            for res in results:  # wait for all images to generate
                res.get()


def save_paletted_image(old_file_path, new_file_path):
    Image(image=cv2.imread(old_file_path), image_file=new_file_path).save_paletted_image()


class Tester:
    def __init__(self):
        self.sut_list = [
            NVIDIASemSeg('/home/adwiii/git/nvidia/semantic-segmentation'),
            NVIDIASDCNet('/home/adwiii/git/nvidia/sdcnet/semantic-segmentation',
                         '/home/adwiii/git/nvidia/large_assets/sdcnet_weights/cityscapes_best.pth'),
            DecoupleSegNet('/home/adwiii/git/DecoupleSegNets'),
            EfficientPS('/home/adwiii/git/EfficientPS'),
            HRNet('/home/adwiii/git/HRNet-Semantic-Segmentation')
        ]
        self.sut_manager = SUTManager(self.sut_list)

    def execute_tests(self, mutation_folder: MutationFolder, mutation_type: MutationType, arg_dict,
                      num_tests=600, pool_count=30, compute_metrics=True):
        start_time = time.time()
        create_fuzz_images(mutation_folder, num_tests, pool_count=pool_count,
                           mutation_type=mutation_type, arg_dict=arg_dict)
        end_time = time.time()
        total_time = end_time - start_time
        time_per = total_time / num_tests
        print('Generated %d mutations in %0.2f s (%0.2f s/im, ~%0.2f cpus/im)' % (num_tests, total_time,
                                                                                  time_per, time_per * pool_count))
        # TODO add discriminator here or move it into the create_fuzz_images call
        self.sut_manager.run_suts(mutation_folder)
        if compute_metrics:
            self.compute_cityscapes_metrics(mutation_folder)

    def compute_cityscapes_metrics(self, mutation_folder: MutationFolder,
                                   exclude_high_dnc=False, quiet=False, pool_count=28):
        cityscapes_eval_args = copy.copy(orig_cityscapes_eval_args)
        cityscapes_eval_args.evalPixelAccuracy = True
        cityscapes_eval_args.quiet = quiet
        results = {}
        # black_pixel = [0, 0, 0]
        # dnc_count = []
        with Pool(pool_count) as pool:
            for sut in self.sut_list:
                print('--- Evaluating %s ---' % sut.name)
                pred_img_list = []
                gt_img_list = []
                skip_count = 0
                folder = mutation_folder.get_sut_folder(sut.name)
                for file in glob.glob(folder + '*edit_prediction.png'):
                    file_name = file[file.rfind('/') + 1:]
                    short_file = file_name[file_name.rfind('/') + 1:]
                    base_img = short_file[:-20]  # remove the _edit_prediction.png part
                    mutation_gt_file = mutation_folder.mutations_gt_folder +\
                                       base_img + '_mutation_gt.png'
                    if exclude_high_dnc and (base_img in HIGH_DNC or (len(base_img) > 52 and base_img[37:] in HIGH_DNC)):
                        skip_count += 1
                        continue
                    # else:
                    #     gt_im = cv2.imread(mutation_gt_file)
                    #     dnc = np.count_nonzero(np.all(gt_im == black_pixel,axis=2))
                    #     if dnc > 200000:
                    #         skip_files.append(mutation_gt_file)
                    #         continue
                    #     # dnc_count.append(dnc)
                    pred_img_list.append(file)
                    gt_img_list.append(mutation_gt_file)
                print('Skipped %d, kept %d' % (skip_count, len(pred_img_list)))
                # exit()
                results[sut.name] = pool.apply_async(evaluateImgLists,
                                                     (pred_img_list, gt_img_list, cityscapes_eval_args))
            for sut in self.sut_list:
                results[sut.name] = results[sut.name].get()
        with open(mutation_folder.base_folder + 'results.txt', 'w') as f:
            out = 'Exclude high DNC: ' + str(exclude_high_dnc)
            print(out)
            f.write(out + '\n')
            for sut in self.sut_list:
                out = '%s: %0.4f' % (sut.name, 100*results[sut.name]['averageScoreClasses'])
                print(out)
                f.write(out + '\n')
            f.write('All Results:\n')
            for sut in self.sut_list:
                f.write('SUT: %s\n' % sut.name)
                f.write(str(results[sut.name]) + '\n\n')
        return results

    def run_on_cityscapes_benchmark(self, pool_count=28):
        mutation_folder = MutationFolder(CITYSCAPES_DATA_ROOT + '/sut_gt_testing')
        # for camera_image in glob.glob(CITYSCAPES_DATA_ROOT + "/gtFine_trainvaltest/gtFine/leftImg8bit/train/**/*_leftImg8bit.png", recursive=True):
        #     short_file = camera_image[camera_image.rfind('/') + 1:-len('_leftImg8bit.png')]
        #     copyfile(camera_image, mutation_folder.folder + short_file + '_edit.png')
        # results = []
        # with Pool(pool_count) as pool:
        #     for gt_image in glob.glob(CITYSCAPES_DATA_ROOT + "/gtFine_trainvaltest/gtFine/train/**/*_gtFine_color.png", recursive=True):
        #         short_file = gt_image[gt_image.rfind('/') + 1:-len('_gtFine_color.png')]
        #         new_file = mutation_folder.mutations_gt_folder + short_file + '_mutation_gt.png'
        #         results.append(pool.apply_async(save_paletted_image, (gt_image, new_file)))
        #     for res in results:
        #         res.wait()
        # self.sut_manager.run_suts(mutation_folder)
        self.compute_cityscapes_metrics(mutation_folder)

    def visualize_diffs(self, sut_diffs):
        bin_count = max([len(sut_diffs[sut.name].values()) for sut in self.sut_list]) // 10
        other_bins = None
        for sut in self.sut_list:
            diffs = sut_diffs[sut.name]
            if other_bins is None:
                _, other_bins, _ = plt.hist(diffs.values(), bins=bin_count, alpha=0.5, label=sut.name,
                                         histtype='step')
            else:
                plt.hist(diffs.values(), bins=other_bins, alpha=0.5, label=sut.name,
                         histtype='step')
        plt.xlabel("SUT Differences")
        plt.ylabel("Count")
        plt.title("SUT Differences")
        plt.legend(loc='upper right')
        plt.show()

    def compute_differences(self, truth, predicted, ignore_black=False):
        # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
        truth_gray = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
        predicted_gray = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(truth_gray, predicted_gray, full=True)
        diff = (diff * 255).astype("uint8")
        diff = np.stack((diff,) * 3, axis=-1)
        # diff_image = np.copy(predicted)
        diff_image = cv2.bitwise_xor(truth, predicted)
        diff_image[np.where((diff_image!=[0,0,0]).any(axis=2))] = [255, 255, 255]  # convert not black to white
        if ignore_black:
            diff_image[np.where((truth == [0, 0, 0]).any(axis=2))] = [0, 0, 0]  # convert black in truth to black in diff since we are ignoring
        num_pixels = np.count_nonzero(np.where((diff_image!=[0,0,0]).any(axis=2)))
        diff_image_pred = cv2.add(predicted, diff_image)
        return score, num_pixels, diff_image, diff_image_pred

    def draw_text(self, image, text_list, start_x=10, start_y=20):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.5
        fontColor = (255, 0, 255)
        lineType = 3
        for index, text in enumerate(text_list):
            cv2.putText(image,
                        text,
                        (start_x, start_y + 60 * (index + 1)),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    def visualize_folder(self, mutation_folder):
        orig_file_suffix = '_orig.png'
        orig_predicted_file_suffix = '_orig_prediction.png'
        edit_file_suffix = '_edit.png'
        edit_predicted_file_suffix = '_edit_prediction.png'
        differences = []
        for key, mutation in mutation_folder.mutation_map.items():
            base_file_name = key
            base_file = mutation_folder.folder + base_file_name
            mutation.update_orig_prediction(Image(image_file=base_file + orig_predicted_file_suffix, read_on_load=True))
            mutation.edit_prediction = Image(image_file=base_file + edit_predicted_file_suffix, read_on_load=True)
            top = cv2.hconcat([mutation.orig_image.image, mutation.edit_image.image])
            mid1, _ = self.diff_image_pair(base_file_name, mutation.orig_image.image, mutation.edit_image.image)
            mid = cv2.hconcat([mutation.orig_prediction_mutated.image, mutation.edit_prediction.image])
            bottom, diff_percent = self.diff_image_pair(base_file_name, mutation.orig_prediction_mutated.image, mutation.edit_prediction.image)
            # total = cv2.vconcat([top, mid1, mid, bottom])
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid1, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(base_file + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    def visualize_plain_folder(self, folder):
        orig_file_suffix = '_orig.png'
        orig_predicted_file_suffix = '_orig_prediction.png'
        edit_file_suffix = '_edit.png'
        edit_predicted_file_suffix = '_edit_prediction.png'
        differences = []
        _, _, filenames = next(os.walk(folder))
        for file in filenames:
            base_file_name : str = file[:file.rfind('.png')]
            if not base_file_name.endswith('_orig_prediction'):
                continue
            base_file_name: str = file[:file.rfind('_orig_prediction.png')]
            edit_prediction_loc = folder + '/' + base_file_name + '_prediction.png'
            orig_prediction_loc = folder + '/' + base_file_name + '_orig_prediction.png'
            orig_image_loc = folder + '/' + base_file_name + '.png'
            print(edit_prediction_loc)
            print(orig_prediction_loc)
            print(orig_image_loc)
            edit_prediction = cv2.imread(edit_prediction_loc)
            orig_prediction = cv2.imread(orig_prediction_loc)
            orig_image = cv2.imread(orig_image_loc)
            top = cv2.hconcat([orig_image, orig_image])
            # mid1, _ = self.diff_image_pair(base_file_name, orig_image, orig_image)
            mid = cv2.hconcat([orig_prediction, edit_prediction])
            bottom, diff_percent = self.diff_image_pair(base_file_name, orig_prediction, edit_prediction, ignore_black=True)
        #     # total = cv2.vconcat([top, mid1, mid, bottom])
            total = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([mid, bottom])])
            differences.append((diff_percent, base_file_name))
            cv2.imwrite(folder + '/' + base_file_name + '_total_%0.6f.png' % diff_percent, total)
        for diff, img in sorted(differences):
            print('diff: %f, img: %s' % (diff, img))

    def diff_image_pair(self, base_file_name, orig_im, edit_im, ignore_black=False):
        score, num_pixels, blank_diff, diff = self.compute_differences(orig_im, edit_im, ignore_black)
        diff_percent = 100 * float(num_pixels) / (orig_im.shape[0] * orig_im.shape[1])
        # txt_image = np.zeros((orig_im.shape[0], orig_im.shape[1], 3), np.uint8)
        self.draw_text(blank_diff, [
            base_file_name,
            '# pixels diff: %d (%f%%)' % (num_pixels, diff_percent),
            'Diff Score (-1 to 1): %f' % score
        ])
        mid1 = cv2.hconcat([blank_diff, diff])
        return mid1, diff_percent

# TODO move the below functions into the class above
DATA_ROOT = '/home/adwiii/data/sets/nuimages'
nuim = NuImages(dataroot=DATA_ROOT, version='v1.0-mini', verbose=False, lazy=True)
nuim_mutator = NuScenesMutator(DATA_ROOT, 'v1.0-mini')


def print_distances(polys: List[SemanticPolygon]):
    i = 0
    while i < len(polys) - 1:
        j = i + 1
        while j < len(polys):
            min_dist = polys[i].min_distance(polys[j])
            print(min_dist)
            j += 1
        i += 1


def plot_hist_as_line(data, label, bin_count=None, bins=None):
    # https://stackoverflow.com/questions/27872723/is-there-a-clean-way-to-generate-a-line-histogram-chart-in-python
    n, calced_bins, _ = plt.hist(data, bins=bins if bins is not None else bin_count, histtype='bar', alpha=1, label=label, stacked=True)
    # bin_centers = 0.5 * (calced_bins[1:] + calced_bins[:-1])
    # plt.plot(bin_centers, n, label=label)  ## using bin_centers rather than edges
    return n, calced_bins

def get_base_file(long_file: str):
    long_file = long_file[long_file.rfind('/')+1:]
    if '-' in long_file:
        # strip uuid
        long_file = long_file[37:]
    long_file = long_file.replace('_edit_prediction.png', '')
    return long_file

def get_score(image):
    return 100.0 * (1.0 - image[1]['nbCorrectPixels'] / image[1]['nbNotIgnoredPixels'])

KEY_CLASSES = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'person', 'rider']
if __name__ == '__main__':
    tester = Tester()
    # mutation_folder = MutationFolder('/home/adwiii/git/perception_fuzzing/src/images/new_mutation_gt')
    # tester.compute_cityscapes_metrics(mutation_folder)
    base_folder = '/home/adwiii/git/perception_fuzzing/src/images/fri_'
    folders = []
    for local_folder, mutation_type, arg_dict in [
        ('person_color', MutationType.CHANGE_COLOR, {'semantic_label': 'person'}),
        ('car_color', MutationType.CHANGE_COLOR, {'semantic_label': 'car'}),
        ('add_person', MutationType.ADD_OBJECT, {'semantic_label': 'person'}),
        ('add_car', MutationType.ADD_OBJECT, {'semantic_label': 'car'}),
    ]:
        folder = base_folder + local_folder
        mutation_folder = MutationFolder(folder)
        # tester.execute_tests(mutation_folder, mutation_type=mutation_type, arg_dict=arg_dict, compute_metrics=False)
        folders.append(mutation_folder)
    folders.insert(0, MutationFolder(CITYSCAPES_DATA_ROOT + '/sut_gt_testing'))
    worst_images = {}
    worst_images_drop = {}
    worst_count = 5
    worst_images_for_counts = []
    score_on_training = {}
    for index, mutation_folder in enumerate(folders):
        bins = np.linspace(0.5, 4.5, 20)
        running_total = 0
        results = tester.compute_cityscapes_metrics(mutation_folder, quiet=True)
        worst_images[mutation_folder.short_name] = {}
        worst_images_drop[mutation_folder.short_name] = {}
        for sut, result_dict in results.items():
            image_scores = result_dict["perImageScores"]
            data = [get_score(image) for image in image_scores.items()]
            if index == 0:
                score_on_training[sut] = {get_base_file(image[0]):
                                              get_score(image)
                                          for image in image_scores.items()}
            data_drop = [score_on_training[sut][get_base_file(image[0])] - get_score(image)
                         for image in image_scores.items() if (score_on_training[sut][get_base_file(image[0])] - get_score(image)) > 0.5]
            data_with_images_drop = sorted([(image[0], get_score(image),
                                        score_on_training[sut][get_base_file(image[0])],
                                             score_on_training[sut][get_base_file(image[0])] - get_score(image))
                                       for image in image_scores.items()],  # sort by drop from gt to us
                                      key=lambda x: (x[3], -x[1]), reverse=True)
            data_with_images = sorted(
                [(image[0], get_score(image),
                  score_on_training[sut][get_base_file(image[0])],
                  score_on_training[sut][get_base_file(image[0])] - get_score(image))
                 for image in image_scores.items()],  # sort by worst performance
                key=lambda x: x[1])
            worst_images[mutation_folder.short_name][sut] = data_with_images[:worst_count]
            worst_images_drop[mutation_folder.short_name][sut] = data_with_images_drop[:worst_count]
            worst_images_for_counts.extend([get_base_file(item[0]) for item in data_with_images[:worst_count]])
            n, bins = plot_hist_as_line(data_drop, sut, 40, bins)
            running_total += np.sum(n)
        plt.title('Hist of percentage point drop in %% pixel correct\nfor each image vs non-mutated img: %s\nTotal: %d' % (mutation_folder.short_name.replace('fri', ''), running_total))
        plt.xlabel('Percentage point drop in % pixels correct')
        plt.ylabel('Count of Images')
        plt.legend(loc='upper right')
        plt.show()
    print(worst_images)
    print(worst_images_drop)
    print(Counter(worst_images_for_counts))
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    print('Worst:')
    pp.pprint(worst_images)
    print('Overall Worst:')
    pp.pprint(Counter(worst_images_for_counts))
    pp.pprint({mutation: Counter(list(itertools.chain(*[[get_base_file(item[0]) for item in lst] for lst in suts.values()]))) for mutation, suts in worst_images.items()})
    print()
    print('Worst Drop:')
    pp.pprint(worst_images_drop)
    pp.pprint(
        {mutation: Counter(list(itertools.chain(*[[get_base_file(item[0]) for item in lst] for lst in suts.values()])))
         for mutation, suts in worst_images_drop.items()})

    exit()

    # tester.run_on_cityscapes_benchmark()

    # folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_mutations_10k/'
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_mutations_add_car/'
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_add_person/'
    folder = '/home/adwiii/git/perception_fuzzing/src/images/cityscapes_good_gt_change_person_color/'
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/add_car_check_perspective_rotated/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    diff_folder = folder[:-1] + '_differences/'
    if not os.path.exists(diff_folder):
        os.mkdir(diff_folder)
    # cityscapes_mutator.aggregate_images(folder)
    # semantic_seg_runner.run_semantic_seg(folder)

    mutation_folder = MutationFolder(folder)
    # create_fuzz_images(mutation_folder, 100, pool_count=1)
    #
    # exit()
    # semantic_seg_runner.run_semantic_seg(folder)
    # exit()

    issue_count = defaultdict(lambda:0)
    issue_count_files = defaultdict(lambda:[])
    steering_diffs = []
    # for file in glob.glob('/home/adwiii/git/perception_fuzzing/src/images/cityscapes_gt_predictions/*_prediction.png'):
    for file in glob.glob(folder + '*edit_prediction.png'):
        print(file)
        try:
            file_name = file[file.rfind('/')+1:]
            # if file_name not in ['1255b027-7596-4fcf-912f-8b6734780f86_erfurt_000100_000019_edit_prediction.png', 'ff1d4db2-0cb8-4996-ac32-b525bf9eeed8_erfurt_000100_000019_edit_prediction.png', '7669be9a-2120-46a1-8aea-0a469932ab2b_bremen_000312_000019_edit_prediction.png', '55203bc1-545a-482c-b466-50256afcc304_erfurt_000100_000019_edit_prediction.png', '4c5d6fbf-5978-4ffd-a8eb-db8f3eb79adf_cologne_000059_000019_edit_prediction.png', 'ee683248-339e-4035-bf10-5ca0d984a84b_dusseldorf_000167_000019_edit_prediction.png', '60ddc2ca-549e-4538-8aad-5d0025c7e0e3_dusseldorf_000165_000019_edit_prediction.png', '6cddbf73-8c56-48fe-af9b-1a29eb8700ba_cologne_000059_000019_edit_prediction.png', '6dff1e2b-2022-409e-8827-23c635c8446a_dusseldorf_000167_000019_edit_prediction.png', 'f9887a77-1c34-478b-9e98-d1f4e2f37b7a_erfurt_000100_000019_edit_prediction.png', '4ef90064-cc16-4ab6-8952-cd540f818d12_cologne_000059_000019_edit_prediction.png']:
            #     continue
            # print(file_name)
            # if file_name not in ['7d5e1d74-2613-4122-8cc7-dc9683728300_krefeld_000000_030111_edit_prediction.png', '80bad570-6d37-40f4-aad8-1f6876f1c4c0_aachen_000129_000019_edit_prediction.png'] and \
            #     file_name not in ['b7d1dd45-6a1c-4573-8ef4-33e4f75433d1_dusseldorf_000167_000019_edit_prediction.png', 'c3093d20-2414-4d0f-aba3-54ca8f578b35_dusseldorf_000167_000019_edit_prediction.png', '08d13095-bb02-4141-a031-4ecf87015721_ulm_000049_000019_edit_prediction.png', 'c31bc55e-a1e9-42f6-bec5-bf5b8ca36415_hanover_000000_056457_edit_prediction.png', '82a8c4eb-c47f-4246-9b6f-0a5e8b9d1370_ulm_000008_000019_edit_prediction.png', '623874c8-067d-417a-a9e0-ce263c6be19e_hanover_000000_007897_edit_prediction.png']:
            #     continue
            # if file_name not in ['623874c8-067d-417a-a9e0-ce263c6be19e_hanover_000000_007897_edit_prediction.png']:
            #     continue
            # if file_name not in ['29848911-9773-4a31-a29e-5f739bfa990b_darmstadt_000060_000019_edit_prediction.png']:
            #     continue
            orig_pred_image_file = file.replace('edit_prediction', 'orig_prediction')
            orig_pred_image = Image(image_file=orig_pred_image_file, read_on_load=True)
            short_file = file_name[file_name.rfind('/')+1:]
            orig_pred_mutation_file = mutation_folder.pred_mutations_folder + short_file.replace('edit_prediction', 'mutation_prediction')
            if os.path.exists(orig_pred_mutation_file):
                orig_pred_mutation_image = Image(image_file=orig_pred_mutation_file, read_on_load=True)
                pred_mutation_mask = np.copy(orig_pred_mutation_image.image)
                # convert to black and white
                pred_mutation_mask[np.where((pred_mutation_mask != [0,0,0]).any(axis=2))] = [255, 255, 255]
                pred_mutation_mask = cv2.cvtColor(pred_mutation_mask, cv2.COLOR_BGR2GRAY)
                _, pred_mutation_mask = cv2.threshold(pred_mutation_mask, 40, 255, cv2.THRESH_BINARY)
                inv_mask = cv2.bitwise_not(pred_mutation_mask)
                orig_pred_image.image = cv2.bitwise_and(orig_pred_image.image, orig_pred_image.image, mask=inv_mask)
                # orig_pred_mutated = cv2.bitwise_or(orig_pred_image.image, orig_pred_mutation_image.image, mask=pred_mutation_mask)
                orig_pred_image.image = cv2.add(orig_pred_image.image, orig_pred_mutation_image.image)
            edit_pred_image = Image(image_file=file, read_on_load=True)
            steering_angles = steering_models.evaluate([orig_pred_image.image, edit_pred_image.image])
            print(steering_angles)
            steering_diff = abs(steering_angles[0] - steering_angles[1]) * 180 / np.pi
            # if 10 > steering_diff > 5:
            #     cv2.imshow('orig', orig_pred_image.image)
            #     cv2.imshow('edit', edit_pred_image.image)
            #     cv2.waitKey()
            # exit()
            steering_diffs.append((steering_diff, steering_angles[0] * 180 / np.pi, steering_angles[1] * 180 / np.pi, file))
            # orig_pred_semantics = ImageSemantics(orig_pred_image, CityscapesMutator.COLOR_TO_ID, KEY_CLASSES)
            # edit_pred_semantics = ImageSemantics(edit_pred_image, CityscapesMutator.COLOR_TO_ID, KEY_CLASSES)
            # orig_pred_semantics.compute_semantic_polygons()
            # edit_pred_semantics.compute_semantic_polygons()
            # cityscapes_short_file = file_name[file_name.find('_')+1:file_name.find('_edit')]
            # exclusion_zones: List[CityscapesPolygon] = cityscapes_mutator.get_ego_vehicle(cityscapes_mutator.get_file(cityscapes_short_file))
            # sem_diffs = SemanticDifferences(orig_pred_semantics, edit_pred_semantics, [poly.polygon for poly in exclusion_zones])
            # sem_diffs.calculate_semantic_difference()
            # issues = 0
            # for key in KEY_CLASSES:
            #     total = len(sem_diffs.only_in_orig[key]) + len(sem_diffs.only_in_edit[key])
            #     if total > 0:
            #         edit_pred_image_copy = Image(image=np.copy(edit_pred_image.image), image_file=diff_folder + file_name)
            #         orig_pred_image_copy = Image(image=np.copy(orig_pred_image.image), image_file=diff_folder + file_name.replace('edit', 'orig'))
            #         print('Found differences for %s in image %s' % (key, file_name))
            #         print('Only in gt:')
            #         for orig in sem_diffs.only_in_orig[key]:
            #             print(orig.center, orig.effective_area, len(orig.additions), orig.max_dim)
            #             orig_pred_image_copy.image = cv2.drawContours(orig_pred_image_copy.image, orig.get_inflated_polygon_list(), -1, (255, 255, 255))
            #             issues += 1
            #         print_distances(sem_diffs.only_in_orig[key])
            #         print('Only in predicted:')
            #         for edit in sem_diffs.only_in_edit[key]:
            #             print(edit.center, edit.effective_area, len(edit.additions), edit.max_dim)
            #             edit_pred_image_copy.image = cv2.drawContours(edit_pred_image_copy.image, edit.get_inflated_polygon_list(), -1, (255, 255, 255))
            #             issues += 1
            #         orig_pred_image_copy.save_image()
            #         edit_pred_image_copy.save_image()
            # # cv2.imshow('orig', orig_pred_semantics.image)
            # # cv2.imshow('edit', edit_pred_semantics.image)
            # # cv2.waitKey()
            # issue_count[issues] += 1
            # issue_count_files[issues].append(file_name)
        except:
            traceback.print_exc()
            exit()
            issue_count[-1] += 1
    for count, files in issue_count_files.items():
        if count == 0:
            continue
        print(count, files)
    print(issue_count)
    steering_diffs.sort(reverse=True)
    print('\n'.join([str(s) for s in steering_diffs]))
    plt.hist([item[0] for item in steering_diffs])
    plt.xlabel('Difference in steering angles (deg)')
    plt.ylabel('Count')
    title = folder[:-1]
    title = title[title.rfind('/'):]
    plt.title(title)
    plt.show()



# def check_differences_polygons_cityscapes():
#     cityscapes_color_to_id = {(0, 0, 0): 'static', (0, 74, 111): 'dynamic', (81, 0, 81): 'ground', (128, 64, 128): 'road', (232, 35, 244): 'sidewalk', (160, 170, 250): 'parking', (140, 150, 230): 'rail track', (70, 70, 70): 'building', (156, 102, 102): 'wall', (153, 153, 190): 'fence', (180, 165, 180): 'guard rail', (100, 100, 150): 'bridge', (90, 120, 150): 'tunnel', (153, 153, 153): 'polegroup', (30, 170, 250): 'traffic light', (0, 220, 220): 'traffic sign', (35, 142, 107): 'vegetation', (152, 251, 152): 'terrain', (180, 130, 70): 'sky', (60, 20, 220): 'person', (0, 0, 255): 'rider', (70, 0, 0): 'truck', (100, 60, 0): 'bus', (90, 0, 0): 'caravan', (110, 0, 0): 'trailer', (100, 80, 0): 'train', (230, 0, 0): 'motorcycle', (32, 11, 119): 'bicycle'}
#     for i in range(1, 2001):
#         try:
#             which = str(i)  #  TODO 19, resume at 61
#             orig_image = Image(image_file='/home/adwiii/git/perception_fuzzing/src/images/test_imgs/%s_orig_prediction.png' % which)
#             orig_image.load_image()
#             orig_semantics = ImageSemantics(orig_image, cityscapes_color_to_id)
#             orig_semantics.compute_semantic_polygons()
#             edit_image = Image(image_file='/home/adwiii/git/perception_fuzzing/src/images/test_imgs/%s_edit_prediction.png' % which)
#             edit_image.load_image()
#             edit_semantics = ImageSemantics(edit_image, cityscapes_color_to_id)
#             edit_semantics.compute_semantic_polygons()
#
#             sem_diffs = SemanticDifferences(orig_semantics, edit_semantics)
#             sem_diffs.calculate_semantic_difference()
#             # for key in sem_diffs.all_keys:
#             for key in ['car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']:
#                 total = len(sem_diffs.only_in_orig[key]) + len(sem_diffs.only_in_edit[key])
#                 if total > 0:
#                     print('Found differences for %s in image %s' % (key, which))
#                     print('Only in orig:')
#                     for orig in sem_diffs.only_in_orig[key]:
#                         print(orig.center, orig.effective_area)
#                     print('Only in edit:')
#                     for edit in sem_diffs.only_in_edit[key]:
#                         print(edit.center, edit.effective_area)
#                 # for pair in sem_diffs.matching_pairs[key]:
#                 #     print('Center Distance:', pair.get_center_distance())
#                 #     print('Area Difference:', pair.get_area_difference())
#         except:
#             pass


    # for sem_id in orig_semantics.semantic_maps.keys():
    #     print(sem_id, len(orig_semantics.semantic_maps[sem_id]), len(edit_semantics.semantic_maps[sem_id]))

    # folder = '/home/adwiii/git/SIMS/result/synthesis/transform_order_512'
    # semantic_seg_runner.run_semantic_seg(folder)
    # folder = '/home/adwiii/git/perception_fuzzing/src/images/test_imgs4/'
    # if not os.path.exists(folder):
    #     os.mkdir(folder)
    # mutation_folder = MutationFolder(folder)
    # create_fuzz_images(mutation_folder, 200, pool_count=1)
    # print('Generated images, running semantic seg')
    # semantic_seg_runner.run_semantic_seg(mutation_folder.folder)
    # print('Run Semantic Seg, running analytics')
    # tester = Tester()
    # tester.visualize_plain_folder(folder)

