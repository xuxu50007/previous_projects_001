# ----------------------------------
# Copyright (c)
# Modified by Yan
# Date: 2021
# ----------------------------------

from mmdet.apis import init_detector, inference_detector, show_result_ins
from generate_pkl import generate_pkl
import cv2
import mmcv
import sys
import os
import pickle
from zipfile import ZipFile, ZIP_BZIP2
import argparse

parser = argparse.ArgumentParser(description='Script to run SoloV2')
parser.add_argument("--config", help="config file name")
parser.add_argument("--model", help="pth file name")
parser.add_argument("--inputvideo", help="video file path")
parser.add_argument("--outputzip", help="output_zip")
parser.add_argument("--scoreth", help="score_th")
parser.add_argument("--onlyperson", help='keep only people', action='store_true')
args = parser.parse_args()

config_file = args.config
checkpoint_file = args.model
vid_file_path = args.inputvideo
output_zip = args.outputzip
score_th = float(args.scoreth)
only_person = False
if args.onlyperson:
    only_person = True

# config_name ='solov2/solov2_x101_dcn_fpn_8gpu_3x.py' #1
# pth_name = 'SOLOv2_X101_DCN_3x.pth' #2

# config_file = './configs/'+config_name
# checkpoint_file = './checkpoints/'+pth_name

# config_file = './configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
# checkpoint_file = './checkpoints/SOLOv2_X101_DCN_3x.pth'

zip = ZipFile(output_zip, 'w', compression=ZIP_BZIP2)
zip.close()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Load a video
# vid_file_path = './multi_people.mp4'
vidcap = cv2.VideoCapture(vid_file_path)
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Processing %d frames\n========================" % total_frames)
for frame_index in range(total_frames):
    ret, frame = vidcap.read()
    if ret == False:
        print("exit false")
        break
    print("processing -- %d frame" % frame_index)
    seg_result = inference_detector(model, frame)
    pkl_result = generate_pkl(seg_result, score_th, only_person)

    outname = '%s_%.5d.bin' % (output_zip, frame_index)
    arcname = 'mask_struct_%d.bin' % frame_index

    pickle.dump(pkl_result, open(outname, "wb"))
    zip = ZipFile(output_zip, 'a', compression=ZIP_BZIP2)
    zip.write(outname, arcname=arcname)
    zip.close()
    os.remove(outname)

vidcap.release()
# cv2.destroyAllWindows()

## Test a single image
# img = './demo.jpg'

# seg_result = inference_detector(model, img)
# print(seg_result)
# pkl_file_path = "./nuance_result.pkl"
# generate_pkl(seg_result, pkl_file_path)