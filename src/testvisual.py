from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from dataset_fn import dataset_fn
import tqdm

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

import argparse
parser = argparse.ArgumentParser()
   
parser.add_argument('--test_video_path', type=str)
parser.add_argument('--results_save_path', type=str)

args = parser.parse_args()


setup_logger()

cfg = get_cfg()
cfg.merge_from_file("my_config.yaml")

# path to the model ust trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
predictor = DefaultPredictor(cfg)

save_path = args.results_save_path
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'imgs'), exist_ok=True)

    



#read frames from video
cap = cv2.VideoCapture(args.test_video_path)
framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
images = []

n = 0
while cap.isOpened():
    ret, im = cap.read()
    if not ret:
        break
    if n % framespersecond == 0:
        images.append(im)
    n += 1
cap.release()

#read gps locations from video .srt file
with open('testdata/testvid.srt') as file:
    lines = file.readlines()
gpslines = []
for line in lines:
    if line.startswith('GPS'):
        gpslines.append(line[4:19])

coords = open(os.path.join(save_path, 'detected_coordinates.txt'), 'w')

for n, im in enumerate(tqdm.tqdm(images)):
    outputs = predictor(im)
    if len(outputs["instances"]) > 0:
        try:
            coords.write(gpslines[n])
            coords.write('\n')
        except:
            "not written coord"
        v = Visualizer(im, scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(save_path, 'imgs', str(n) + ".jpg"), out.get_image())

coords.close()


