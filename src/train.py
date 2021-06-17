from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from dataset_fn import dataset_fn

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
)

import argparse
parser = argparse.ArgumentParser()
   
parser.add_argument('--train_dataset_path', type=str, required=True)

args = parser.parse_args()


train_dataset_path = args.train_dataset_path

#register datasets
DatasetCatalog.register("trafo_train", lambda: dataset_fn(train_dataset_path))

MetadataCatalog.get("trafo_train").set(thing_classes=["trafo"])

cfg.DATASETS.TRAIN = ("trafo_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False


#load pretrained model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
)

#Faster RCNN config
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 20
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

with open("my_config.yaml", "w") as my_cfg:
    my_cfg.write(cfg.dump())

trainer.train()
