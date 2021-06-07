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

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

import argparse
parser = argparse.ArgumentParser()
   
parser.add_argument('--test_dataset_path', type=str)
parser.add_argument('--results_save_path', type=str)

args = parser.parse_args()


setup_logger()

cfg = get_cfg()
cfg.merge_from_file("my_config.yaml")

DatasetCatalog.register("trafo_test", lambda: dataset_fn(args.test_dataset_path))

MetadataCatalog.get("trafo_test").set(thing_classes=["trafo"])

cfg.DATASETS.TEST = ("trafo_test",)

# path to the model ust trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
predictor = DefaultPredictor(cfg)

save_path = args.results_save_path
os.makedirs(save_path, exist_ok=True)

for n, d in enumerate(DatasetCatalog.get("trafo_test")):
    print(f"{n}")
    im = cv2.imread(d["file_name"])
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    v = Visualizer(im, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join(save_path, str(n) + ".jpg"), out.get_image())

evaluator = COCOEvaluator("trafo_test", cfg, distributed=False, output_dir="./output/")

test_loader = build_detection_test_loader(cfg, "trafo_test")
inference_on_dataset(predictor.model, test_loader, evaluator)
