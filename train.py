from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
import random
import cv2
import json
from detectron2.data import DatasetCatalog
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


dataset_path = os.path.join(
    os.environ["DATA_PATH"], "trafo_img", "kendi_topladiklarimiz"
)


DatasetCatalog.register("trafo_train", lambda: dataset_fn(dataset_path))

DatasetCatalog.register("trafo_test", lambda: dataset_fn(dataset_path))
MetadataCatalog.get("trafo_train").set(thing_classes=["trafo"])
MetadataCatalog.get("trafo_test").set(thing_classes=["trafo"])

cfg.DATASETS.TRAIN = ("trafo_train",)
cfg.DATASETS.TEST = ()


try:
    os.makedirs("dataset_vis", exist_ok=True)
except:
    print("Populating dataset_vis")
    for n, d in enumerate(DatasetCatalog.get("trafo_train")):
        im = cv2.imread(d["file_name"])
        v = Visualizer(im, metadata=MetadataCatalog.get("trafo_train"), scale=0.5)
        out = v.draw_dataset_dict(d)
        cv2.imwrite(os.path.join("dataset_vis", str(n) + ".jpg"), out.get_image())
        cv2.waitKey()


cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
)

cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

with open("my_config.yaml", "w") as my_cfg:
    my_cfg.write(cfg.dump())

trainer.train()
