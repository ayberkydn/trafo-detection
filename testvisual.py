from detectron2.data import build_detection_test_loader
import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger


setup_logger()

cfg = get_cfg()
cfg.merge_from_file("my_config.yaml")

test_dataset_path = "/home/ayb/Desktop/termalvid/grey/img"

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

save_path = "./sonuc_fotolar"
os.makedirs(save_path, exist_ok=True)


test_images = os.listdir(test_dataset_path)
test_images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

for n, img_path in enumerate(test_images):
    print(n)
    im = cv2.imread(os.path.join(test_dataset_path, img_path))
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    v = Visualizer(im, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(os.path.join(save_path, str(n) + ".jpg"), out.get_image())


