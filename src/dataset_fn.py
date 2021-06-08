import os
import json
from detectron2.structures import BoxMode


def dataset_fn(dataset_path):
    annotations_path = os.path.join(dataset_path, 'ann')
    images_path = os.path.join(dataset_path, 'img')

    output_list = []
    image_file_names = os.listdir(images_path)
    for image_file_name in image_file_names:
        image_file_path = os.path.join(images_path, image_file_name)
        ann_file_path = os.path.join(
            annotations_path, image_file_name + '.json')
        with open(ann_file_path) as json_file:
            annotation_dict = json.load(json_file)
            detectron_dict = {}
            detectron_dict['file_name'] = image_file_path
            detectron_dict['height'] = annotation_dict['size']['height']
            detectron_dict['width'] = annotation_dict['size']['width']
            detectron_dict['image_id'] = image_file_path

            annotations = []
            for object_ in annotation_dict['objects']:
                bbox = object_['points']['exterior']
                bbox = [*bbox[0], *bbox[1]]
                bbox_mode = BoxMode.XYXY_ABS
                category_id = 0
                annotations.append(
                    dict(
                        bbox=bbox,
                        bbox_mode=bbox_mode,
                        category_id=category_id
                    )
                )
            detectron_dict['annotations'] = annotations
        output_list.append(detectron_dict)
    return output_list
