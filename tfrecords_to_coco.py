import sys
import io
import os
import argparse
import logging
import json
import tensorflow.compat.v1 as tf
import PIL.Image as Image
from tqdm import tqdm


parser = argparse.ArgumentParser(
                    description='TF Records converter to coco fromat.')
parser.add_argument('--tfrecords', type=str, nargs='+',
                    help='path to TF record(s) to view')

parser.add_argument('--output_path', type=str, default='output.json',
                    help='path to save json')

parser.add_argument('--image-key', type=str, default="image/encoded",
                    help='Key to the encoded image.')

parser.add_argument('--filename-key', type=str, default="image/filename",
                    help='Key to the unique ID of each record.')


#######################################
# Object detection specific arguments #
parser.add_argument('--bbox-name-key', type=str,
                    default="image/object/class/text",
                    help='Key to the bbox label.')

parser.add_argument('--bbox-xmin-key', type=str,
                    default="image/object/bbox/xmin")
parser.add_argument('--bbox-xmax-key', type=str,
                    default="image/object/bbox/xmax")
parser.add_argument('--bbox-ymin-key', type=str,
                    default="image/object/bbox/ymin")
parser.add_argument('--bbox-ymax-key', type=str,
                    default="image/object/bbox/ymax")

parser.add_argument('--coordinates-in-pixels', action="store_true",
                    help='Set if bounding box coordinates are \
                     saved in pixels, not in %% of image width/height.')

parser.add_argument('--labels', type=str, default="car;bird",
                    help='Labels for which bounding boxes \
                    should be written to output json.')

###########################################
# Image classification specific arguments #
parser.add_argument('--class-label-key', type=str,
                    default="image/class/text",
                    help='Key to the image class label.')


def get_categories(args):
    logger.info('Setup categories...')
    labels_to_highlight = list(args.labels.split(';'))
    categories = []
    start_id = 1000
    for label in labels_to_highlight:
        category_dict = {
            "supercategory": label,
            "id": start_id,
            "name": label}
        categories.append(category_dict)
        start_id += 1
    logger.info('Done categories')
    return categories


def get_mapper_from_label_to_category_id(categories):
    mapper = {}
    for record in categories:
        label_ = record['name']
        id_ = record['id']
        mapper[label_] = id_
    return mapper


def bboxes_to_pixels(bbox, im_width, im_height):
    """
    Convert bounding box coordinates to pixels.
    (It is common that bboxes are parametrized as percentage of image size
    instead of pixels.)

    Args:
      bboxes (tuple): (xmin, xmax, ymin, ymax)
      im_width (int): image width in pixels
      im_height (int): image height in pixels

    Returns:
      bboxes (tuple): (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = bbox
    return xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height


def box_corner_to_coco_format(x_min, y_min, x_max, y_max, im_width, im_height):
    """
    According to this: https://github.com/cocodataset/cocoapi/issues/102
    Convert from (x_min, y_min, x_max, y_max)
    to (x_min, y_min, width, height)
    """
    bbox_coords = x_min, x_max, y_min, y_max
    x_min, x_max, y_min, y_max = bboxes_to_pixels(bbox_coords, im_width, im_height)
    w = x_max - x_min
    h = y_max - y_min
    return x_min, y_min, w, h


def get_bbox_tuples_and_labels(args, feature, im_width, im_height):
    """
    args: Namespace with arguments from argparser
    feature: features from tensorflow tfrecords
    returns bboxes: List[label: str], List[tuple(x_min, y_min, x_max, y_max)]
    """
    labels, bboxes = [], []
    if args.bbox_name_key in feature:
        for ibbox, label in enumerate(
                            feature[args.bbox_name_key].bytes_list.value):
            bbox = box_corner_to_coco_format(
                        feature[args.bbox_xmin_key].float_list.value[ibbox],
                        feature[args.bbox_ymin_key].float_list.value[ibbox],
                        feature[args.bbox_xmax_key].float_list.value[ibbox],
                        feature[args.bbox_ymax_key].float_list.value[ibbox],
                        im_width, im_height)
            bboxes.append(bbox)
            labels.append(label.decode("utf-8"))
    else:
        print("Bounding box key '%s' not present." % (args.bbox_name_key))
    return labels, bboxes


def process_data(args, map_label_to_categoty_id):
    """
    Load all images annotation
    returns dict (will be json)
    """
    logger.info('Start process tfrecords')
    images, annotations = [], []
    img_id, annotation_id = 1, 1
    for tfrecord_path in tqdm(args.tfrecords, position=0, leave=True):
        for i, record in enumerate(
                         tf.python_io.tf_record_iterator(tfrecord_path)):
            example = tf.train.Example()
            example.ParseFromString(record)
            feat = example.features.feature
            img = feat[args.image_key].bytes_list.value[0]
            w, h = Image.open(io.BytesIO(img)).size
            filename = feat[args.filename_key].bytes_list.value[0].decode("utf-8")

            labels, bboxes = get_bbox_tuples_and_labels(args, feat, w, h)
            img_dict = {"id": img_id,
                        "width": w,
                        "height": h,
                        "file_name": filename}
            images.append(img_dict)
            for label, bbox_info in zip(labels, bboxes):
                annotation = {"id": annotation_id,
                              "image_id": img_id,
                              "category_id": map_label_to_categoty_id[label],
                              "segmentation": [[]],
                              "area": bbox_info[2] * bbox_info[3],
                              "bbox": list(bbox_info),
                              "iscrowd": 0}
                annotations.append(annotation)
                annotation_id += 1
            img_id += 1
    logger.info('Finish!')
    return images, annotations


def create_coco_json(images, annotations, categories):
    data_json = {}
    data_json['images'] = images
    data_json['annotations'] = annotations
    data_json['categories'] = categories
    return data_json


def save_json(json_data, output_path):
    dir_path = os.path.dirname(output_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(output_path, 'w') as fp:
        json.dump(json_data, fp, indent=4)
    logging.info(f'Json was saved. File path: {output_path}')


class NotTFDepricatedMessage(logging.Filter):
    def filter(self, record):
        return not ('deprecated' in record.getMessage()
                    and
                    'tensorflow' in record.getMessage())


def logging_setup():
    formatter = logging.Formatter(
                '%(asctime)s | from: %(name)s  [%(levelname)s]: %(message)s')
    logger = logging.getLogger(__name__)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)  # include all kind of messages
    tf_logger = tf.get_logger()
    tf_logger.addFilter(NotTFDepricatedMessage())  # not pass tf depricated messages
    return logger


logger = logging_setup()  # create logger


def main():
    args = parser.parse_args()
    categories = get_categories(args)
    maper_label_to_categoty_id = get_mapper_from_label_to_category_id(categories)
    images, annotations = process_data(args, maper_label_to_categoty_id)
    coco_json_data = create_coco_json(images, annotations, categories)
    save_json(coco_json_data, args.output_path)


if __name__ == '__main__':
    main()
