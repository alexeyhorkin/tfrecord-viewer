import io
import os
import argparse
import logging
import json
import tensorflow.compat.v1 as tf
import utils
import PIL.Image as Image
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(
                    description='TF Records converter to coco fromat.')
parser.add_argument('--tfrecords', type=str, nargs='+',
                    help='path to TF record(s) to view')

parser.add_argument('--output_path', type=str, default='output.json',
                    help='path to save json')

parser.add_argument('--cat_start_id', type=int, default=0,
                    help='Start category id, if N categories correspond to \
                    cat_start_id, cat_start_id + 1, ... cat_start_id + N ids')

parser.add_argument('--split', action='store_true',
                    help='if specified, images and their annotation will be splitted on test and train')

utils.update_parser_to_object_detection_args(parser)


def get_categories(args):
    logger.info('Setup categories...')
    labels_to_highlight = list(args.labels.split(';'))
    categories = []
    start_id = args.cat_start_id
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
    filenames = {}
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
            if '/' in filename:
                filename = filename.replace('/', '-')
            if filename not in filenames:
                filenames[filename] = img_id
                labels, bboxes = get_bbox_tuples_and_labels(args, feat, w, h)
                if len(labels) == 0:
                    logger.info(f"NO LABELS in: {filename}")
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
                                "area": np.round(bbox_info[2] * bbox_info[3], 3),
                                "bbox": list(map(lambda x: np.round(x, 2), bbox_info)),
                                "iscrowd": 0}
                    annotations.append(annotation)
                    annotation_id += 1
                img_id += 1
    logger.info('Finish!')
    return images, annotations


def process_data_split_test_train(args, map_label_to_categoty_id, test_size=0.33):
    """
    Load all images annotation
    returns dict (will be json)
    """
    logger.info('Start process tfrecords')
    filenames = {}
    images_train, annotations_train = [], []
    images_test, annotations_test = [], []
    all_img_names = []
    for tfrecord_path in args.tfrecords:
        for i, record in enumerate(
                         tf.python_io.tf_record_iterator(tfrecord_path)):
            example = tf.train.Example()
            example.ParseFromString(record)
            feat = example.features.feature
            img = feat[args.image_key].bytes_list.value[0]
            w, h = Image.open(io.BytesIO(img)).size
            filename = feat[args.filename_key].bytes_list.value[0].decode("utf-8")
            if '/' in filename:
                filename = filename.replace('/', '-')
            if filename not in filenames:
                filenames[filename] = True
                all_img_names.append(filename)

    filenames = {}

    img_names_train, _ =  train_test_split(all_img_names, test_size=test_size, shuffle=False, random_state=42)

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
            if '/' in filename:
                filename = filename.replace('/', '-')
            if filename not in filenames:
                filenames[filename] = img_id
                labels, bboxes = get_bbox_tuples_and_labels(args, feat, w, h)
                if len(labels) == 0:
                    logger.info(f"NO LABELS in: {filename}")
                img_dict = {"id": img_id,
                            "width": w,
                            "height": h,
                            "file_name": filename}
                if filename in img_names_train:
                    images_train.append(img_dict)
                else:
                    images_test.append(img_dict)
                for label, bbox_info in zip(labels, bboxes):
                    annotation = {"id": annotation_id,
                                "image_id": img_id,
                                "category_id": map_label_to_categoty_id[label],
                                "segmentation": [[]],
                                "area": np.round(bbox_info[2] * bbox_info[3], 3),
                                "bbox": list(map(lambda x: np.round(x, 2), bbox_info)),
                                "iscrowd": 0}
                    if filename in img_names_train:
                        annotations_train.append(annotation)
                    else:
                        annotations_test.append(annotation)
                    annotation_id += 1
                img_id += 1
    logger.info('Finish!')
    return images_train, annotations_train, images_test, annotations_test 


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


logger = utils.logging_setup(__name__)  # create logger


def main():
    args = parser.parse_args()
    categories = get_categories(args)
    maper_label_to_categoty_id = get_mapper_from_label_to_category_id(categories)
    if not args.split:
        images, annotations = process_data(args, maper_label_to_categoty_id)
        coco_json_data = create_coco_json(images, annotations, categories)
        save_json(coco_json_data, args.output_path)

    else:
        images_train, annotations_train, images_test, annotations_test = process_data_split_test_train(args, maper_label_to_categoty_id)
        coco_json_train = create_coco_json(images_train, annotations_train, categories)
        coco_json_test = create_coco_json(images_test, annotations_test, categories)
        save_json(coco_json_train, os.path.join(args.output_path, 'train.json'))
        save_json(coco_json_test, os.path.join(args.output_path, 'test.json'))


if __name__ == '__main__':
    main()
