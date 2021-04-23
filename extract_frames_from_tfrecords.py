import os
import io
from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from utils import logging_setup
import PIL.Image as Image
import tensorflow.compat.v1 as tf
import utils


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    parser.add_argument('--tfrecords', type=str, nargs='+',
                        help='path to TF record(s) to extract images')
    args.add_argument('-o', '--output_path',
                      help='Required. Path to output folder.',
                      default='frames', type=str)
    args.add_argument('-n', '--num_workers',
                      help='CPU workes count',
                      default=4, type=int)

    utils.update_parser_to_object_detection_args(parser)

    return parser


def process_data(args, tfrecord_path):
    """
    Load all images in threcods, and save it on args.output_path
    """
    logger.info(f'Start process tfrecord {tfrecord_path}')
    path_to_save = args.output_path
    for i, record in enumerate(
                        tf.python_io.tf_record_iterator(tfrecord_path)):
        example = tf.train.Example()
        example.ParseFromString(record)
        feat = example.features.feature
        img = feat[args.image_key].bytes_list.value[0]
        img_ = Image.open(io.BytesIO(img))
        img_name = feat[args.filename_key].bytes_list.value[0].decode("utf-8")
        if '/' in img_name:
            img_name = img_name.replace('/', '-')
        pts = os.path.join(path_to_save, img_name)
        img_.save(pts)


def get_all_tfrecords_path_files(args):
    paths = []
    for path in args.tfrecords:
        paths.append(path)
    return paths


def main(args, parallel_func):
    if not os.path.exists(args.output_path):
        logger.info(f'Create folder to save images: {args.output_path}')
        os.makedirs(args.output_path)

    tfrecord_paths = get_all_tfrecords_path_files(args)
    parallel_func = partial(parallel_func, args)
    outputs_codes = []
    tasks = []
    with Pool(processes=args.num_workers) as pool:
        for path_to_tfrecord in tqdm(tfrecord_paths):
            tasks.append(pool.apply_async(parallel_func,
                         args=(path_to_tfrecord,),
                         error_callback=lambda e: logger.info(e)))
        for task in tasks:
            task.wait()
            outputs_codes.append(task.get())
        pool.close()
        pool.join()
    logger.info('FINISH')


logger = logging_setup(__name__)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args, process_data)
