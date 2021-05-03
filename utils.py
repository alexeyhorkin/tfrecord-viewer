import sys
import logging
import tensorflow.compat.v1 as tf


class NotTFDepricatedMessage(logging.Filter):
    def filter(self, record):
        return not ('deprecated' in record.getMessage()
                    and
                    'tensorflow' in record.getMessage())


def update_parser_to_object_detection_args(parser):

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


def logging_setup(logger_name):
    formatter = logging.Formatter(
                '%(asctime)s | from: %(name)s  [%(levelname)s]: %(message)s')
    logger = logging.getLogger(logger_name)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)  # include all kind of messages
    tf_logger = tf.get_logger()
    tf_logger.addFilter(NotTFDepricatedMessage())  # not pass tf depricated messages
    return logger
