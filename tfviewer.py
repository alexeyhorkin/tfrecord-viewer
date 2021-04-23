#!/usr/bin/env python3
import io
import os
import argparse

import PIL.Image as Image

import tensorflow.compat.v1 as tf
from flask import Flask, render_template, send_file

from overlays import overlay_factory
from utils import update_parser_to_object_detection_args

from tqdm import tqdm

app = Flask(__name__)

parser = argparse.ArgumentParser(description='TF Record viewer.')
parser.add_argument('tfrecords', type=str, nargs='+',
                    help='path to TF record(s) to view')

parser.add_argument('--max-images', type=int, default=200,
                    help='Max. number of images to load.')

parser.add_argument('--host', type=str, default="0.0.0.0",
                    help='host/IP to start the Flask server.')

parser.add_argument('--port', type=int, default=5000,
                    help='Port to start the Flask server.')

parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")

parser.add_argument('--overlay', type=str, default="detection",
                    help='Overlay to display. (detection/classification/none)')

parser.add_argument('--savepath', type=str, default='',
                    help='path to save images')

parser.add_argument('--imgnames', type=str, default='',
                    help='path to file with image names which need to save and visualize;')

update_parser_to_object_detection_args(parser)

args = parser.parse_args()


# Variables to be loaded with preload_images()
images = []
filenames = []
captions = []
bboxes = []


def save_images(path_to_save):
  print(f'Start image saving into {path_to_save} ...')
  if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
  for i, img in tqdm(enumerate(images)):
    img_ = Image.open(io.BytesIO(img))
    filename, _ = os.path.splitext(filenames[i])
    pts = os.path.join(path_to_save, filename + '.png')
    img_.save(pts)


def load_image_names(imgnames_path):
  with open(imgnames_path, 'r') as f:
    res = f.readlines()
    content = [x.strip() for x in res]
    return set(content)


def preload_images(max_images, imgnames):
  """
  Load images to be displayed in the browser gallery.

  Args:
    max_images (int): Maximum number of images to load.
  Returns:
    count (int): Number of images loaded.
  """
  count = 0
  overlay = overlay_factory.get_overlay(args.overlay, args)
  imgnames = load_image_names(args.imgnames) if args.imgnames else set()
  already_added_filenames = set()

  for tfrecord_path in args.tfrecords:
    print("Filename: ", tfrecord_path)
    for i, record in enumerate(tf.python_io.tf_record_iterator(tfrecord_path)):
      if args.verbose:
        print("######################### Record", i, "#########################")
      example = tf.train.Example()
      example.ParseFromString(record)
      feat = example.features.feature

      if len(images) < max_images:
        filename = feat[args.filename_key].bytes_list.value[0].decode("utf-8")
        if imgnames and filename not in imgnames:
          continue  # just skip file
        if filename in already_added_filenames:
          continue
        img = feat[args.image_key].bytes_list.value[0]

        if not args.disable_bboxes:
          img_with_overlay = overlay.apply_overlay(img, feat)
        else:
          img_with_overlay = img

        filenames.append(filename)
        already_added_filenames.add(filename)
        images.append(img_with_overlay)
        captions.append(tfrecord_path + ":" + filename)
      else:
        return count
      count += 1
  return count


@app.route('/')
def frontpage():
  html = ""
  for i, filename in enumerate(filenames):
    html += '<img data-u="image" src="image/%s" data-caption="%s" />\n' % (i, captions[i])
  return render_template('gallery.html', header='Tfrecords visualization', images=html)


@app.route('/image/<key>')
def get_image(key):
  """Get image by key (index) from images preloaded when starting the viewer by preload_images().
  """
  key = int(key)
  img = images[key]
  img_buffer = io.BytesIO(img)
  return send_file(img_buffer,
                   attachment_filename=str(key)+'.jpeg',
                   mimetype='image/jpg')


@app.after_request
def add_header(r):
  """
  Add headers to disable Caching,
  (So that images with the same index in different TFRecords are displayed correctly.)
  """
  r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
  r.headers["Pragma"] = "no-cache"
  r.headers["Expires"] = "0"
  r.headers['Cache-Control'] = 'public, max-age=0'
  return r


if __name__ == "__main__":
  print("Pre-loading up to %d examples.." % args.max_images)
  count = preload_images(args.max_images, args.imgnames)
  print("Loaded %d examples" % count)
  if args.savepath:
    save_images(args.savepath)
  app.run(host=args.host, port=args.port)
