# Sample usage:
# python models/research/object_detection/run_model.py  --model runs\1\model\frozen_inference_graph.pb --labels runs\1\data\costco_label_map.pbtxt --box-output temp\boxes --output temp\output temp\input

import argparse
from datetime import datetime
import json
import os
import re

from PIL import Image
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_graph(model_file):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  return detection_graph

def load_label_map_category_index(labels_file):
  with open(labels_file, 'r') as opened_file:
    label_contents = opened_file.read()
    num_classes = len(re.findall('id:', label_contents))

  label_map = label_map_util.load_labelmap(labels_file)
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  return category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Converts numpy-format box to PIL format box
def pil_box_from_np_box(np_box, im_width, im_height):
  (ymin, xmin, ymax, xmax) = np_box
  # For reference, box = (left, upper, right, lower)
  return (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)

def run_graph_on_images(image_paths, detection_graph, category_index, min_score, json_output_dir, image_dir=None):
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # Definite input and output Tensors for detection_graph
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      for image_num, image_path in enumerate(image_paths):
        file_name = os.path.basename(image_path)
        json_path = os.path.join(json_output_dir, file_name + '.json')

        # Skip if we already have a JSON output file
        if os.path.isfile(json_path):
          print('{}: Skipped {} at {}'.format(image_num, image_path, datetime.now()))
          continue

        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Run the model!
        (boxes, scores, classes, _) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Write images if we have a directory
        if image_dir is not None:
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np, # Modified as output
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          writeable_image = Image.fromarray(image_np, 'RGB')
          writeable_image.save(os.path.join(image_dir, file_name + '.jpg'))

        # Get the objects into a encodable JSON file
        image_objects = []
        (im_width, im_height) = image.size
        for i, score in enumerate(scores[0]):
          if score >= min_score:
            native_score = np.asscalar(score)
            np_box = [np.asscalar(coord) for coord in boxes[0][i]]
            pil_box = pil_box_from_np_box(np_box, im_width, im_height)
            image_objects.append({'score': native_score, 'box': pil_box})

        with open(json_path, 'w') as json_file:
          json_file.write(json.dumps(image_objects, indent=2))

        print('{}: Done {} at {}'.format(image_num, image_path, datetime.now()))


def get_images_in_dir(image_dir):
  files = os.listdir(image_dir)
  files = [file_name for file_name in files if '.jpg' in file_name or '.jpeg' in file_name]
  return [os.path.join(image_dir, file_name) for file_name in files]

def clamp(num, min_num, max_num):
  return min(max(num, min_num), max_num)

def crop_image_to_boxes(image_path, identified_objects, cropped_output_dir, buffer_ratio=0.10):
  file_name = os.path.basename(image_path)
  image = Image.open(image_path)
  (im_width, im_height) = image.size

  for object_info in identified_objects:
    (left, top, right, bottom) = object_info['box']

    # Add a bit to width/height depending on buffer ratio
    crop_width = right - left
    crop_height = bottom - top
    extra_width_per_side = crop_width * (buffer_ratio / 2)
    extra_height_per_side = crop_height * (buffer_ratio / 2)
    left = left - extra_width_per_side
    right = right + extra_width_per_side
    top = top - extra_height_per_side
    bottom = bottom + extra_height_per_side
    box = [
        clamp(left, 0, im_width - 1),
        clamp(top, 0, im_height - 1),
        clamp(right, 0, im_width - 1),
        clamp(bottom, 0, im_height - 1)
    ]
    box = [int(round(coord)) for coord in box]
    new_image = image.crop(box)
    (left, right, top, bottom) = box
    suffix = '{}-{}-{}-{}'.format(left, right, top, bottom)
    new_image.save(os.path.join(cropped_output_dir, '{}.{}.jpg'.format(file_name, suffix)))

def crop_images_to_boxes(image_paths, json_dir, cropped_output_dir, crop_threshold=0.96):
  for (image_num, image_path) in enumerate(image_paths):
    file_name = os.path.basename(image_path)
    json_path = os.path.join(json_dir, file_name + '.json')
    if not os.path.exists(json_path):
      print('{}: Skipped cropping {} at {}'.format(image_num, image_path, datetime.now()))
      continue

    with open(json_path, 'r') as json_file:
      identified_objects = json.loads(json_file.read())

    scores = [obj['score'] for obj in identified_objects]
    print(scores)
    identified_objects = [obj for obj in identified_objects if obj['score'] > crop_threshold]
    crop_image_to_boxes(image_path, identified_objects, cropped_output_dir)
    print('{}: Cropped {} at {}'.format(image_num, image_path, datetime.now()))

def main():
  parser = argparse.ArgumentParser(description='Runs a tensorflow model against a directory of images')
  parser.add_argument('--model', dest='model_file', required=True, help='Usually a frozen_inference_graph.pb')
  parser.add_argument('--labels', dest='labels_file', required=True, help='A label-map pbtxt')
  parser.add_argument('--output', dest='cropped_output_dir', required=True, help='Where to output cropped images')
  parser.add_argument('--json-output', dest='json_output_dir', required=True, help='Directory to put JSON outputs for each image')
  parser.add_argument('--box-output', dest='box_output_dir', default=None, help='Directory to put files with outline boxes')
  parser.add_argument('--min-score', dest='min_score', type=float, default=0.1, help='Default 0.1, Minimum score to record boxes over')
  parser.add_argument('--crop-only', dest='crop_only', action='store_true', help='Whether to only crop without running training')
  parser.add_argument('dir', help='Directory with images')

  args = parser.parse_args()

  image_paths = get_images_in_dir(args.dir)

  if not args.crop_only:
    detection_graph = load_graph(args.model_file)
    category_index = load_label_map_category_index(args.labels_file)
    run_graph_on_images(image_paths, detection_graph, category_index, args.min_score, args.json_output_dir, args.box_output_dir)

  crop_images_to_boxes(image_paths, args.json_output_dir, args.cropped_output_dir)

if __name__ == '__main__':
  main()
