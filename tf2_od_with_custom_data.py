# ------------------------------
# import libraries
# ------------------------------
import os
import pathlib

# Clone the tensorflow models repository if it doesn't already exist
#if "models" in pathlib.Path.cwd().parts:
#  while "models" in pathlib.Path.cwd().parts:
#    os.chdir('..')
#elif not pathlib.Path('models').exists():
#  !git clone --depth 1 https://github.com/tensorflow/models

#!cd models/research/
#!protoc object_detection/protos/*.proto --python_out=.
#!cp object_detection/packages/tf2/setup.py .
#!python -m pip install .

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import random
import io
import imageio
import glob
from bs4 import BeautifulSoup
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

#from object_detection.utils import label_map_util
#from object_detection.utils import config_util
#from object_detection.utils import visualization_utils as viz_utils
from utils import visualization_utils as viz_utils
#from object_detection.utils import colab_utils
#from object_detection.builders import model_builder


# ------------------------------
# Utilities
# ------------------------------
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


# ------------------------------
# load and show f-16 data
# ------------------------------
def load_show_data(train_images_np):
  train_image_dir = '/home/tensorflow/models/research/object_detection/test_images/f-16/train/'
  train_image_filelist = os.listdir(train_image_dir)
  for i in range(1, len(train_image_filelist)+1):
    image_path = os.path.join(train_image_dir, str(i+20) + '.jpg')
    train_images_np.append(load_image_into_numpy_array(image_path))

  plt.rcParams['axes.grid'] = False
  plt.rcParams['xtick.labelsize'] = False
  plt.rcParams['ytick.labelsize'] = False
  plt.rcParams['xtick.top'] = False
  plt.rcParams['xtick.bottom'] = False
  plt.rcParams['ytick.left'] = False
  plt.rcParams['ytick.right'] = False
  plt.rcParams['figure.figsize'] = [14, 7]

  # show only 6 images by 2 x 3 subplot.
  for idx, train_image_np in enumerate(train_images_np[:6]):
    plt.subplot(2, 3, idx+1)
    plt.imshow(train_image_np)
  plt.show()
#  plt.savefig('train_images.png', bbox_inches='tight')


train_images_np = []
load_show_data(train_images_np)


# ------------------------------
# load annotation information
# ------------------------------
gt_boxes = []

# get file list of annotations
annotation_path = "/home/tensorflow/models/research/object_detection/test_images/f-16/annotations/*"
annotation_file_list = glob.glob(annotation_path)

for annotation_file in annotation_file_list:
  xml = open(annotation_file, "r", encoding="utf-8").read()

  soup = BeautifulSoup(xml, 'html.parser')

  for size in soup.find_all('size'):
     width = size.find('width').string
     height = size.find('height').string
     depth = size.find('depth').string

  for object in soup.find_all('object'):
     name = object.find('name').string # edited
     for bndbox in object.find_all('bndbox'):
       xmin = object.find('xmin').string
       ymin = object.find('ymin').string
       xmax = object.find('xmax').string
       ymax = object.find('ymax').string

     #print('width : ' + width)
     #print('height : ' + height)
     #print('depth : ' + depth)
     #
     #print('label : ' + str(name))
     #print('xmin : ' + str(xmin))
     #print('ymin : ' + str(ymin))
     #print('xmax : ' + str(xmax))
     #print('ymax : ' + str(ymax))

     gt_boxes.append(np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32))


# ------------------------------
# Prepare data for training
# ------------------------------
# 클래스를 하나만 예측할 것이기 때문에 클래스 ID를 1로 지정한다.
f16_class_id = 1
num_classes = 1

category_index = {f16_class_id: {'id': f16_class_id, 'name': 'f-16'}}

# class label을 one-hot으로 변환하고 모든 것을 tensor로 변환한다.
# 여기서 'label_id_offset'은 모든 클래스를 특정 수의 인덱스만큼 이동시킨다.
# 우리는 여기서 백그라운드가 아닌 클래스가 0 번째 인덱스에서 카운트를 시작하는 one-hot 레이블을 모델이 받도록 수행한다. 이것은 보통 학습 바이너리에서 자동으로 처리되지만 여기서 reproduce한다.
label_id_offset = 1
train_image_tensors = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []
for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
  train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
      train_image_np, dtype=tf.float32), axis=0))
  gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
  zero_indexed_groundtruth_classes = tf.convert_to_tensor(
      np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
  gt_classes_one_hot_tensors.append(tf.one_hot(
      zero_indexed_groundtruth_classes, num_classes))
print('Done prepping data.')


# ------------------------------
# visualize the f-16s as a sanity check
# ------------------------------
dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%

plt.figure(figsize=(30, 15))
for idx in range(5):
  plt.subplot(2, 3, idx+1)
  plot_detections(
      train_images_np[idx],
      gt_boxes[idx],
      np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32),
      dummy_scores, category_index)
plt.show()
#plt.savefig('sanity_images.png', bbox_inches='tight')
