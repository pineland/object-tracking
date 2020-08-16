from PIL import Image
import glob
import os
import xml.etree.ElementTree as elemTree

image_path = "./train/*"
annotation_path = "./annotations/*"

image_file_list = glob.glob(image_path)
annotation_file_list = glob.glob(annotation_path)

jpg_filelist = [file for file in image_file_list if file.endswith(".jpg")]
xml_filelist = [file for file in annotation_file_list if file.endswith(".xml")]

new_width = 853
new_height = 640

image_save_path = "./resized_train/";
annotation_save_path = "./resized_annotations/";
if not os.path.isdir(image_save_path):
  os.mkdir(image_save_path)
if not os.path.isdir(annotation_save_path):
  os.mkdir(annotation_save_path)

for image_file, xml_file in zip(jpg_filelist, xml_filelist):
  if not glob.glob(xml_file): # check if xml exists.
    print("Annotation file of " + image_file + " NOT exists.")
    contiune

  # resize image
  image = Image.open(image_file)
  resize_image = image.resize((new_width, new_height))
  only_image_filename = image_file.split('/')[-1]
  resize_image.save(image_save_path+only_image_filename)
  print(image_file, ' converted.')

  # rewrite xml
  origin_width, origin_height = image.size
  x_scale = new_width/origin_width
  y_scale = new_height/origin_height

  tree = elemTree.parse(xml_file)
  size = tree.find('./size')
  size.find('./width').text = str(new_width)
  size.find('./height').text = str(new_height)

  objects = tree.findall('./object')
  for i, object_ in enumerate(objects):
      bndbox = object_.find('./bndbox')
      bndbox.find('./xmin').text = str(round(int(bndbox.find('./xmin').text) * x_scale))
      bndbox.find('./ymin').text = str(round(int(bndbox.find('./ymin').text) * y_scale))
      bndbox.find('./xmax').text = str(round(int(bndbox.find('./xmax').text) * x_scale))
      bndbox.find('./ymax').text = str(round(int(bndbox.find('./ymax').text) * y_scale))
  only_xml_filename = xml_file.split('/')[-1]
  tree.write(annotation_save_path+only_xml_filename, encoding='utf8')
  print(xml_file, ' converted.')
