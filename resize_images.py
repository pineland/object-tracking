from PIL import Image
import glob
import os

path = "./test/*"
file_list = glob.glob(path)
jpg_filelist = [file for file in file_list if file.endswith(".jpg")]

save_path = "./resized_test/";
if not os.path.isdir(save_path):
  os.mkdir(save_path)

for image_file in jpg_filelist:
  image = Image.open(image_file)
  resize_image = image.resize((853, 640))
  only_image_filename = image_file.split('/')[-1]
  resize_image.save(save_path+only_image_filename)
  print(image_file, ' converted.')
