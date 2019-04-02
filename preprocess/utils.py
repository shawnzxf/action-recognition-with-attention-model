import skimage
import skimage.io
import skimage.transform
import numpy as np
import random
import os
import pdb

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_images(dir_path, num_frames_per_clip):
  ret_arr = np.zeros((num_frames_per_clip,224,224,3),dtype=np.float32)
  s_index = 0
  for parent, dirnames, filenames in os.walk(dir_path):
    if(len(filenames)<num_frames_per_clip):
      pad_num = num_frames_per_clip - len(filenames)
      filenames = sorted(filenames)
      for i in range(len(filenames)):
        image_name = str(dir_path) + '/' + str(filenames[i])
        img = read_image(image_name)
        img_data = np.array(img).astype(np.float32)
        ret_arr[i,:,:,:] = img_data
      for i in range(pad_num):
        # pdb.set_trace()
        ret_arr[len(filenames)+i,:,:,:] = img_data
      return ret_arr
    else:
      filenames = sorted(filenames)
      s_index = random.randint(0, len(filenames) - num_frames_per_clip)
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(dir_path) + '/' + str(filenames[i])
        img = read_image(image_name)
        img_data = np.array(img)
        ret_arr[i-s_index,:,:,:] = img_data
      return ret_arr

def read_image(image_path):
  # load image
  img = skimage.io.imread(image_path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img

# returns the top1 string
def print_prob(prob):
  #print prob
  print "prob shape", prob.shape
  pred = np.argsort(prob)[::-1]

  # Get top1 label
  top1 = synset[pred[0]]
  print "Top1: ", top1
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  print "Top5: ", top5
  return top1
