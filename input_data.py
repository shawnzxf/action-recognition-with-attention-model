from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import time
import h5py

import pdb

def read_clip_and_label(filename, 
                        batch_size, 
                        start_pos=-1, 
                        num_frames_per_clip=60, 
                        crop_size=7, 
                        shuffle=False):
  """ Read video clips and label them.

  Args:
    filename: string, File name of dataset list to find the video data.
    batch_size: int, Number of video clips in a batch.
    start_pos: int, Initial position to read video from the list.
    num_frames_per_clip: int, Number of frames to extract from a video.
    crop_size: int, Crop size of a cropped video frame.
    shuffle: bool, Flag to indicate whether to shuffle video in the list.
  """
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    # Get the file name of video data
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    filename = dirname.split('/')
    filename = filename[-1]
    # Get the preprocessed CNN feature and the corresponding label.
    file = h5py.File(dirname + '.h5', 'r')
    tmp_data = file[filename+'_data'][:]
    img_datas = [];
    if(len(tmp_data)!=0):
      img_datas = tmp_data
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)

    file.close()
  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len
