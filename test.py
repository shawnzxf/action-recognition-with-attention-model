from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from six.moves import xrange
import model
import input_data
import os
import time

import pdb

def placeholder_inputs(batch_size,timesteps, crop_size, channels):
  """Generate placeholder variables to represent the input tensors.

  Args:
    batch_size: int, Number of video clips in a batch.
    timesteps: int, Number of timesteps in a video clip.
    crop_size: int, Crop size of a cropped video frame.
    channels: int, Number of input(CNN features) channels.
  """
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         timesteps,
                                                         crop_size,
                                                         crop_size,
                                                         channels))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def test(model_name,
         test_list_file,
         get_model,
         batch_size,
         num_class,
         timesteps,
         crop_size,
         channels,
         test_report_name):
  """Test network model.

  Args:
    model_name: string, File name of the model to be tested.
    test_list_file: string, File path of test dataset list.
    get_model: network function, Network to learning and classify different
      videos. There 3 types of network for selection: model.cham, 
      model.conv_atten and model.fc_atten.
    batch_size: int, Number of video clips in a batch.
    num_class: int, Number of all classes in the dataset.
    timesteps: int, Number of timesteps in a video clip.
    crop_size: int, Crop size of a cropped video frame.
    channels: int, Number of input(CNN features) channels.
    test_report_name: string, File name of test report.
  """
  num_test_videos = len(list(open(test_list_file,'r')))
  print('Number of test videos = %d' % num_test_videos)

  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(
      batch_size, timesteps=timesteps, crop_size=crop_size, channels=channels)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    with tf.Session(config=config) as sess:
      logits_wo_softmax = get_model(
        images_placeholder, batch_size, timesteps, crop_size, num_class, drop_out=1) 
      logits = tf.nn.softmax(logits_wo_softmax,name='pred')
      train_var_list = [v for v in tf.trainable_variables()]
      total_para = np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
      print('total number of parameters:', total_para)

      # Create a saver for writing training checkpoints
      saver = tf.train.Saver()

      sess.run(tf.global_variables_initializer())

      # Load the model to be tested
      if os.path.isfile(model_name+'.meta'):
        saver.restore(sess, model_name)
        print('Finish relaoding model:' + model_name)
      else:
        print('Pretrained model can not be found, please check again')

      bufsize = 0
      write_file = open(test_report_name, 'w+', bufsize)
      next_start_pos = 0
      num_iter = int(num_test_videos/batch_size)
      acc_sum = 0

      for step in xrange(num_iter):
        # Get label prediction on test dataset
        start_time = time.time()
        test_images, test_labels, next_start_pos, _, valid_len = \
        input_data.read_clip_and_label(
          filename=test_list_file, batch_size=batch_size, start_pos=next_start_pos)
        predict_score = logits.eval(
          session=sess, feed_dict={images_placeholder:test_images})

        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step+1,duration))

        # Write test report
        for i in range(0, valid_len):
          true_label = test_labels[i]
          top1_predicted_label = np.argmax(predict_score[i])
          write_file.write('{}, {}, {}, {}\n'.format(
                          true_label,
                          predict_score[i][true_label],
                          top1_predicted_label,
                          predict_score[i][top1_predicted_label]))
          if true_label == top1_predicted_label:
		    acc_sum += 1 

      write_file.write('accuracy:%.4f' % (acc_sum / num_test_videos))
      write_file.close()
  print('accuracy:%.4f' % (acc_sum / num_test_videos))
  print('Finish test!')

def main(_):
  test(model_name = './models/cham_hmdb51.model',
       test_list_file = './list/224/hmdb51_test.list',
       get_model = model.cham,
       batch_size = 10,
       num_class = 51,
       timesteps = 60,
       crop_size = 7,
       channels = 512,
       test_report_name = 'predict_ret.txt')

if __name__ == '__main__':
  tf.app.run()
