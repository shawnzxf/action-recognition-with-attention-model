import tensorflow as tf
import numpy as np
from six.moves import xrange
import model
import input_data
import os
import time

from test import test

import pdb

def placeholder_inputs(batch_size,timesteps, crop_size, channels):
  """Generate placeholder variables to represent the input tensors.

  Args:
    batch_size: int, Number of video clips in a batch.
    timesteps: int, Number of timesteps in a video clip.
    crop_size: int, Crop size of a cropped video frame.
    channels: int, Number of input(CNN features) channels.  
  """
  images_placeholder = tf.placeholder(tf.float32, 
                                      shape=(batch_size,
                                             timesteps,
                                             crop_size,
                                             crop_size,
                                             channels), 
                                      name='input_frames')
  labels_placeholder = tf.placeholder(tf.int64, 
                                      shape=(batch_size),
                                      name='input_labels')
  return images_placeholder, labels_placeholder

def loss_compute(logit, labels):
  """Compute loss function for training, including weight decay losses.

  Args:
    logit: 4-D tensor, Output(label prediction) of network with shape [
      batch_size, timesteps, width, height, channels].
    labels: 1-D tensor, True labels of logit with shape[batch_size].
  """
  # Compute cross entropy for netowrk
  cross_entropy_mean = tf.reduce_sum(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit))
  tf.summary.scalar('cross_entropy',cross_entropy_mean)
  # Compute weight decay loss
  weight_decay_loss = tf.get_collection('weightdecay_losses')
  tf.summary.scalar('weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

  # Calculate the total loss.
  total_loss = cross_entropy_mean + weight_decay_loss 
  tf.summary.scalar('total_loss', tf.reduce_mean(total_loss) )
  return total_loss

def acc_compute(logit, labels):
  """Compute accuracy for model prediction.

  Args:
    logit: 4-D tensor, Output(label prediction) of network with shape [
      batch_size, timesteps, width, height, channels].
    labels: 1-D tensor, True labels of logit with shape[batch_size].
  """
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def acc_count(logit, labels):
  """Count the number of accurate prediction.

  Args:
    logit: 4-D tensor, Output(label prediction) of network with shape [
      batch_size, timesteps, width, height, channels].
    labels: 1-D tensor, True labels of logit with shape[batch_size].
  """
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
  return accuracy

def train(model_save_dir,
          model_name,
          train_list_file,
          validation_list_file,
          log_file,
          get_model,
          use_pretrained_model,
          crop_size,
          timesteps,
          channels,
          batch_size,
          num_class,
          keep_prob,
          starter_lr,
          num_iter,
          num_epoch):
  """Train the network constructed.

  Args:
    model_save_dir: string, Directory path to save the model trained.
    model_name: string, Name given to the model trained.
    train_list_file: string, File path of training dataset list.
    validation_list_file: string, File path of validation dataset list.
    log_file: string, File path to save training logs.
    get_model: network function, Network to learning and classify different
      videos. There 3 types of network for selection: model.cham, 
      model.conv_atten and model.fc_atten.
    use_pretrained_model: bool, Flag to indicate whether to train on the basis
      of a trained network model.
    crop_size: int, Crop size of a cropped video frame.
    timesteps: int, Number of timesteps in a video clip.
    channels: int, Number of input(CNN features) channels.
    batch_size: int, Number of video clips in a batch.
    num_class: int, Number of all classes in the dataset.
    keep_prob: float, Rate of drop out while training.
    starter_lr: float, Initial learning rate for training.
    num_iter: int, number of iteration of training process.
    num_epoch: int, number of epoch of training process.
  """
  # Create directory to save model
  if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
  model_filename = model_name

  with tf.Graph().as_default():
    # Set the parameters needed for training
    global_step = tf.get_variable('global_step',
                                  [], 
                                  initializer=tf.constant_initializer(),
                                  trainable = False)
    # Learning rate decays as training step increases
    learning_rate = tf.train.exponential_decay(
      starter_lr, global_step,3000,0.1,staircase=True)

    images_placeholder, labels_placeholder = placeholder_inputs(
      batch_size, timesteps, crop_size, channels)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    with tf.Session(config=config) as sess:
      logits_wo_softmax = get_model(
        images_placeholder,batch_size,timesteps,crop_size,num_class,keep_prob)
      # Softmax normalization
      logits = tf.nn.softmax(logits_wo_softmax, name='pred') 
      # Compute loss function for training with logits without softmax
      loss = loss_compute(logits_wo_softmax, labels_placeholder)
      acc = acc_compute(logits, labels_placeholder)
      acc_counter = acc_count(logits, labels_placeholder)
      train_var_list = [v for v in tf.trainable_variables()]
      train_op = tf.train.AdamOptimizer(
        learning_rate).minimize(loss, global_step=global_step, var_list=train_var_list)

      total_para = np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
      print('total number of parameters:', total_para)

      # Create a saver for writing training checkpoints
      saver = tf.train.Saver(train_var_list, max_to_keep=4)

      sess.run(tf.global_variables_initializer())
      # Load trained model if there is any
      if os.path.isfile(model_save_dir+'/'+model_filename+'.meta') and use_pretrained_model:
        saver.restore(sess, model_save_dir+'/'+model_filename)
        print('Finish laoding model:' + model_filename)

      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(log_file, sess.graph)
      test_writer = tf.summary.FileWriter(log_file, sess.graph)
      
      for epoch in xrange(num_epoch):
        for step in xrange(num_iter):
          # Train the network model
          start_time = time.time()
          train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                                      filename=train_list_file,
                                      batch_size = batch_size,
                                      num_frames_per_clip = timesteps,
                                      crop_size = crop_size,
                                      shuffle=True)
          sess.run(train_op, feed_dict = {images_placeholder:train_images,
                                          labels_placeholder:train_labels})

          duration = time.time() - start_time
          print('Epoch %d Step %d: %.3f sec' % (epoch,step,duration))

          # Save a checkpoints and evaluate the model periodically
          if (step%1000==0 and step!=0) or (step + 1) == num_iter:
            saver.save(sess, os.path.join(model_save_dir, model_name),global_step=step)
            # Test the network model trained above
            test(model_name=os.path.join(model_save_dir,model_name)+'-'+str(step),
                 test_list_file=validation_list_file,
                 batch_size=batch_size,
                 num_class=num_class,
                 timesteps=timesteps,
                 crop_size=crop_size,
                 channels=channels,
                 test_report_name='predict_ret_'+str(step)+'.txt')
          if (step) % 10 == 0 or (step + 1) == num_iter:
            print('Training Data Eval:')
            summary, accuracy, acc_num = sess.run([merged, acc, acc_counter], 
                                    feed_dict={images_placeholder:train_images,
                                               labels_placeholder:train_labels})
            print('accuracy rate:%.5f    accuracy number:%d'%(accuracy, acc_num))
            train_writer.add_summary(summary, step)
            
            print('Validation Data Eval:')
            val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                                      filename=validation_list_file,
                                      batch_size = batch_size,
                                      num_frames_per_clip = timesteps,
                                      crop_size = crop_size,
                                      shuffle=True)
            summary, accuracy, acc_num = sess.run([merged, acc, acc_counter],
                                         feed_dict={images_placeholder:val_images,
                                                    labels_placeholder:val_labels})
            print('accuracy rate:%.5f    accuracy number:%d'%(accuracy, acc_num))
            test_writer.add_summary(summary, step)
  print('Finish training!')

def main(_):
  train(model_save_dir = './models',
        model_name = 'cham_hmdb51.model',
        train_list_file = './list/224/hmdb51_train.list',
        validation_list_file = './list/224/hmdb51_test.list',
        log_file = './visual_logs/train',
        get_model = model.cham,
        use_pretrained_model = True,
        crop_size = 7,
        timesteps = 60,
        channels = 512,
        batch_size = 10,
        num_class = 51,
        keep_prob = 0.5,
        starter_lr = 0.0001,
        num_iter = 9000,
        num_epoch = 1)

if __name__ == '__main__':
  tf.app.run()
