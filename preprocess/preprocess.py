import tensorflow as tf
import utils
import h5py
import pdb
import os

# Indicate file paths to read VGG model, dataset list and directory 
# to store CNN feature
model_name = "/home/zhaoxf/tutorials/tf-vgg16/vgg16.tfmodel"
test_list_file = './list/ucf101_test.list'
saved_dir = '/home/zhaoxf/data/ucf_101/224'
batch_size = 60

# Import the trained VGG16 model
with open(model_name, mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print "graph loaded from disk"

graph = tf.get_default_graph()

# Read lines from the list
lines = open(test_list_file,'r')
lines = list(lines)
num_videos = len(lines)

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  print "variables initialized"

  for video_index in range(num_videos):
    # Get the path to read video frames
    line = lines[video_index].strip('\n').split()
    dirname = line[0]
    label = line[1]

    # Set filename for the CNN feature
    path = dirname.split('/')
    data_path = saved_dir + '/' + path[-2]
    data_name = path[-1] + '.h5'
    
    if not os.path.exists(data_path):
      os.makedirs(data_path)
    if not os.path.exists(data_path + '/' + data_name):
      frames = utils.load_images(dirname, num_frames_per_clip=60)
      assert frames.shape == (60, 224, 224, 3)

      # Get CNN feature from pool5 layer of the VGG16
      feed_dict = {images: frames}
      preprocessed_tensor = graph.get_tensor_by_name("import/pool5:0")
      features = sess.run(preprocessed_tensor, feed_dict=feed_dict)

      # Store the CNN feature
      print('%d/%d storing preprocessed features %s'
        %(video_index+1,num_videos,data_name))
      write_file = h5py.File(data_path + '/' + data_name,'w')
      write_file.create_dataset(path[-1]+'_data', data = features)
      write_file.create_dataset(path[-1]+'_label',data = label)
      write_file.close()

    else:
      print('%d/%d preprocessed features already existed %s'
        %(video_index+1,num_videos,data_name))
      continue
  print('done!')
 
