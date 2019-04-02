# TensorFlow VGG-16 pre-trained model

'list': to store ucf-101/hmdb-51 dataset list which tells the path 
        and the lable of each video.
'preprocess.py': to pre-process video frames and produce CNN feature.
'utils.py': to provide functions to read images.
N.B. 'list' folder under 'preprocess' directory is different from the one under
 'CHAM' directory

HOW-TO:
# 1 Download the tensorflow version of VGG16 throught the link below
https://github.com/ry/tensorflow-vgg16/raw/master/vgg16-20160129.tfmodel.torrent

# 2 Preprocess the video frames by running 'preprocess.py'
modify paths in 'preprocess.py' if needed and run this script
e.g. run command "CUDA_VISIBLE_DEVICES=0,1 python preprocess.py"

The input ("images") to the TF model is expected to be [batch, height, width, 
channel] where height = width = 224 and channel = 3. Values should be between 0
 and 1.