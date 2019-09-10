# Convolutional Hierarchical Attention Model for action recognition in video

This code is for action recognition in video through a convolutional hierarchical attention model, and basically is based on the paper [CHAM: action recognition using convolutional hierarchical attention model, Shiyan et al](https://arxiv.org/abs/1705.03146). 3 different kinds of network models are built and train on UCF-101 or HMDB-51 dataset in this code package. 

## HOW-TO:

#### 1 Install package needed
Package needed includes tensorflow, PIL, numpy and h5py.  
N.B. version: python 2.7 and tensorflow 1.4.0

#### 2 Modify the parameters for network model training and testing
Important parameters are specified on each script, change those parameters if needed.

#### 3 Run training script to train the network model
E.g. run command as below. 

```CUDA_VISIBLE_DEVICES=0,1 python train.py```

#### 4 Test the model just trained
E.g. run command as below. 

```CUDA_VISIBLE_DEVICES=0,1 python test.py```  
N.B. Testing are automatically conducted in train.py during training by default.

## Results
#### UCF-101 dataset
On the UCF-101 dataset, results are as following:

| models  | accuracy |
| ------------- | ------------- |
|  FC-Attention  |  68.59%  |
|  Conv-Attention  |  70.16%  |
|  Hierarchical Conv-Attention  |  73.80%  |


## Files

list: to store ucf-101/hmdb-51 dataset list which tells the path and the lable of preprocessed CNN feature of each video.

models: to store the models trained by this code.

proprecess: to store the codes which complete pre-processing of videos on the dataset. There is another README.md for pre-processing in this folder.

input_data.py: to provide functions that read and label videos for training and testing.

model.py: to provide classes and functions for constructing 3 different kinds of network model.

test.py: to provide functions to test the model trained.

train.py: to provide functions to train the model defined on 'model.py'.

N.B. 'list' folder under this directory is different from the one under 'preprocess' directory.
