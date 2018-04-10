'''
bag of visual words
'''

import numpy as np
import os
import pickle
from cal_optical_flow import *
from extract_keypoints import *
from evaluate import *
from clustering import *
from create_bovw import *
from create_data import *
from cal_optical_flow_seq import *
from sklearn.svm import SVC


# calculate optical flow vectors of training and validation dataset
train_data_path = '/opt/datasets/KTH/kth_actions_train'
#extract_optical_flow_seq(train_data_path,"opt_flow_train_seq_2.p")
#extract_optical_flow_seq(train_data_path,"opt_flow_train.p")
val_data_path = '/opt/datasets/KTH/kth_actions_validation'
#extract_optical_flow(val_data_path,"opt_flow_val_seq_2.p")
#extract_optical_flow(val_data_path,"opt_flow_val.p")

# extract keypoints (optical flow vectors of consecutive frames for all videos in training dataset)
#make_keypoints("data/opt_flow_train_seq_2.p","train_keypoints_seq_2.p")
#make_keypoints("data/opt_flow_train.p","train_keypoints.p")

# generate codebook using k-means clustering
#make_codebook("data/train_keypoints_seq_2.p","codebook_seq_2.p")
#make_codebook("data/train_keypoints.p","codebook.p")

# generate bag of words vectors from optical flow data and codebook
# for training dataset
#make_bovw("codebook_seq_2.p","opt_flow_train_seq_2.p","train_bovw_seq_2.p")
#make_bovw("codebook.p","opt_flow_train.p","train_bovw.p")

#for validation dataset
make_bovw("codebook_seq_2.p","opt_flow_val_seq_2.p","val_bovw_2.p")
#make_bovw("codebook_seq.p","opt_flow_val.p","val_bovw_2.p")

#Xtrain,Ytrain = create_labels("data/train_bovw.p")
#Xval,Yval = create_labels("data/val_bovw.p")

#Xtrain,Ytrain = create_labels("data/train_bovw_seq.p")
#Xval,Yval = create_labels("data/val_bovw_2.p")

Xtrain,Ytrain = create_labels("data/train_bovw_seq_2.p")
Xval,Yval = create_labels("data/val_bovw_2.p")

#train SVM on training data
evaluate(Xtrain,Ytrain,Xval,Yval)

print("SVM trained")
