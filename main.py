
import numpy as np
import os
import pickle
from cal_optical import *
from extract_keypoints import *
from clustering import *
from create_bovw import *
from create_data import *
from sklearn.svm import SVC

# calculate optical flow vectors of training and validation dataset
train_data_path =
extract_optical_flow(train_data_path,"opt_flow_train.p")
val_data_path =
extract_optical_flow(val_data_path,"opt_flow_val.p")

# extract keypoints (optical flow vectors of consecutive frames for all videos in training dataset)
make_keypoints("data/opt_flow_train.p","train_keypoints.p")

# generate codebook using k-means clustering
make_codebook("data/train_keypoints.p","codebook.p")

# generate bag of words vectors from optical flow data and codebook
# for training dataset
make_bovw("codebook.p","opt_flow_train.p","train_bovw.p")

#for validation dataset
make_bovw("codebook.p","opt_flow_val.p","val_bovw.p")



Xtrain,Ytrain = create_labels("data/opt_flow_train.p")
Xval,Yval = create_labels("data/opt_flow_val.p")

#train SVM on training data
clf = SVC(1,kernel="linear",verbose=True)
clf.fit(Xtrain,Ytrain)
pickle.dump(clf,open("data/svm_C1.p","wb"))
