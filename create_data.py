import argparse
import numpy as np
import os
import pickle

from sklearn.svm import SVC

def get_video_label(srcVid):
    if "boxing" in srcVid:
        return 0
    elif "handclapping" in srcVid:
        return 1
    elif "handwaving" in srcVid:
        return 2
    elif "jogging" in srcVid:
        return 3
    elif "running" in srcVid:
        return 4
    elif "walking" in srcVid:
        return 5

def create_labels(dataset_path):
    dataset = pickle.load(open(dataset_path,"rb")
    X = []
    Y = []

    for video in dataset:
        X.append(video["features"])
    for video in dataset:
        label = get_video_label(video["filename"])
        Y.append(label)

    return X, Y
