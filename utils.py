import numpy as np
import os
import pickle
import csv

from sklearn.svm import SVC

# function to read the kth_sequences.txt file and return the dictionary of 
# frame sequences where actions occur.
# returns d: dictionary eg {'filename': '1-10, 20-30, 50-100'}
def read_seq(filePath):
    d = {}
    with open(filePath,'r') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        #csvout = csv.writer(csvout)
        for row in tsvin:
            # if row is not and empty list then add the
            if len(row)>0:
                key = row[0].strip()
                value = row[-1].strip()
                d[key] = value
    print(len(d))
    return (d)


# function to return the label (int) given the name of the video
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

# load the data given the path of the pkl file.
def create_labels(dataset_path):
    dataset = pickle.load(open(dataset_path,"rb"))
    X = []
    Y = []
    for video in dataset:
        X.append(video["features"])
    for video in dataset:
        label = get_video_label(video["filename"])
        Y.append(label)
    return X, Y
