import numpy as np
import os
import pickle

def make_keypoints(optical_flow_data_path,target_file):
    # Store keypoints of training set.
    train_keypoints = []
    data = pickle.load(open(optical_flow_data_path,"rb"))
    for video in data:
        for frame in video["features"]:
            train_keypoints.append(frame)

    print("Saving keypoints of training set")
    pickle.dump(train_keypoints, open(os.path.join('data',target_file), "wb"))
