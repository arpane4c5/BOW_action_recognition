import argparse
import numpy as np
import os
import pickle

from sklearn.cluster import KMeans
from numpy import size

def make_codebook(keypoints_path,codebook_file_name):
    train_features = pickle.load(open(keypoints_path, "rb"))
    n_features = len(train_features)
    print("Running KMeans clustering")
    #clustering with k-means
    kmeans = KMeans(init='k-means++', n_clusters=200, n_init=10, n_jobs=2,
    verbose=1)
    kmeans.fit(train_features)
    
    pickle.dump(kmeans, open(os.path.join('data',codebook_file_name), "wb"))
