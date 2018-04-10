import argparse
import numpy as np
import os
import pickle

from scipy.cluster.vq import vq

def make_bovw(codebook_filename,dataset_filename,target_bow_filename):
    codebook = pickle.load(open(os.path.join('data',codebook_filename), "rb"))
    clusters = codebook.cluster_centers_


    dataset = pickle.load(open(os.path.join('data',dataset_filename), "rb"))

    
    print("Make bow vector for each frame")
    n_videos = len(dataset)
    bow = np.zeros((n_videos, clusters.shape[0]), dtype=np.float)
    
    print("check1")
    print(clusters[0].shape)
    
    # Make bow vectors for all videos.
    video_index = 0
    for video in dataset:
        visual_word_ids = vq(video["features"], clusters)[0]
        for word_id in visual_word_ids:
            bow[video_index, word_id] += 1
        video_index += 1

   
    
    ##print("Applying TF-IDF weighting")
    freq = np.sum((bow > 0) * 1, axis = 0)
    idf = np.log((n_videos + 1) / (freq + 1))
    bow = bow * idf

    # Replace features in dataset with the bow vector we've computed.
    video_index = 0
    for i in range(len(dataset)):

        dataset[i]["features"] = bow[video_index]
        video_index += 1

        if (i + 1) % 50 == 0:
            print("Processed %d/%d videos" % (i + 1, len(dataset)))


    pickle.dump(dataset, open(os.path.join('data',target_bow_filename), "wb"))
    
