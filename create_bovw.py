import numpy as np
import os
import pickle
import pandas as pd

from scipy.cluster.vq import vq

# function to extract magnitude space vectors and angle space vectors
# optical_flow_data_path: 
def extract_vec_points(optical_flow_data_path):
    # Convert into magnitude space points and angle space points.
    mag_vecs = []
    ang_vecs = []
    data = pickle.load(open(optical_flow_data_path,"rb"))
    # Iterate over the videos contained in the dictionary
    for k, v in data.iteritems():
        # Iterate over the 2x16x12 dimensional matrices, for each flow feature
        for flow_feature in v:
            mag_vecs.append(flow_feature[0,...].ravel())
            ang_vecs.append(flow_feature[1,...].ravel())
        print("Video {} : Size_mag : {}".format(k, len(mag_vecs)))
    print("Done for {}".format(optical_flow_data_path))
    return np.array(mag_vecs), np.array(ang_vecs)


# Form a magnitude and angle dataframes using extracted features kept in 
# feats_data_dict. BOW rows are calculated using counts of 
# Returns two dataframes of (nVids, nClusters). 
# assignment to one of the centroids.
# feats_data_dict: {'filename':[array(2,16,12), array(2,16,12), ....]}
def create_bovw_traindf(feats_data_dict, kmeans_mag, kmeans_ang, \
                        mag_min, mag_max, ang_min, ang_max, idfpath):
    
    clusters_mag = kmeans_mag.cluster_centers_
    clusters_ang = kmeans_ang.cluster_centers_
    vids_list = feats_data_dict.keys()
    n_videos = len(vids_list)
    
    # Create a dataframe of size n_videos X n_clusters
    print("Make bow vector for each frame")
    bow_mag = np.zeros((n_videos, clusters_mag.shape[0]))
    bow_ang = np.zeros((n_videos, clusters_ang.shape[0]))
    
    print("Shape of bow_mag : {}".format(bow_mag.shape))
    print("Shape of bow_ang : {}".format(bow_ang.shape))
    
    # Make bow vectors for all videos.
    for video_index, video in enumerate(vids_list):
        # Read the features from the dictionary and get the
        frames_feats = feats_data_dict[video]
        mag, ang = [], []
        
        # feat is an array(2,16,12)
        mag = [feat[0,...].ravel() for feat in frames_feats]
        ang = [feat[1,...].ravel() for feat in frames_feats]
        mag = np.array(mag)  # (nFlowFrames, 192)
        ang = np.array(ang)
        # change inf values to 0
        mag[np.isinf(mag)] = 0
        ang[np.isinf(ang)] = 0
            
        ##### Normalize
        mag = (mag - mag_min)/(mag_max - mag_min)
        ang = (ang - ang_min)/(ang_max - ang_min)
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        word_ids_mag = vq(mag, clusters_mag)[0]  # ignoring the distances in [1]
        word_ids_ang = vq(ang, clusters_ang)[0]  # ignoring the distances in [1]
        for word_id in word_ids_mag:
            bow_mag[video_index, word_id] += 1  
        
        for word_id in word_ids_ang:
            bow_ang[video_index, word_id] += 1  
        
        print("Done video {} : {}".format(video_index, video))
    
    ##print("Applying TF-IDF weighting")
    # This is applicable for only the training set
    # For validation/test set, the idf will be same as for the training set
    freq_mag = np.sum((bow_mag > 0) * 1, axis = 0)
    idf_mag = np.log((n_videos + 1.0) / (freq_mag + 1.0))
    bow_mag = bow_mag * idf_mag
    
    freq_ang = np.sum((bow_ang > 0) * 1, axis = 0)
    idf_ang = np.log((n_videos + 1.0) / (freq_ang + 1.0))
    bow_ang = bow_ang * idf_ang
    
    # save idf_mag to disk
    pickle.dump(idf_mag, open(os.path.join(idfpath,"idf_mag.pkl"), "wb"))
    pickle.dump(idf_ang, open(os.path.join(idfpath,"idf_ang.pkl"), "wb"))
    print("Saved IDF weights to disk.")
    
    # form the training dataframe
    bow_mag = pd.DataFrame(bow_mag, index=vids_list)
    bow_ang = pd.DataFrame(bow_ang, index=vids_list)
    return bow_mag, bow_ang
    
    
# Form a dataframe using extracted features kept in feats_data_dict by finding the 
# assignment to one of the centroids.
# feats_data_dict: {'filename':[array(2,16,12), array(2,16,12), ....]}
def create_bovw_testdf(feats_data_dict, kmeans_mag, kmeans_ang, \
                       mag_min, mag_max, ang_min, ang_max, idfpath):
    clusters_mag = kmeans_mag.cluster_centers_
    clusters_ang = kmeans_ang.cluster_centers_
    vids_list = feats_data_dict.keys()
    n_videos = len(vids_list)
    
    # Create a dataframe of size n_videos X n_clusters
    print("Make bow vector for each frame")
    bow_mag = np.zeros((n_videos, clusters_mag.shape[0]))
    bow_ang = np.zeros((n_videos, clusters_ang.shape[0]))
    
    print("Shape of bow_mag : {}".format(bow_mag.shape))
    print("Shape of bow_ang : {}".format(bow_ang.shape))
    
    # Make bow vectors for all videos.
    for video_index, video in enumerate(vids_list):
        # Read the features from the dictionary and get the
        frames_feats = feats_data_dict[video]
        mag, ang = [], []
        
        # feat is an array(2,16,12)
        mag = [feat[0,...].ravel() for feat in frames_feats]
        ang = [feat[1,...].ravel() for feat in frames_feats]
        mag = np.array(mag)  # (nFlowFrames, 192)
        ang = np.array(ang)
        # change inf values to 0
        mag[np.isinf(mag)] = 0
        ang[np.isinf(ang)] = 0
            
        ##### Normalize
        mag = (mag - mag_min)/(mag_max - mag_min)
        ang = (ang - ang_min)/(ang_max - ang_min)
        # find cluster centroid assignments for all points
        # returns a tuple, with first element having ids of the cluster centroid 
        # to which the row i belongs to. Second element is the distance between 
        # the nearest code and the ith row.
        # visual_word_ids is a 1D array
        print("Shape mag : {} :: Shape clusters : {}".format(mag.shape, clusters_mag.shape))
        word_ids_mag = vq(mag, clusters_mag)[0]  # ignoring the distances in [1]
        word_ids_ang = vq(ang, clusters_ang)[0]  # ignoring the distances in [1]
        for word_id in word_ids_mag:
            bow_mag[video_index, word_id] += 1
               
        for word_id in word_ids_ang:
            bow_ang[video_index, word_id] += 1
        
        print("Done video {} : {}".format(video_index, video))

    ##print("Applying TF-IDF weighting")
    # This is applicable for only the training set
    # For validation/test set, the idf will be same as for the training set
    # load idf_mag from disk
    idf_mag = pickle.load(open(os.path.join(idfpath,"idf_mag.pkl"), "rb"))
    idf_ang = pickle.load(open(os.path.join(idfpath,"idf_ang.pkl"), "rb"))
    print("Loaded IDF weights from disk.")
    bow_mag = bow_mag * idf_mag
    bow_ang = bow_ang * idf_ang
    
    # form the test/validation dataframe
    bow_mag = pd.DataFrame(bow_mag, index=vids_list)
    bow_ang = pd.DataFrame(bow_ang, index=vids_list)
    return bow_mag, bow_ang     # return the test dataframes
