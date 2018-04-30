'''
Author: Harshita, Siddharth and Arpan
@Description: Bag of Visual Words based action recognition in videos
'''

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.externals import joblib
from cal_optical_flow_seq import extract_flow_seq_train
from cal_optical_flow_seq import extract_flow_val
from clustering import make_codebook
from clustering import extract_vec_points
from create_bovw import create_bovw_traindf
from create_bovw import create_bovw_testdf
from utils import get_video_label
from utils import evaluate


# Paths and parameters
DATASET = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
#DATASET = '/opt/datasets/KTH'
flow_filename = "flow_data"
km_model_filename = "km"
gdsize = 5
destpath = "data_grid"+str(gdsize)
clf_modelname = "model_svm"

        
def main():
    ###########################################################################
    # Step 1: Extract optical flow from training videos and save to disk
    # calculate optical flow vectors of training dataset
    print "Extract optical flow data for training set..."
    flow_filepath = os.path.join(destpath, flow_filename+"_train.pkl")
    
#    features = extract_flow_seq_train(DATASET, grid_size=gdsize)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
#    with open(flow_filepath, "wb") as out_file:
#        pickle.dump(features, out_file)
#    print "Written training features to disk..."
    with open(flow_filepath, 'rb') as infile:
        features = pickle.load(infile)
    
    # extract keypoints (optical flow vectors of consecutive frames for
    # all videos in training dataset)
    mag, ang = extract_vec_points(flow_filepath)
    # change -inf value to 0,  
    mag[np.isinf(mag)] = 0
    print("Magnitude : {}".format(mag.shape))
    print("Angle : {}".format(ang.shape))
    # Magnitude : (74016, 192)
    # Angle : (74016, 192)
    ###########################################################################
    # Normalize
    mag_min, mag_max = np.min(mag), np.max(mag)
    ang_min, ang_max = np.min(ang), np.max(ang)
    mag = (mag - mag_min)/(mag_max - mag_min)
    ang = (ang - ang_min)/(ang_max - ang_min)
    
    ###########################################################################
    # Execute once while training KMeans, to find the cluster centroids
    # generate codebook using k-means clustering
    # The cluster centres represent the vocabulary words. 
    km_file_mag = os.path.join(destpath, km_model_filename+"_mag.pkl")
    km_file_ang = os.path.join(destpath, km_model_filename+"_ang.pkl")
    ##################################
    # Uncomment only while training.
    km_mag = make_codebook(mag, 150)
    km_ang = make_codebook(ang, 130)
    
    # Save to disk, if training is performed
    print("Writing the KMeans models to disk...")
    with open(km_file_mag, "wb") as outfile:
        pickle.dump(km_mag, outfile)
    with open(km_file_ang, "wb") as outfile:
        pickle.dump(km_ang, outfile)
    ##################################
    # Load from disk, for validation and test sets.
    with open(km_file_mag, 'rb') as infile:
        km_mag = pickle.load(infile)
    with open(km_file_ang, 'rb') as infile:
        km_ang = pickle.load(infile)
    ###########################################################################
    # Form the training dataset for supervised classification using SVM
    # Assign the words (flow frames) to their closest cluster centres and count the 
    # frequency for each document(video). Create IDF bow dataframe by weighting
    # df_train is (nVids, 50) for magnitude, with index as videonames
    print("Create a dataframe for magnitudes and angles...")
    df_train_mag, df_train_ang = create_bovw_traindf(features, km_mag, km_ang, \
                                                     mag_min, mag_max, ang_min, ang_max,\
                                                     destpath)
    vids_list = list(df_train_mag.index)
    labels = np.array([get_video_label(v) for v in vids_list])
    # form the labels dataframe having one columns of labels
    labs_df = pd.DataFrame(labels, index=vids_list, columns=['label'])
        
    print("Training dataframe formed.")
    ###########################################################################
    # Train a classifier on the features.
    print("Training with SVM (mag)")
    #clf_mag = RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=134)
    #clf_mag = SVC(kernel="linear",verbose=True)
    clf_mag = LinearSVC(verbose=True, random_state=134, max_iter=2000)
    clf_mag.fit(df_train_mag, labels)
    print("Training with SVM (ang)")
    #clf_ang = RandomForestClassifier(max_depth=5, n_estimators=1000, random_state=124)
    #clf_ang = SVC(kernel="linear",verbose=True)
    clf_ang = LinearSVC(verbose=True, random_state=124, max_iter=2000)
    clf_ang.fit(df_train_ang, labels)
    
    #print("Training complete. Saving to disk.")
    # Save model to disk
    joblib.dump(clf_mag, os.path.join(destpath, clf_modelname+"_mag.pkl"))
    joblib.dump(clf_ang, os.path.join(destpath, clf_modelname+"_ang.pkl"))
    # Load trained model from disk
    clf_mag = joblib.load(os.path.join(destpath, clf_modelname+"_mag.pkl"))
    clf_ang = joblib.load(os.path.join(destpath, clf_modelname+"_ang.pkl"))


    # Train a classifier on both the features.
    print("Training with SVM")
    df_train = pd.concat([df_train_mag, df_train_ang], axis=1)
    #clf_both = SVC(kernel="linear",verbose=True)
    clf_both = LinearSVC(verbose=True, random_state=123, max_iter=2000)
    clf_both.fit(df_train, labels)
    #print("Training with SVM (ang)")
    #clf_ang = SVC(kernel="linear",verbose=True)
    #clf_ang.fit(df_train_ang, labels)
    
    print("Eval on train set mag, ang and both")
    evaluate(clf_mag, df_train_mag, labs_df)
    evaluate(clf_ang, df_train_ang, labs_df)
    evaluate(clf_both, df_train, labs_df)
    ###########################################################################
    # Evaluation on validation set
    # extract the optical flow information from the validation set videos and form dictionary
    bgthresh = 70000 # should be <=90k, to prevent error vq(mag, clusters_mag[0])
    # 
    target_file = os.path.join(destpath, flow_filename+"_test_BG"+str(bgthresh)+".pkl")
    features_val = extract_flow_val(DATASET, bgthresh, grid_size=gdsize, partition="testing")
    with open(target_file, "wb") as outfile:
        pickle.dump(features_val, outfile)

    # Load feaures from disk
    with open(target_file, "rb") as infile:
        features_val = pickle.load(infile)
    
    print("Create dataframe BOVW validation set (mag)")
    df_test_mag, df_test_ang = create_bovw_testdf(features_val, km_mag, km_ang, \
                                                  mag_min, mag_max, ang_min, ang_max,\
                                                  destpath)
    vids_list = list(df_test_mag.index)
    labels = np.array([get_video_label(v) for v in vids_list])
    labs_df = pd.DataFrame(labels, index=vids_list, columns=['label'])
    
    print("Evaluating on the validation set (mag)")
    evaluate(clf_mag, df_test_mag, labs_df)
    
    print("Evaluating on the validation set (ang)")
    evaluate(clf_ang, df_test_ang, labs_df)

    print("Evaluating on the validation set (both features)")
    df_test = pd.concat([df_test_mag, df_test_ang], axis=1)
    evaluate(clf_both, df_test, labs_df)

    ###########################################################################
#    import visualize as vis
#    vis.bar_graph(df_train_mag, df_train_ang, df_train_mag.index, 150)

if __name__ == '__main__':
    main()