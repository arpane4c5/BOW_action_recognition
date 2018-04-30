from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
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
gdsize = 20
destpath = "data_grid"+str(gdsize)
clf_modelname = "model_svm"

def cluster_visualize(km, data, txt):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    cluster_labels = km.predict(data)
    labels = pd.DataFrame(cluster_labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(pca_result[:,0],pca_result[:,1], c=labels[0],s=40,alpha=.5)
    plt.colorbar(scatter)
    plt.savefig(txt)


def make_graph(run,walk,jog,box,wave,clap,title,txt):
    N = 70
    ind = np.arange(0,350,5)
    print(ind)
    width = .5
    
    fig,ax = plt.subplots(1,1,figsize=(175,100))
   
    rects1 = ax.bar(ind,run,width,color='r')
    rects2 = ax.bar(ind+width,walk,width,color='g')
    rects3 = ax.bar(ind+2*width,jog,width,color='b')
    rects4 = ax.bar(ind+3*width,box,width,color='y')
    rects5 = ax.bar(ind+4*width,wave,width,color='c')
    rects6 = ax.bar(ind+5*width,clap,width,color='m')
    
    plt.legend((rects1[0],rects2[0],rects3[0],rects4[0],rects5[0],rects6[0]) ,("running","walking","joging","boxing","waving","claping"),prop={'size': 125})
    #ax.set_xticks(ind + width/6)
    #ax.set_xticklabels(np.arange(70))
    plt.xlabel('codewords',fontsize=150)
    plt.ylabel('average frequency count',fontsize=150)
    plt.suptitle(title,fontsize=250)
    
    #plt.xticks(np.arange(0, 70, step=1))
    plt.tick_params(axis='y', which='major', labelsize=100)
    plt.tick_params(axis='y', which='minor', labelsize=100)
    plt.tick_params(axis='x', which='major', labelsize=75)
    plt.tick_params(axis='x', which='minor', labelsize=75)
    
    '''y_pos = np.arange(len(val_m))
    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
    fig.suptitle(title,fontsize=15)
    ax1.bar(y_pos,val_m,align='center',alpha=1)
    ax1.set_title("magitude")
    ax1.set_ylim([0,80])
    y_pos = np.arange(len(val_a))
    ax2.bar(y_pos,val_a,align='center',alpha=1)
    ax2.set_title("angle")
    ax2.set_ylim([0,30])'''
    #plt.savefig(txt)
    plt.show()
    
def draw_line_graph(box,clap,wave,jog,run,walk,title,txt, m):
    ind = np.arange(1, 151)
    plt.figure(m)
    
    plt.subplot(321)
    plt.plot(ind, box, 'r-')
    plt.legend(["Boxing"], loc="upper right")

    plt.subplot(322)
    plt.plot(ind, clap, 'b-')
    plt.legend(["Hand-clapping"], loc="upper left")

    plt.subplot(323)
    plt.plot(ind, wave, 'g-')
    plt.legend(["Hand-waving"], loc="upper right")
    
    plt.subplot(324)
    plt.plot(ind, jog, 'y-')
    plt.legend(["Jogging"], loc="upper left")

    plt.subplot(325)
    plt.plot(ind, run, 'c-')
    plt.legend(["Running"], loc="upper right")

    plt.subplot(326)
    plt.plot(ind, walk, 'k-')
    plt.legend(["Walking"], loc="upper left")

    #plt.plot(ind, jog)
    #plt.plot(ind, run)
    #plt.plot(ind, walk)
    #plt.legend(['box', 'clap', 'wave', 'jog', 'run', 'walk'], loc='upper left')
    plt.suptitle(title, fontsize=16)
    #plt.xlabel("Size of Codebook")
    #plt.ylabel("Averaged Action Frequency Count")
    plt.show()
    plt.savefig(txt)
    

def line_graph(df_M, df_A, list_names, c):
    walk_m = np.zeros(c, dtype=float)
    run_m = np.zeros(c, dtype=float)
    box_m = np.zeros(c, dtype=float)
    jog_m = np.zeros(c, dtype=float)
    clap_m = np.zeros(c, dtype=float)
    wave_m = np.zeros(c, dtype=float)
    
    walk_a = np.zeros(c, dtype=float)
    run_a = np.zeros(c, dtype=float)
    box_a = np.zeros(c, dtype=float)
    jog_a = np.zeros(c, dtype=float)
    clap_a = np.zeros(c, dtype=float)
    wave_a = np.zeros(c, dtype=float)
    
    w=0
    r=0
    b=0
    j=0
    c=0
    wv=0
    
    for i in range(len(list_names)):
        if "boxing" in list_names[i]:
            b=b+1
            box_m = box_m + np.array(df_M.iloc[i])
            box_a = box_a + np.array(df_A.iloc[i])
        if "handclapping" in list_names[i]:
            c=c+1
            clap_m = clap_m + np.array(df_M.iloc[i])
            clap_a = clap_a + np.array(df_A.iloc[i])
        if "handwaving" in list_names[i]:
            wv=wv+1
            wave_m = wave_m + np.array(df_M.iloc[i])
            wave_a = wave_a + np.array(df_A.iloc[i])
        if "jogging" in list_names[i]:
            j=j+1
            jog_m = jog_m + np.array(df_M.iloc[i])
            jog_a = jog_a + np.array(df_A.iloc[i])
        if "running" in list_names[i]:
            r=r+1
            run_m = run_m + np.array(df_M.iloc[i])
            run_a = run_a + np.array(df_A.iloc[i])
        if "walking" in list_names[i]:
            w=w+1
            walk_m = walk_m + np.array(df_M.iloc[i])
            walk_a = walk_a + np.array(df_A.iloc[i])

    walk_a = walk_a/w
    walk_m = walk_m/w
    run_a = run_a/r
    run_m = run_m/r
    jog_a = jog_a/j
    jog_m = jog_m/j
    wave_a = wave_a/wv
    wave_m = wave_m/wv
    clap_a = clap_a/c
    clap_m = clap_m/c
    box_a = box_a/b
    box_m = box_m/b

    #make_graph(run_m,walk_m,jog_m,box_m,wave_m,clap_m,"Magnitudes","mag.png")
    #make_graph(run_a,walk_a,jog_a,box_a,wave_a,clap_a,"Angles","ang.png")
    draw_line_graph(box_m, clap_m, wave_m, jog_m, run_m, walk_m,"Averaged magnitude values over codebook","mag.png",1)
    draw_line_graph(box_a, clap_a, wave_a, jog_a, run_a, walk_a,"Averaged angle values over codebook","ang.png",2)

if __name__ == '__main__':
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
#    km_mag = make_codebook(mag, 150)
#    km_ang = make_codebook(ang, 150)
    
    # Save to disk, if training is performed
#    print("Writing the KMeans models to disk...")
#    with open(km_file_mag, "wb") as outfile:
#        pickle.dump(km_mag, outfile)
#    with open(km_file_ang, "wb") as outfile:
#        pickle.dump(km_ang, outfile)
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
    bgthresh = 20000 # should be <=90k, to prevent error vq(mag, clusters_mag[0])
    # 
    target_file = os.path.join(destpath, flow_filename+"_val_BG"+str(bgthresh)+".pkl")
#    features_val = extract_flow_val(DATASET, bgthresh, grid_size=gdsize, partition="validation")
#    with open(target_file, "wb") as outfile:
#        pickle.dump(features_val, outfile)

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

    line_graph(df_train_mag, df_train_ang, df_train_mag.index, 150)


