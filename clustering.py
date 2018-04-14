import numpy as np
import pickle
from sklearn.cluster import KMeans

# function to extract magnitude space vectors and angle space vectors
# It does not return any labeling information. Hence used only for finding 
# cluster centres.
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


# function to find the clusters using KMeans
# vecs: any dataframe representing the input space points
# nclusters: No. of clusters to be formed
# returns the KMeans object, containing the 
def make_codebook(vecs, nclusters):
#    pickle.dump(train_keypoints, open(os.path.join('data',target_file), "wb"))
    print("Clustering using KMeans: Input size -> {} :: n_clusters -> {}"\
          .format(vecs.shape, nclusters))   
    
    #train_features = pickle.load(open(keypoints_path, "rb"))
    #clustering with k-means
    #kmeans = KMeans(init='k-means++', n_clusters=200, n_init=10, n_jobs=2, verbose=1)
    kmeans = KMeans(n_clusters=nclusters, n_init=10, n_jobs=2)
    kmeans.fit(vecs)
    print("Done Clustering!")
    return kmeans
    
