import argparse
import numpy as np
import os
import pickle
import cv2
import numpy as np
import os
import pickle

def extract_optical_flow(path,target_file) :
    data_path = path
    filenames = os.listdir(data_path)
    os.makedirs("data", exist_ok=True)
    farneback_params = dict(winsize = 20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)
    n_processed_files = 0
    
    features = []
    for filename in filenames:
            filepath = os.path.join(data_path,filename)
            ##print(filepath)
            vid = cv2.VideoCapture(filepath)
            
            ret,prev_frame = vid.read()
            prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(prev_frame)
            hsv[...,1] = 255

            # Store features in current file.
            features_current_file = []
            
            while(True):
                #print("hello 1")
                ret, frame = vid.read()
                if not ret:
                    break
                #print("hello 2")
                # Only care about gray scale.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flows = cv2.calcOpticalFlowFarneback(prev_frame,frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                feature = []
                for r in range(120):
                    if r % 10 != 0:
                        continue
                    for c in range(160):
                        if c % 10 != 0:
                            continue
                        feature.append(flows[r,c,0])
                        feature.append(flows[r,c,1])
                feature = np.array(feature)
                #print("hello")
                features_current_file.append(feature)

                prev_frame = frame
