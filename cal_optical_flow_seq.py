import argparse
import numpy as np
import os
import pickle
import cv2
import numpy as np
import os
import pickle
import csv
import cv2




    
    
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

def extract_optical_flow_seq(path,target_file) :
    data_path = path
    filenames = os.listdir(data_path)
    os.makedirs("data", exist_ok=True)
    farneback_params = dict(winsize = 20, iterations=1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
        pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)
    n_processed_files = 0
    
    seq_path = os.path.join('/opt/datasets/KTH','kth_sequences.txt')
    seq = read_seq(seq_path)
    
    features = []
    for filename in filenames:
            bgThresh = 105000
            filepath = os.path.join(data_path,filename)
            ##print(filepath)
            cap = cv2.VideoCapture(filepath)
            
            ret,prev_frame = cap.read()
            prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(prev_frame)
            hsv[...,1] = 255
            
            key = filename.rsplit('_',1)[0]
            action_seq_str = seq[key]
            # Store features in current file.
            features_current_file = []
            start_frames = []
            end_frames = []
            for marker in action_seq_str.split(','):
                temp = marker.split('-')        # ' 120-190'
                start_frames.append(int(temp[0]))   # 120
                end_frames.append(int(temp[1]))
            # sanity check condition
            if len(start_frames)!=len(end_frames):
                sys.exit(0)
            ##print("Error in reading the frame markers from file ! Abort !")
            
            
            dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for i, stime in enumerate(start_frames):
                fgbg = cv2.createBackgroundSubtractorMOG2()
                cap.set(cv2.CAP_PROP_POS_FRAMES, stime)
                ret, prev_frame = cap.read()
                fgmask = fgbg.apply(prev_frame)

                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                stime = stime + 1
       
                while(cap.isOpened() and stime <= end_frames[i]):
                    ret, frame = cap.read()
            
                    if not ret:
                        break
                    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # To find the background mask and skip the frame if foreground is absent
                    fgmask = fgbg.apply(frame)
                    if np.sum(fgmask)<bgThresh:
                        #print ("BG frame skipped !!")
                        prev_frame = curr_frame
                        continue
            
                    flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    
                    
                    feature = []
                    for r in range(120):
                        if r % 10 != 0:
                            continue
                        for c in range(160):
                            if c % 10 != 0:
                                continue
                            feature.append(flow[r,c,0])
                            feature.append(flow[r,c,1])
                    feature = np.array(feature)
                    #print(feature.shape)
                    features_current_file.append(feature)
                    stime = stime + 1
                    prev_frame = curr_frame
            cap.release()            
            cv2.destroyAllWindows()
            features.append({
                "filename": filename,
                "features": features_current_file 
            })

            n_processed_files += 1
            if n_processed_files % 30 == 0:
                print("Done %d files" % n_processed_files)
    pickle.dump(features, open(os.path.join('data',target_file), "wb"))
            
            

