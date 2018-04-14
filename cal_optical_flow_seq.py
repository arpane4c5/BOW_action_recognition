import os
import cv2
import sys
import numpy as np
from utils import read_seq

# function to extract optical flow based grid features from training set
# and save the features in a dictionary on disk.
# applicable only for the training set
def extract_flow_seq_train(dataset_base, grid_size=10):
    seq_path = os.path.join(dataset_base,'kth_sequences.txt')
    dataset_path = os.path.join(dataset_base, 'kth_actions_train')
    filenames = os.listdir(dataset_path)
        
    #farneback_params = dict(winsize = 20, iterations=1,
    #    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
    #    pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)
    n_processed_files = 0
    seq = read_seq(seq_path)
    features = {}  # save features in this dictionary
    for filename in filenames:
        filepath = os.path.join(dataset_path, filename)
        ##print(filepath)
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            continue
        dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #ret,prev_frame = cap.read()
        #prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

        # Store features in current file.
        features_current_file = []
        start_frames = []
        end_frames = []
        # Get the action sequences from the sequences dictionary.
        key = filename.rsplit('_',1)[0]
        action_seq_str = seq[key]
        for marker in action_seq_str.split(','):
            temp = marker.split('-')        # ' 120-190'
            start_frames.append(int(temp[0]))   # 120
            end_frames.append(int(temp[1]))
        
        # Iterate over the sequences to get the optical flow features.
        for i, stime in enumerate(start_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, stime)
            ret, prev_frame = cap.read()

            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            stime += 1
       
            while(cap.isOpened() and stime <= end_frames[i]):
                ret, frame = cap.read()
                if not ret:
                    break
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
                flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                # stack sliced arrays along the first axis (2, 12, 16)
                sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                        ang[::grid_size, ::grid_size]), axis=0)
                
                #feature.append(sliced_flow[..., 0].ravel())
                #feature.append(sliced_flow[..., 1].ravel())
                #feature = np.array(feature)
                features_current_file.append(sliced_flow)
                stime +=1
                prev_frame = curr_frame
        cap.release()
        cv2.destroyAllWindows()
        features[filename] = features_current_file
        n_processed_files += 1
        print("Done {} files : {}".format( str(n_processed_files), filename))
    
    return features
            
# function to extract optical flow based grid features from validation/test set
# Consider Background subtraction, using BGThreshold
def extract_flow_val(dataset_base, bgThresh, grid_size=10, partition="validation"):
    if partition == 'validation':
        dataset_path = os.path.join(dataset_base, 'kth_actions_validation')
    elif partition == 'testing':
        dataset_path = os.path.join(dataset_base, 'kth_actions_test')
    else:
        print("Invalid partition name! Abort! ")
        sys.exit(0)
        
    filenames = os.listdir(dataset_path)
        
    #farneback_params = dict(winsize = 20, iterations=1,
    #    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
    #    pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)
    n_processed_files = 0
    
    features = {}
    for filename in filenames:
        filepath = os.path.join(dataset_path, filename)
        ##print(filepath)
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            continue
        dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #ret,prev_frame = cap.read()
        #prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

        # Store features in current file.
        features_current_file = []
        start_frames = [0]
        end_frames = [nframes]
        
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
                    stime += 1
                    continue
        
                flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                #feature = []
                # stack sliced arrays along the first axis (2, 12, 16)
                sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                        ang[::grid_size, ::grid_size]), axis=0)
                
                features_current_file.append(sliced_flow)
                stime = stime + 1
                prev_frame = curr_frame
        cap.release()
        cv2.destroyAllWindows()
        features[filename] = features_current_file

        n_processed_files += 1
        #if n_processed_files % 30 == 0:
        print("Done {} files : {}".format( str(n_processed_files), filename))
    
    return features
