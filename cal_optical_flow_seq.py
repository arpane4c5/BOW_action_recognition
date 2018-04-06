import argparse
import numpy as np
import os
import pickle
import cv2
import numpy as np
import os
import pickle

'''def get_optical_flow_vid(srcVideo, action_seq_str, X_list, imgPrefix):
    
    flow_means_x = []
    flow_means_y = []
    
    cap = cv2.VideoCapture(srcVideo)
    if not cap.isOpened():
        print ("Error in reading the video file ! Abort !")
        sys.exit(0)
        
        
    start_frames = []
    end_frames = []
    for marker in action_seq_str.split(','):
        temp = marker.split('-')        # ' 120-190'
        start_frames.append(int(temp[0]))   # 120
        end_frames.append(int(temp[1]))
    # sanity check condition
    if len(start_frames)!=len(end_frames):
        print "Error in reading the frame markers from file ! Abort !"
        sys.exit(0)
        
#     start_frames = [0]
#     end_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
  
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
   
    print ("Start Times : "+str(start_frames))
    print ("End Times   : "+str(end_frames))

    curr_label = get_video_label(srcVideo)
    print(curr_label)
    
    for i, stime in enumerate(start_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, stime)
        ret, prev_frame = cap.read()

        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        stime = stime + 1
       
        while(cap.isOpened() and stime <= end_frames[i]):
            ret, frame = cap.read()
            
            if not ret:
                break
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #print ("For frames: ("+str(stime-1)+","+str(stime)+") :: shape : "+str(flow.shape))
            
            
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

                prev_frame = curr_frame

            features.append({
                "filename": filename,
                "features": features_current_file 
            })

            n_processed_files += 1
            if n_processed_files % 30 == 0:
                print("Done %d files" % n_processed_files)
            
            
            
            
    
            X_list.append(flow)  
            stime = stime + 1
            prev_frame = curr_frame
            
 
    cap.release()            
    cv2.destroyAllWindows()
       
    return curr_label, start_frames,end_frames, X_list




   for video in videosList:
    X_list = []
    key = video.rsplit('_',1)[0]
    print(key)
    count = 0
    action_seq_str = seq[key]
    print(action_seq_str)
    curr_label, start_frames,end_frames, X_list= get_optical_flow_vid(os.path.join(srcVideoFolder, video), action_seq_str, X_list, key)
    print(np.array(X_list).shape)'''
    
    
    '''print(curr_label[0])
    Zlist.append(X_list)
    Zlabel.append(curr_label[0])'''
    
    


def extract_optical_flow(path,target_file) :
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
            print "Error in reading the frame markers from file ! Abort !"
            sys.exit(0)
            
            dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for i, stime in enumerate(start_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, stime)
                ret, prev_frame = cap.read()

                prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                stime = stime + 1
       
                while(cap.isOpened() and stime <= end_frames[i]):
                    ret, frame = cap.read()
            
                    if not ret:
                        break
                    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                    flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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
            
            
            
           ''' while(True):
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

                

            features.append({
                "filename": filename,
                "features": features_current_file 
            })

            n_processed_files += 1
            if n_processed_files % 30 == 0:
                print("Done %d files" % n_processed_files)
                
    pickle.dump(features, open(os.path.join('data',target_file), "wb"))'''
