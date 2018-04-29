import numpy as np
import csv

# function to read the kth_sequences.txt file and return the dictionary of 
# frame sequences where actions occur.
# returns d: dictionary eg {'filename': '1-10, 20-30, 50-100'}
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


# function to return the label (int) given the name of the video
def get_video_label(srcVid):
    if "boxing" in srcVid:
        return 0
    elif "handclapping" in srcVid:
        return 1
    elif "handwaving" in srcVid:
        return 2
    elif "jogging" in srcVid:
        return 3
    elif "running" in srcVid:
        return 4
    elif "walking" in srcVid:
        return 5

# function to make predictions on the validation set and evaluate the results
# Xval is the features dataframe (nvideos, 50), Yval is the labels dataframe 
def evaluate(clf, Xval, Yval, partition="validation"):
    print("Evaluate on "+partition+" set")
    confusion_matrix = np.zeros((6,6))
    pred = clf.predict(Xval)    # predict using the trained model
    correct = 0
    for i in range(len(Yval)):
        if pred[i] == int(Yval.iloc[i]):
            correct +=1
        confusion_matrix[pred[i],int(Yval.iloc[i])] +=1
    print("%d/%d Correct" % (correct, len(pred)))
    print("Accuracy = {} ".format( float(correct) / len(pred) ))
    print("Confusion matrix")
    print(confusion_matrix)
