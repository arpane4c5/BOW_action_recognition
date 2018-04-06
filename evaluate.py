from sklearn.svm import SVC
import numpy as np
import os
import pickle

def evaluate(Xtrain,Ytrain,Xval,Yval):
    print("training with SVM")
    clf = SVC(1,kernel="linear",verbose=True)
    clf.fit(Xtrain,Ytrain)
    confusion_matrix = np.zeros((6,6))

    pred = clf.predict(Xval)


    correct = 0

    for i in range(len(Y)):
        if pred[i] == Y[i]:
            correct +=1
        confusion_matrix[pred[i],Y[i]] +=1
    print("%d/%d Correct" % (correct, len(pred)))
    print("Accuracy =", correct / len(pred))

    print("Confusion matrix")
    print(confusion_matrix)
