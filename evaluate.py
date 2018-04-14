from sklearn.svm import SVC
import numpy as np

#def evaluate(Xtrain, Ytrain, Xval, Yval):
#    print("Training with SVM")
#    clf = SVC(1,kernel="linear",verbose=True)
#    clf.fit(Xtrain, Ytrain)
#    confusion_matrix = np.zeros((6,6))
#    
#    pred = clf.predict(Xval)
#    
#    correct = 0
#    
#    for i in range(len(Yval)):
#        if pred[i] == Yval[i]:
#            correct +=1
#        confusion_matrix[pred[i],Yval[i]] +=1
#    print("%d/%d Correct" % (correct, len(pred)))
#    print("Accuracy =", correct / len(pred))
#
#    print("Confusion matrix")
#    print(confusion_matrix)

# function to make predictions on the validation set and evaluate the results
# Xval is the features dataframe (nvideos, 50), Yval is the labels dataframe 
def evaluate(clf, Xval, Yval):
    print("Evaluate on validation set")
    confusion_matrix = np.zeros((6,6))
    pred = clf.predict(Xval)
    
    correct = 0
    
    for i in range(len(Yval)):
        if pred[i] == int(Yval.iloc[i]):
            correct +=1
        confusion_matrix[pred[i],int(Yval.iloc[i])] +=1
    print("%d/%d Correct" % (correct, len(pred)))
    print("Accuracy = {} ".format( float(correct) / len(pred) ))
    print("Confusion matrix")
    print(confusion_matrix)

