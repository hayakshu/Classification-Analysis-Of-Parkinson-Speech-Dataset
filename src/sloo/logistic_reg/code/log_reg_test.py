from sklearn.svm import SVC
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef


def lr(trainName, trainPatientTypeName, testName, testPatientTypeName):
    # Load data from files by converting to Pandas DataFrame
    trainData = pd.read_csv(trainName)
    trainPatientType = pd.read_csv(trainPatientTypeName)
    testData = pd.read_csv(testName)
    testPatientType = pd.read_csv(testPatientTypeName)

    # Remove header and patient index numbers from Pandas DataFrame
    # and convert to Numpy Array
    trainData = trainData.iloc[:,1:]
    trainData = np.array(trainData)
    trainPatientType = trainPatientType.iloc[:,1:]
    trainPatientType = np.array(trainPatientType).ravel()

    testData = testData.iloc[:,1:]
    testData = np.array(testData)
    testPatientType = testPatientType.iloc[:,1:]
    testPatientType = np.array(testPatientType).ravel()
 
    scaler = sklearn.preprocessing.StandardScaler()
    trainData=scaler.fit_transform(trainData)
    testData=scaler.transform(testData)
    
    print("Training Model...")
    clf = LogisticRegression(penalty='l2', C= 0.5)

    clf.fit(trainData, trainPatientType)
    predictions = clf.predict(testData)
    print clf.coef_
    
    print("Done!")
    print("Calculating Accuracy...")
    TP, FN, TN, FP = 0, 0, 0, 0
    for i, prediction in enumerate(predictions):
        if(prediction == 1 and testPatientType[i] == 1):
            TP += 1
        elif(prediction == 0 and testPatientType[i] == 1):
            FN += 1
        elif(prediction == 0 and testPatientType[i] == 0):
            TN += 1
        elif(prediction == 1 and testPatientType[i] == 0):
            FP += 1
        else:
            pass
    # testPatientType
    # predictions
    Accuracy = (TP + TN)/float(TP + TN + FP + FN)
    print "Accuracy of Prediction {} ".format(Accuracy) 
    
    CM=confusion_matrix(predictions, testPatientType)
    print (classification_report(predictions,testPatientType))
    print "Confusion matrix"
    print CM
    print predictions
    print testPatientType

    #specifity=(float(TN)/(TN+FP))*100
    #sensitivity=(float(TP)/(TP+FN))*100

    #print "Specifity is %f"%specifity
    #print "Sensitivity is %f"%sensitivity
    print "Matthews Correlation Coefficient"
    print(matthews_corrcoef(predictions,testPatientType))
    

print("Predicting Logistic Regression")

trainName = "train.csv"
trainPatientTypeName = "type_train.csv"
testName = "test.csv"
testPatientTypeName = "type_test.csv"

lr(trainName, trainPatientTypeName, testName, testPatientTypeName)





