
from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from numpy import loadtxt
import numpy as np
import pandas as pd

def knn(X, Y):

    neighbors = [1,2,3,4,5,6,7,8,9,10]
    weights = ['distance']
    n_components = [5,10,15,20,25,30]

    parameters = [{'k_nn__n_neighbors': neighbors, 'k_nn__weights': weights,'pca__n_components':n_components}]
    
    
    dataLength = len(X)
    lv = LeaveOneOut(dataLength)
    
    pipeline = Pipeline([
    ('pca', PCA()),
    ('k_nn', KNeighborsClassifier()),])  
              
            
    clf = GridSearchCV(pipeline, parameters, cv= lv)
    clf.fit(X, Y)
    print "Best Parameters: {}".format(clf.best_params_)
    print "Accuracy: {}".format(clf.best_score_)
    
    
print "Performing Cross Validation on knn SLOO Data"

trainName = "train.csv"
trainPatientTypeName = "type_train.csv"


trainData = pd.read_csv(trainName)
trainPatientType = pd.read_csv(trainPatientTypeName)
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()


knn(trainData, trainPatientType)

