from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from numpy import loadtxt
import numpy as np
import pandas as pd

def logreg(X, Y):

    pen = ['l1','l2','elasticnet']     #parameter for SGDC
    alpha = [0.1,0.01,0.001,0.0001,0.00001,0.5,0.05,0.005,0.0005,0.00005] #parameter for SGDC
    n_iter = [5,10,15,20,25,30,40,50,100]
    loss_fun=['log']

    parameters = [{'loss': loss_fun, 'alpha': alpha,'n_iter': n_iter,'penalty' :pen}]
    dataLength = len(X)
    log_reg = SGDClassifier('penalty: 'l2', 'alpha': 1e-05, 'n_iter': 50, 'loss': 'log')
    lv = LeaveOneOut(dataLength)
    clf = GridSearchCV(log_reg, parameters, cv= lv)
    clf.fit(X, Y)
    print "Best Parameters: {}".format(clf.best_params_)
    print "Accuracy: {}".format(clf.best_score_)
    
    
print "Performing Cross Validation on logistic regression SLOO Data"

trainName = "train.csv"
trainPatientTypeName = "type_train.csv"

# Load data from files by converting to Pandas DataFrame
trainData = pd.read_csv(trainName)
trainPatientType = pd.read_csv(trainPatientTypeName)

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()


logreg(trainData, trainPatientType)

