from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from numpy import loadtxt
import numpy as np
import pandas as pd

# ---------------------------------------------- #
# SCRIPT FOR CROSS VALIDATION ON SLOO DATASETS
# ---------------------------------------------- #

# Cross Validation Script on SLOO Data to determine 
# Linear Kernal Cross Validation 
def linear(X, Y):

	C = [1,2,5,10,15,20,25,30,50,100,200,500,1000,2000,5000,10000]
	dataLength = len(X)
	print dataLength
	loo = LeaveOneOut(dataLength)
	avg_Accuracy = dict()
	for c in C:
		print "Performing Cross Validation on Penalty: {}".format(c)
		predictions = []
		expected = []
		TP, FN, TN, FP = 0, 0, 0, 0
		Accuracy = 0
		for train_index, test_index in loo:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			clf = SVC(C=c, kernel='linear')
			clf.fit(X_train, Y_train)
			prediction = clf.predict(X_test)[0]
			print("Prediction: {}".format(prediction))
			print("Expected Result: {}".format(Y_test))
			predictions.append(prediction)
			expected.append(Y_test)

		print("Calculating Accuracy of Prediction")
		for i, prediction in enumerate(predictions):
			if(prediction == 1 and expected[i] == 1):
				TP += 1
			elif(prediction == 0 and expected[i] == 1):
				FN += 1
			elif(prediction == 0 and expected[i] == 0):
				TN += 1
			elif(prediction == 1 and expected[i] == 0):
				FP += 1
			else:
				pass
		Accuracy = (TP + TN)/float(TP + TN + FP + FN)
		print("Accuracy of Prediction: {} @ Penalty: {}".format(Accuracy, c))
		avg_Accuracy[c] = Accuracy

	print(avg_Accuracy)

	bestC = max(avg_Accuracy.iterkeys(), key=(lambda k: avg_Accuracy[k]))
	bestAccuracy = avg_Accuracy[bestC]
	print("Most Accurate Penalty Value: {}, Accuracy: {}".format(bestC, bestAccuracy))

def rbf(X, Y):
	# Performing Grid Search for Parameter Selection
	C = [1,2,5,10,15,20,25,30,50,100,200,500,1000,2000,5000,10000]
	gamma = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.5,0.05,0.005,0.0005,0.00005,0,000005]
	parameters = [{'kernel': ['rbf'], 'gamma': gamma,'C': C}]
	dataLength = len(X)
	svm = SVC()
	lv = LeaveOneOut(dataLength)
	clf = GridSearchCV(svm, parameters, cv= lv)
	clf.fit(X, Y)
	print(clf.best_params_)
	print(clf.best_score_)

# Cross Validation for pca
# File path for training set - preprocessed via PCA and not
trainName = "../dataset/pca_train.csv"
trainPatientTypeName = "../dataset/train_patientType.csv"

# Load data from files by converting to Pandas DataFrame
trainData = pd.read_csv(trainName)
trainPatientType = pd.read_csv(trainPatientTypeName)

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()

linear(trainData, trainPatientType)
rbf(trainData, trainPatientType)

# Cross Validation on Non PCA Method
trainName = "../dataset/train.csv"

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()

linear(trainData, trainPatientType)
rbf(trainData, trainPatientType)