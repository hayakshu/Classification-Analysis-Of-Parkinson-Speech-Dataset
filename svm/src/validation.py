from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
from sklearn.svm import SVC
from numpy import loadtxt
from confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------- #
# SCRIPT FOR CROSS VALIDATION ON SLOO DATASETS
# ---------------------------------------------- #

# Cross Validation Script on SLOO Data to determine 
# Linear Kernal Cross Validation 
def linear(X, Y, title, filename):

	C = [1,2,5,10,15,20,25,30,50,100,200,500,1000,2000,5000,10000]
	dataLength = len(X)
	
	loo = LeaveOneOut(dataLength)
	avg_Accuracy = dict()
	sensitivity = dict()
	specificity = dict()
	for c in C:
		#print "Performing Cross Validation on Penalty: {}".format(c)
		predictions = []
		expected = []
		TP, FN, TN, FP = 0, 0, 0, 0
		Accuracy = 0
		for train_index, test_index in loo:
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index][0]

			clf = SVC(C=c, kernel='linear')
			clf.fit(X_train, Y_train)
			prediction = clf.predict(X_test)[0]
			#print("Prediction: {}".format(prediction))
			#print("Expected Result: {}".format(Y_test))
			predictions.append(prediction)
			expected.append(Y_test)

		#print("Calculating Accuracy of Prediction")
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
		Sensitivity = TP/float(TP + FN)
		Specificity = TN/float(TN + FP)
		Accuracy = (TP + TN)/float(TP + TN + FP + FN)
		#print("Accuracy of Prediction: {} @ Penalty: {}".format(Accuracy, c))
		avg_Accuracy[c] = Accuracy
		sensitivity[c] = Sensitivity
		specificity[c] = Specificity

	bestC = max(avg_Accuracy.iterkeys(), key=(lambda k: avg_Accuracy[k]))
	# We are hashing the Specificity and Sensitivity based on the key that gives best accuracy
	bestSensitivity = sensitivity[bestC]
	bestSpecificity = specificity[bestC]
	bestAccuracy = avg_Accuracy[bestC]

	with open(filename, 'ab') as f:

		f.write("All Accuracy Values @ Each Penalty: {} \n".format(avg_Accuracy))
		f.write("Most Accurate Penalty Value: {}\n".format(bestC))
		f.write("Accuracy of Prediction: {} @ Penalty: {}\n".format(bestAccuracy, c))
		f.write("Sensitivity of Prediction: {} @ Penalty: {}\n".format(bestSensitivity, c))
		f.write("Specificity of Prediction: {} @ Penalty: {}\n".format(bestSpecificity, c))
		f.write("Matthews Correlation Coeefficient Value: {}\n".format(matthews_corrcoef(predictions, expected)))
		f.write("Classification Report: \n")
		f.write(classification_report(predictions, expected))
		f.write("Confusion Matrix\n")
		cm = confusion_matrix(predictions, expected)
		f.write(str(cm))
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		label1 = "Negative"
		label2 = "Positive"
		
		plt.figure()
		plot_confusion_matrix(cm, title, label1, label2)
	

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
	print("Best Params for RBF: {}".format(clf.best_params_))
	print("Accuracy: {}".format(clf.best_score_))
	return clf.best_params_

# Reports results for validation on RBF: calculates the Sensitiviy and Specificity of Best Hyperparameters from validation set
def rbf_analysis(X, Y, c, g, title, filename):

	print "Performing Cross Validation on Penalty: {}".format(c)
	dataLength = len(X)
	loo = LeaveOneOut(dataLength)
	predictions = []
	expected = []
	TP, FN, TN, FP = 0, 0, 0, 0
	Accuracy = 0
	for train_index, test_index in loo:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index][0]

		clf = SVC(C=c, gamma=g, kernel='rbf')
		clf.fit(X_train, Y_train)
		prediction = clf.predict(X_test)[0]
	
		predictions.append(prediction)
		expected.append(Y_test)

	print("Calculating.....")
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

	Sensitivity = TP/float(TP + FN)
	Specificity = TN/float(TN + FP)
	Accuracy = (TP + TN)/float(TP + TN + FP + FN)

	# Saving data to file
	with open(filename, 'ab') as f:
		f.write("Sensitivity of Prediction: {} @ Penalty: {} @ Gamma: {}\n".format(Sensitivity, c, g))
		f.write("Specificity of Prediction: {} @ Penalty: {} @ Gamma: {}\n".format(Specificity, c, g))
		f.write("Accuracy of Prediction: {} @ Penalty: {} @ Gamma: {}\n".format(Accuracy, c, g))
		f.write("Matthews Correlation Coeefficient Value: {}\n".format(matthews_corrcoef(predictions, expected)))
		f.write("Classification Report:\n")
		f.write(classification_report(predictions, expected))
		f.write("Confusion Matrix\n")
		cm = confusion_matrix(predictions, expected)
		f.write(str(cm))
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		label1 = "Negative"
		label2 = "Positive"
			
		plt.figure()
		plot_confusion_matrix(cm, title, label1, label2)
		


# ------------------------------------------ #
# PCA DATA VALIDATION						 #
# ------------------------------------------ #

# Cross Validation for pca
# File path for training set - preprocessed via PCA and not
print("Performing Cross Validation on PCA SLOO Data")
trainName = "../dataset/pca/train.csv"
trainPatientTypeName = "../dataset/patient_type/train.csv"

# Load data from files by converting to Pandas DataFrame
trainData = pd.read_csv(trainName)
trainPatientType = pd.read_csv(trainPatientTypeName)

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()

print("\nPerforming Linear SVM Cross-Validation...")
title="Validation Results For PCA Linear SVM" 
linear(trainData, trainPatientType, title, "../class_report/pca/linear.txt")

print("\nPerforming RBF SVM Cross-Validation...")
hyperparameters = rbf(trainData, trainPatientType)
c = hyperparameters['C']
gamma = hyperparameters['gamma']
title="Validation Results For PCA RBF SVM" 
rbf_analysis(trainData, trainPatientType, c, gamma, title, "../class_report/pca/rbf.txt")


# ------------------------------------------ #
# NON-PCA DATA VALIDATION					 #
# ------------------------------------------ #
print("\nPerforming Cross Validation on regular SLOO Data")
# Cross Validation on Non PCA Method
trainName = "../dataset/pandas/train.csv"
trainData = pd.read_csv(trainName)

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)

print("\nPerforming Linear SVM Cross-Validation...")
title="Validation Results For NON-PCA Linear SVM" 
linear(trainData, trainPatientType, title, "../class_report/no_pca/linear.txt")

print("\nPerforming RBF SVM Cross-Validation...")
hyperparameters = rbf(trainData, trainPatientType)
c = hyperparameters['C']
gamma = hyperparameters['gamma']
title="Validation Results For NON-PCA RBF SVM" 
rbf_analysis(trainData, trainPatientType, c, gamma, title, "../class_report/no_pca/rbf.txt")