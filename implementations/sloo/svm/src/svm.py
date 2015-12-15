from sklearn.svm import SVC
import numpy as np
import pandas as pd


def svm(trainName, trainPatientTypeName, testName, testPatientTypeName, c=1, k="linear", g='auto'):
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

	print("Training Model...")
	clf = SVC(kernel=k, C=c, gamma=g)
	clf.fit(trainData, trainPatientType)
	predictions = clf.predict(testData)
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
	return "Accuracy of Prediction {}: {}\n".format(k.upper(),Accuracy)


with open('../svm_test_prediction_results/pca.txt', 'ab') as f: 
	f.write("Predicting PCA Data\n")
	# File path for training set - preprocessed via PCA and not
	trainName = "../dataset/pca/train.csv"
	trainPatientTypeName = "../dataset/patient_type/train.csv"
	testName = "../dataset/pca/test.csv"
	testPatientTypeName = "../dataset/patient_type/test.csv"
	linear = svm(trainName, trainPatientTypeName, testName, testPatientTypeName, 1)
	rbf = svm(trainName, trainPatientTypeName, testName, testPatientTypeName, 100,"rbf",1e-06)
	f.write(linear)
	f.write(rbf)
	print("Done Predicting PCA Data\n")

with open('../svm_test_prediction_results/no_pca.txt', 'ab') as f:
	f.write("Predicting NON-PCA Data\n")
	trainName = "../dataset/pandas/train.csv"
	testName = "../dataset/pandas/test.csv"
	linear = svm(trainName, trainPatientTypeName, testName, testPatientTypeName, 1)
	rbf = svm(trainName, trainPatientTypeName, testName, testPatientTypeName, 100, "rbf",1e-06)
	f.write(linear)
	f.write(rbf)
	print("Done Predicting NON-PCA Data\n")



