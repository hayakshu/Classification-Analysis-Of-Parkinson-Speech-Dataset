import numpy as np
import pandas as pd
from numpy import loadtxt, unique

# Script for extracting expected patient type into seperate array

# For extracting unqiue patient type from training and testing
def patient_type(dataset, unique):
	count = 0 # To keep track of patient number
	data_results = []	
	for i in unique:
		prevPatient = None
		for data in dataset:
			if(i == data[0] and prevPatient != data[0]):
				result = data[(dataset.shape[1]-1)]
				data_results.append(result)
				prevPatient = data[0]

	return data_results

trainName = "../dataset/train_data.txt"
testName = "../dataset/test_data.txt"

trainData = loadtxt(trainName, delimiter=",")
testData = loadtxt(testName, delimiter=",")

trainData = np.array(trainData)
testData = np.array(testData)

# Extracting number of patients
n_patientsTrain = unique(trainData[:,0])
n_patientsTest = unique(testData[:,0])

# Extracting Data if each patient has Parkinsons or Not
train_results = patient_type(trainData, n_patientsTrain)
test_results = patient_type(testData, n_patientsTest)

# Saving Results through Pandas DataFrame
train_results = pd.DataFrame(train_results)
test_results = pd.DataFrame(test_results)
print("Saving Results Data to CSV File")
train_results.to_csv("../dataset/train_patientType.csv")
test_results.to_csv("../dataset/test_patientType.csv")