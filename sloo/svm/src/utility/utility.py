import numpy as np
import pandas as pd
from numpy import loadtxt, unique

# Convert csv to pandas csv
def csv_to_pandas(trainName, testName):

	# Loding Data from SLOO Files
	trainData = loadtxt(trainName, skiprows=1)
	testData = loadtxt(testName, skiprows=1)
	
	# Convert to numpy datastructure
	trainData = np.array([x[1:trainData.shape[1]] for x in trainData])
	testData = np.array([x[1:testData.shape[1]] for x in testData])

	# Saving Regular Dataset as Pandas DataFrame format
	print("Saving CSV to Pandas CSV........")
	trainData = pd.DataFrame(trainData)
	testData = pd.DataFrame(testData)
	trainData.to_csv("../../dataset/pandas/train.csv")
	testData.to_csv("../../dataset/pandas/test.csv")
	print("Done Save!")

# Convert txt to pandas csv
def data_to_pandas(train_results=None, test_results=None):
	# Saving Results to Pandas DataFrame csv
	print("Saving Python Data to CSV File..........")
	train_results = pd.DataFrame(train_results)
	test_results = pd.DataFrame(test_results)
	train_results.to_csv("../../dataset/patient_type/train.csv")
	test_results.to_csv("../../dataset/patient_type/test.csv")
	print("Done Save!")

# Extracting Parkinson's assesment results from training and testing
def patient_type(dataset):
	count = 0 # To keep track of patient number
	uniques = unique(dataset[:,0])
	data_results = []	
	for i in uniques:
		prevPatient = None
		for data in dataset:
			if(i == data[0] and prevPatient != data[0]):
				result = data[(dataset.shape[1]-1)]
				data_results.append(result)
				prevPatient = data[0]

	return data_results
