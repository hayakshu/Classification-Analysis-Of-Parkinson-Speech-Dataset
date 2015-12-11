import numpy as np
import pandas as pd
from numpy import loadtxt, unique
from sklearn.decomposition import PCA

# PCA Script for SLOO Dataset
def pca(trainName, testName):

	# Loading Data from SLOO Files
	trainData = loadtxt(trainName, skiprows=1)
	testData = loadtxt(testName, skiprows=1)

	# Convert to numpy datastructure
	trainData = np.array([x[1:trainData.shape[1]] for x in trainData])
	testData = np.array([x[1:testData.shape[1]] for x in testData])

	# Dimensionality reduction using pca
	n_component = 90
	print("Performing PCA Analysis")
	pca = PCA(n_components=n_component)
	pca.fit(trainData)
	trainData_pca = pca.transform(trainData)
	testData_pca = pca.transform(testData)

	# Saving PCA Performance to object
	print("Saving PCA Data to csv")
	trainData_pca = pd.DataFrame(trainData_pca)
	testData_pca = pd.DataFrame(testData_pca)
	trainData_pca.to_csv("../../dataset/pca/train.csv")
	testData_pca.to_csv("../../dataset/pca/test.csv")





