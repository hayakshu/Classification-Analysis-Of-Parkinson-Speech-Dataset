from pca import *

# ---------------------------------------------- #
# SCRIPT TO RUN PCA ON SLOO  					 #
# ---------------------------------------------- #

# Filepath to SLOO Data
trainName = "../dataset/second_dataset_train.csv"
testName = "../dataset/second_dataset_test.csv"
pca(trainName, testName)