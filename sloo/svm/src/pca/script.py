from pca import *

# ---------------------------------------------- #
# SCRIPT TO RUN PCA ON SLOO  					 #
# ---------------------------------------------- #

# Filepath to SLOO Data
trainName = "../../dataset/sloo/train.csv"
testName = "../../dataset/sloo/test.csv"
pca(trainName, testName)