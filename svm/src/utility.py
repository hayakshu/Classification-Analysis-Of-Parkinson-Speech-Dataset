import numpy as np

# Fix so we split results from training set
def load_training():
	x_train = []
	y_train = []
	with open('../dataset/train_data.txt') as train:
		for row in train:
			tple = eval(row)
			x_train.append(list(tple[:(len(tple)-1)]))
			y_train.append(tple[(len(tple)-1):][0])
	
	return (np.array(x_train), np.array(y_train))

def load_test():
	test_set = []
	with open('../dataset/test_data.txt') as test:
		for row in test:
			tple = eval(row)
			test_set.append(list(tple[:(len(tple))]))
	
	return np.array(test_set)