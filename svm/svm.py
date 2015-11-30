from utility import *
from sklearn import svm
from pdb import set_trace as bp
from numpy import unique, count_nonzero
from sklearn.svm import LinearSVC
from sklearn import linear_model
import numpy as np

train_x, train_y = load_training()
test_set = load_test()
size = len(train_x)
train_x = np.array([row[1:27] for row in train_x])


# Prediction for each patient
num_patients = unique(test_set[:,0])
patient_predictions = dict()
patient_validations = dict()

print("Start of making predictions")
for i in num_patients:
	
	patient_set = []
	patient_val = []
	for row in test_set:
		if row[0] == i:
			patient_set.append(row[1:27])
			patient_val.append(row[27])
	
	# Training our model on test set
	print("Training Model for SVM")
	#clf = linear_model.SGDClassifier().fit(train_x,train_y)
	#clf = LinearSVC().fit(train_x,train_y)
	clf = svm.SVC().fit(train_x, train_y)

	predictions = clf.predict(patient_set)
	parkinson_prediction = count_nonzero(predictions)
	neg_predictions = len(predictions) - parkinson_prediction
	
	# Predicting if he or she has Parkinsons
	print predictions
	if neg_predictions > parkinson_prediction:
		print("Patient {} does not have Parkinsons".format(i))
		prediction = 0
	else:
		prediction = 1
		print("Patient {} does have Parkinsons".format(i))
	patient_predictions[i] = prediction
	
	# Check for most frequent result in validation set
	parkinson_validation = count_nonzero(patient_val)
	neg_validation = len(patient_val) - parkinson_validation
	if neg_validation > parkinson_validation:
		validation = 0
	else:
		validation = 1
	patient_validations[i] = validation

print("Done Prediction \n")
print("Start of Validation \n")

total = len(patient_predictions)
result = 0
for key, val in patient_predictions.iteritems():
	if patient_validations[i] == val:
		result += 1

accuracy = float(result)/total
print("Accuracy {}".format(accuracy))

