__author__ = 'Admin'
import sys
#sys.path.append('C:/Users/Admin/Documents/libsvm-3.20/libsvm-3.20')
from numpy import loadtxt,unique,count_nonzero,asarray
import numpy as np
#import svmutil
from sklearn import linear_model,neighbors
from  sklearn.svm import NuSVC,SVC,LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')
#X_train = data[:]

X = data[:]
num_patients = unique(X[:,0])
results = []
true_labels = []
for i in num_patients:
    rest_patient = []
    valid_examples =[]
    valid_labels =[]
    train_examples = []
    train_labels = []
    one_patient = [row for row in X if i == row[0]]
    rest_patients = [ row for row in X if i != row[0]]


    for j in one_patient:
        valid_ex = j[1:27]
        valid_la = j[28]
        valid_examples.append(valid_ex)
        valid_labels.append(valid_la)
    true_labels.append(int(valid_la))

    for k in rest_patients:
        train_ex = k[1:27]
        train_la = k[28]
        train_examples.append(train_ex)
        train_labels.append(train_la)




    #clf = linear_model.SGDClassifier(loss='log',penalty = 'l1' ,n_iter = 30,alpha = 0.006)
    #clf = LinearSVC(C= 5)
    #clf = SVC(C=1,gamma=0.00005)
    #print clf
    clf = neighbors.KNeighborsClassifier(n_neighbors= 5,weights = 'distance')
    clf.fit(train_examples,train_labels)
    #print clf
    #clf = neighbors.KNeighborsClassifier(n_neighbors= 5,weights = 'distance')
    #clf.fit(train_examples,train_labels)
    predict = clf.predict(valid_examples)
    #print predict
    total = len(predict)
    parkin_pos = count_nonzero(predict)
    parkin_neg = total - parkin_pos
    if parkin_pos > parkin_neg:
        result = 1
        print "Subject %d is detected with positive parkinson "%i
    else:
        result = 0
        print "Subject %d is detected with negative parkinson" %i

    results.append(result)
asarray(results)
asarray(true_labels)

print "True labels are ",len(true_labels)
print "results are",len(results)
correct_results =  filter(lambda x : x[0] == x[1],zip(true_labels,results))
print "Total correct predictions are %d"%len(correct_results)
percent_correct = (float(len(correct_results))/len(results))*100
print "Classifier is correct %f percent times"%percent_correct

cm=confusion_matrix(true_labels, results)
print(classification_report(true_labels,results))
print cm
print matthews_corrcoef(results,true_labels)
def plot_confusion_matrix(cm, title="Confusion Matrix", label1="Parkinson", label2="No Parkinson", cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(2)
	plt.xticks(tick_marks, (label1, label2), rotation=45)
	plt.yticks(tick_marks, (label1, label2))
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.axis('tight')
	plt.show()
plot_confusion_matrix(cm)