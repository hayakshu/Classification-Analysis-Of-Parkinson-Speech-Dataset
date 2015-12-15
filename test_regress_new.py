__author__ = 'Timardeep'


from numpy import loadtxt,unique,count_nonzero,asarray
from sklearn import linear_model,neighbors
from sklearn.svm import LinearSVC,SVC

from natsort import natsorted
data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')
test_data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/dataset/test_data.txt',delimiter=',')
#print len(cols_to_add)
a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 50.0, 45.0, 47.5, 50.0, 52.5, 52.5, 55.000000000000014, 52.5, 55.000000000000014, 55.000000000000014, 55.000000000000014, 55.000000000000014, 55.000000000000014, 55.000000000000014, 55.000000000000014, 52.5, 60.0, 50.0, 50.0, 55.000000000000014, 52.5, 50.0, 55.000000000000014, 57.499999999999986, 60.0, 62.5, 57.499999999999986, 55.000000000000014, 50.0, 57.499999999999986, 52.5, 45.0, 47.5, 50.0, 52.5, 50.0, 52.5, 52.5, 55.000000000000014, 55.000000000000014, 55.000000000000014, 55.000000000000014, 57.499999999999986, 57.499999999999986, 55.000000000000014, 52.5, 52.5, 55.000000000000014, 50.0, 47.5, 37.5, 40.0, 42.5, 42.5, 40.0, 42.5, 47.5, 52.5, 50.0, 50.0, 55.000000000000014, 55.000000000000014, 55.000000000000014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 47.5, 50.0, 50.0, 50.0, 50.0, 45.0, 50.0, 50.0, 50.0, 57.499999999999986, 52.5, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0, 52.5, 52.5, 55.000000000000014, 55.000000000000014, 55.000000000000014, 52.5, 50.0, 52.5, 60.0, 55.000000000000014, 60.0, 62.5, 62.5, 62.5, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
b = natsorted(a, reverse = True)
print b

X_train = data[:]
X_test = test_data[:]
num_patients = unique(X_test[:,0])
results = []
true_labels = []
train_examples = []
train_labels = []
rest_patients = [ row for row in X_train ]
#print rest_patients
for k in rest_patients:
    train_ex = k[1:27]
    train_la = k[28]
    train_examples.append(train_ex)
    train_labels.append(train_la)
for i in num_patients:
    test_examples =[]
    one_patient = [row for row in X_test if i == row[0]]
    #rest_patients = [ row for row in X_train]
    for j in one_patient:
        test_ex = j[1:27]
        test_examples.append(test_ex)

    #clf = linear_model.SGDClassifier(loss='log',penalty = 'l1' ,n_iter = 30,alpha = 0.006)
    #clf = linear_model.SGDClassifier(loss='log',penalty = 'l1' ,n_iter = 50,alpha = 0.005)
    #clf = LinearSVC(C= 5)
    #clf = SVC(gamma=0.00005,C=1)
    clf = neighbors.KNeighborsClassifier(n_neighbors= 3,weights = 'distance')
    clf.fit(train_examples,train_labels)
    predict = clf.predict(test_examples)
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
correct_results =  count_nonzero(results)
print "Total correct predictions are %d"%(correct_results)
percent_correct = (float(correct_results)/len(results))*100
print "Classifier is correct %f percent times"%percent_correct