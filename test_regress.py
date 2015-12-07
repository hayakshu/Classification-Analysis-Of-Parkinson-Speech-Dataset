__author__ = 'Admin'

__author__ = 'Admin'
from numpy import loadtxt,unique,count_nonzero,asarray
from sklearn import linear_model,neighbors
from sklearn.svm import SVC,LinearSVC
data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')
test_data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/dataset/test_data.txt',delimiter=',')
#print len(cols_to_add)


X_train = data[:]
X_test = test_data[:]
num_patients = unique(X_test[:,0])
true_labels = []
results =[]
for i in num_patients:
    test_examples =[]
    train_examples = []
    train_labels = []
    one_patient = [row for row in X_test if i == row[0]]
    rest_patients = [ row for row in X_train]
    for j in one_patient:
        test_ex = j[1:27]
        test_examples.append(test_ex)

    for k in rest_patients:
        train_ex = k[1:27]
        train_la = k[28]
        train_examples.append(train_ex)
        train_labels.append(train_la)


    #clf = linear_model.SGDClassifier(loss='log')
    clf = SVC(C = 10, gamma = 0.00001)
    #clf = neighbors.KNeighborsClassifier(n_neighbors= 5,weights = 'distance')


    clf.fit(train_examples,train_labels)
    predict = clf.predict(test_examples)
    print predict
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