import sys
from numpy import loadtxt,unique,count_nonzero,asarray
#import svmutil
import sklearn
from sklearn import linear_model,neighbors
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier


data = loadtxt('/Users/elcinergin/Dropbox/MCGILL/COURSES/2ndyear/comp598 machine learning/Project 4/COMP_598_A4/dataset/train_data.txt',delimiter=',')
#X_train = data[:]
X = data[:]
scaler = sklearn.preprocessing.StandardScaler()
num_patients = unique(X[:,0])
results = []
true_labels = []
for i in num_patients:
    valid_examples =[]
    valid_labels =[]
    train_examples = []
    train_labels = []
    one_patient = [row for row in X if i == row[0]]
    rest_patients = [ row for row in X if i != row[0]]
    
    for j in one_patient:
        valid_ex = j[1:27]
        valid_la = j[28]
        #valid_ex=scaler.fit_transform(valid_ex)
        valid_examples.append(valid_ex)
        valid_labels.append(valid_la)
    true_labels.append(int(valid_la))

    for k in rest_patients:
        train_ex = k[1:27]
        train_la = k[28]
        #train_ex=scaler.transform(train_ex)
        train_examples.append(train_ex)
        train_labels.append(train_la)

    #clf = linear_model.LogisticRegression()
    clf = LinearSVC()
    #clf = neighbors.KNeighborsClassifier(n_neighbors= 5,weights = 'distance')
    #clf = RandomForestClassifier(n_estimators=10)
    clf.fit(train_examples,train_labels)
    predict = clf.predict(valid_examples)
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
asarray(true_labels)

correct_results =  filter(lambda x : x[0] == x[1],zip(true_labels,results))
print "Total correct predictions are %d"%len(correct_results)
percent_correct = (float(len(correct_results))/len(results))*100
print "Classifier is correct %f percent times"%percent_correct

CM=confusion_matrix(true_labels, results)
print(classification_report(true_labels,results))
print CM
TP=CM[1,1]
TN=CM[0,0]
FP=CM[0,1]
FN=CM[1,0]

specifity=(float(TN)/(TN+FP))*100
sensitivity=(float(TP)/(TP+FN))*100

print "Specifity is %f"%specifity
print "Sensitivity is %f"%sensitivity
print(matthews_corrcoef(true_labels,results))

