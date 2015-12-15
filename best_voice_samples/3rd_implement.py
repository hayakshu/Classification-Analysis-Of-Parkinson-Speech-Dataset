from numpy import loadtxt,unique,count_nonzero
from sklearn import linear_model,neighbors
from sklearn.svm import NuSVC,SVC,LinearSVC
from collections import defaultdict


#----------------------------------------Loading the entire Dataset-----------------------------------------------------
data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')
indiv_acc = open("indiv_acc.csv",'w')
columns = ['Speech Sample','Accuracy']
indiv_acc.write('\t'.join(columns) + '\n')
X = data[:]
num_patients = unique(X[:,0])

true_labels = []
d = defaultdict(list)
li = []

for i in num_patients:

    train_examples = []
    train_labels = []
    one_patient = [row for row in X if i == row[0]]
    rest_patients = [ row for row in X if i != row[0]]
    for k in rest_patients:
        train_ex = k[1:27]
        train_la = k[28]
        train_examples.append(train_ex)
        train_labels.append(train_la)
    for j in range(len(one_patient)):
        valid_examples =[]
        valid_labels =[]
        print "Prediction taking only %d examples"%j

        valid_ex =one_patient[j][1:27]
        valid_la = one_patient[j][28]
        valid_examples.append(valid_ex)
        valid_labels.append(valid_la)

        clf = SVC(C=1,gamma=0.00005)
        clf.fit(train_examples,train_labels)
        predict = clf.predict(valid_examples)
        total = len(predict)
        parkin_pos = count_nonzero(predict)
        parkin_neg = total - parkin_pos
        if parkin_pos > parkin_neg:
            result = 1
            print "Subject %d is detected with positive parkinson "%i
        else:
            result = 0
            print "Subject %d is detected with negative parkinson" %i

        li.append((j,result))

    true_labels.append(int(valid_la))
for key, value in li:
    d[key].append(value)
a = d.items()
print a
for t in a:
    correct_results =  filter(lambda x : x[0] == x[1],zip(true_labels,t[1]))

    print "Total correct predictions for speech sample %d are %d"%(t[0],len(correct_results))
    percent_correct = (float(len(correct_results))/len(num_patients))*100
    print "Percentage of correct predictions for speech sample %d are %d "%(t[0],percent_correct)

    log = {}
    log['Speech Sample'] = t[0]
    log['Accuracy'] = percent_correct
    values = map(lambda col:log.get(col,''),columns)
    values = [str(v) for v in values]
    indiv_acc.write('\t'.join(values) + '\n')



'''if j in d.keys():
            d[value].append(value)
        else:
            d[value] = value
            d[j] = j'''



