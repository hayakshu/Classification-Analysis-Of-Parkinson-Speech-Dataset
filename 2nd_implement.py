__author__ = 'Timardeep'

import sys
#sys.path.append('C:/Users/Admin/Documents/libsvm-3.20/libsvm-3.20')
from numpy import loadtxt,unique,count_nonzero,asarray,mean
#import svmutil
from sklearn import linear_model,neighbors
from sklearn.svm import LinearSVC,SVC
def extract_features(record,i):
    log = dict()
    log['subject_id'] = i+1
    mean_features = mean(record,axis=0)
    log['mean_jitter(local)'] = mean_features[0]
    print mean_features[0]
    return log



data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')
#X_train = data[:]
new_data = open('new_dataset.csv','w')
cols_to_add =['subject_id','mean_jitter(local)','mean_jitter(local,absolute)','mean_jitter(rap)','mean_jitter(ppq5)','mean_jitter(ddp)',
              'mean_shimmer(local)','mean_shimmer(local,dB)','mean_shimmer(apq3)','mean_shimmer(apq5)','mean_shimmer(apq11)','mean_shimmer(dda)'
              ,'mean_AC','mean_NTH','mean_HTN','mean_median_pitch','mean_mean_pitch','mean_std_dev','mean_min_pitch','mean_max_pitch','mean_num_pulses'
              ,'mean_num_periods','mean_mean_period','mean_std_dev_period','mean_frac_locallyunvoiced_frames','mean_num_voice_breaks','mean_degree_voicebreaks',
              'median_jitter(local)','median_jitter(local,absolute)','median_jitter(rap)','median_jitter(ppq5)','median_jitter(ddp)',
              'median_shimmer(local)','median_shimmer(local,dB)','median_shimmer(apq3)','median_shimmer(apq5)','median_shimmer(apq11)','median_shimmer(dda)'
              ,'median_AC','median_NTH','median_HTN','median_median_pitch','median_mean_pitch','median_std_dev','median_min_pitch','median_max_pitch','median_num_pulses'
              ,'median_num_periods','median_mean_period','median_std_dev_period','median_frac_locallyunvoiced_frames','median_num_voice_breaks','median_degree_voicebreaks'
              ]
new_data.write('\t'.join(cols_to_add) + '\n')

X = data[:]
num_patients = unique(X[:,0])
results = []
true_labels = []
all_patients =[]
for i in num_patients:
    patient = [row [:][1:27] for row in X if i == row[0] ]
    all_patients.append(patient)

    asarray(all_patients)

for i in range(len(num_patients)):
    log = extract_features(all_patients[i],i)
    values = map(lambda col:log.get(col,''),cols_to_add)
    values = [str(v) for v in values]
    new_data.write('\t'.join(values) + '\n')


'''valid_examples =[]
    #valid_labels =[]
    #train_examples = []
    #train_labels = []
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

    #clf = linear_model.LogisticRegression()
    clf = LinearSVC()
    #clf = neighbors.KNeighborsClassifier(n_neighbors= 5,weights = 'distance')
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
print "Classifier is correct %f percent times"%percent_correct'''