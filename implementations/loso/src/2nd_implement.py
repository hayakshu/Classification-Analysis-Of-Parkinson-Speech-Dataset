__author__ = 'Timardeep'

import sys
import numpy as np
#sys.path.append('C:/Users/Admin/Documents/libsvm-3.20/libsvm-3.20')
from numpy import loadtxt,unique,count_nonzero,asarray,mean,median,std,absolute
#import svmutil
from sklearn import linear_model,neighbors
from sklearn.svm import LinearSVC,SVC
from scipy import stats
def extract_features(record,i):
    log = dict()
    log['subject_id'] = i+1
    mean_features = mean(record,axis=0)
    median_features = median(record,axis=0)
    trim_mean_10_features = stats.trim_mean(record,0.1,axis = 0)
    trim_mean_25_features = stats.trim_mean(record,0.25,axis = 0)
    std_dev_features = std(record,axis =0)
    iqr_features = np.subtract(*np.percentile(record, [75, 25],axis=0))
    mad_features = mean(absolute(record - mean(record, axis = 0)), axis = 0)
    print mad_features
    log['mean_jitter(local)'] = mean_features[0]
    log['mean_jitter(local,absolute)']= mean_features[1]
    log['mean_jitter(ppq5)'] = mean_features[2]
    log['mean_jitter(rap)'] = mean_features[3]
    log['mean_jitter(ddp)'] = mean_features[4]
    log['mean_shimmer(local)'] = mean_features[5]
    log['mean_shimmer(local,dB)'] = mean_features[6]
    log['mean_shimmer(apq3)'] = mean_features[7]
    log['mean_shimmer(apq5)'] = mean_features[8]
    log['mean_shimmer(apq11)'] = mean_features[9]
    log['mean_shimmer(dda)'] = mean_features[10]
    log['mean_AC'] = mean_features[11]
    log['mean_NTH'] = mean_features[12]
    log['mean_HTN'] = mean_features[13]
    log['mean_median_pitch'] = mean_features[14]
    log['mean_mean_pitch'] = mean_features[15]
    log['mean_std_dev'] = mean_features[16]
    log['mean_min_pitch'] = mean_features[17]
    log['mean_max_pitch'] = mean_features[18]
    log['mean_num_pulses'] = mean_features[19]
    log['mean_num_periods'] = mean_features[20]
    log['mean_mean_period'] = mean_features[21]
    log['mean_std_dev_period'] = mean_features[22]
    log['mean_frac_locallyunvoiced_frames'] = mean_features[23]
    log['mean_num_voice_breaks'] = mean_features[24]
    log['mean_degree_voicebreaks']= mean_features[25]
    log['median_jitter(local)'] = median_features[0]
    log['median_jitter(local,absolute)'] = median_features[1]
    log['median_jitter(rap)'] = median_features[2]
    log['median_jitter(ppq5)'] = median_features[3]
    log['median_jitter(ddp)'] = median_features[4]
    log['median_shimmer(local)'] = median_features[5]
    log['median_shimmer(local,dB)'] = median_features[6]
    log['median_shimmer(apq3)'] = median_features[7]
    log['median_shimmer(apq5)'] = median_features[8]
    log['median_shimmer(apq11)'] = median_features[9]
    log['median_shimmer(dda)'] = median_features[10]
    log['median_AC'] = median_features[11]
    log['median_NTH'] = median_features[12]
    log['median_HTN'] = median_features[13]
    log['median_median_pitch'] = median_features[14]
    log['median_mean_pitch'] = median_features[15]
    log['median_std_dev'] = median_features[16]
    log['median_min_pitch'] = median_features[17]
    log['median_max_pitch'] = median_features[18]
    log['median_num_pulses'] = median_features[19]
    log['median_num_periods'] = median_features[20]
    log['median_mean_period'] = median_features[21]
    log['median_std_dev_period'] = median_features[22]
    log['median_frac_locallyunvoiced_frames'] = median_features[23]
    log['median_num_voice_breaks'] = median_features[24]
    log['median_degree_voicebreaks'] = median_features[25]
    log['trim25mean_jitter(local)'] = trim_mean_10_features[0]
    log['trim25mean_jitter(local,absolute)']= trim_mean_10_features[1]
    log['trim25mean_jitter(ppq5)'] = trim_mean_10_features[2]
    log['trim25mean_jitter(rap)'] = trim_mean_10_features[3]
    log['trim25mean_jitter(ddp)'] = trim_mean_10_features[4]
    log['trim25mean_shimmer(local)'] = trim_mean_10_features[5]
    log['trim25mean_shimmer(local,dB)'] = trim_mean_10_features[6]
    log['trim25mean_shimmer(apq3)'] = trim_mean_10_features[7]
    log['trim25mean_shimmer(apq5)'] = trim_mean_10_features[8]
    log['trim25mean_shimmer(apq11)'] = trim_mean_10_features[9]
    log['trim25mean_shimmer(dda)'] = trim_mean_10_features[10]
    log['trim25mean_AC'] = trim_mean_10_features[11]
    log['trim25mean_NTH'] = trim_mean_10_features[12]
    log['trim25mean_HTN'] = trim_mean_10_features[13]
    log['trim25mean_median_pitch'] = trim_mean_10_features[14]
    log['trim25mean_mean_pitch'] = trim_mean_10_features[15]
    log['trim25mean_std_dev'] = trim_mean_10_features[16]
    log['trim25mean_min_pitch'] = trim_mean_10_features[17]
    log['trim25mean_max_pitch'] = trim_mean_10_features[18]
    log['trim25mean_num_pulses'] = trim_mean_10_features[19]
    log['trim25mean_num_periods'] = trim_mean_10_features[20]
    log['trim25mean_mean_period'] = trim_mean_10_features[21]
    log['trim25mean_std_dev_period'] = trim_mean_10_features[22]
    log['trim25mean_frac_locallyunvoiced_frames'] = trim_mean_10_features[23]
    log['trim25mean_num_voice_breaks'] = trim_mean_10_features[24]
    log['trim25mean_degree_voicebreaks']= trim_mean_10_features[25]
    log['trim25mean_jitter(local)'] = trim_mean_25_features[0]
    log['trim25mean_jitter(local,absolute)']= trim_mean_25_features[1]
    log['trim25mean_jitter(ppq5)'] = trim_mean_25_features[2]
    log['trim25mean_jitter(rap)'] = trim_mean_25_features[3]
    log['trim25mean_jitter(ddp)'] = trim_mean_25_features[4]
    log['trim25mean_shimmer(local)'] = trim_mean_25_features[5]
    log['trim25mean_shimmer(local,dB)'] = trim_mean_25_features[6]
    log['trim25mean_shimmer(apq3)'] = trim_mean_25_features[7]
    log['trim25mean_shimmer(apq5)'] = trim_mean_25_features[8]
    log['trim25mean_shimmer(apq11)'] = trim_mean_25_features[9]
    log['trim25mean_shimmer(dda)'] = trim_mean_25_features[10]
    log['trim25mean_AC'] = trim_mean_25_features[11]
    log['trim25mean_NTH'] = trim_mean_25_features[12]
    log['trim25mean_HTN'] = trim_mean_25_features[13]
    log['trim25mean_median_pitch'] = trim_mean_25_features[14]
    log['trim25mean_mean_pitch'] = trim_mean_25_features[15]
    log['trim25mean_std_dev'] = trim_mean_25_features[16]
    log['trim25mean_min_pitch'] = trim_mean_25_features[17]
    log['trim25mean_max_pitch'] = trim_mean_25_features[18]
    log['trim25mean_num_pulses'] = trim_mean_25_features[19]
    log['trim25mean_num_periods'] = trim_mean_25_features[20]
    log['trim25mean_mean_period'] = trim_mean_25_features[21]
    log['trim25mean_std_dev_period'] = trim_mean_25_features[22]
    log['trim25mean_frac_locallyunvoiced_frames'] = trim_mean_25_features[23]
    log['trim25mean_num_voice_breaks'] = trim_mean_25_features[24]
    log['trim25mean_degree_voicebreaks']= trim_mean_25_features[25]
    log['std_jitter(local)'] = std_dev_features[0]
    log['std_jitter(local,absolute)']= std_dev_features[1]
    log['std_jitter(ppq5)'] = std_dev_features[2]
    log['std_jitter(rap)'] = std_dev_features[3]
    log['std_jitter(ddp)'] = std_dev_features[4]
    log['std_shimmer(local)'] = std_dev_features[5]
    log['std_shimmer(local,dB)'] = std_dev_features[6]
    log['std_shimmer(apq3)'] = std_dev_features[7]
    log['std_shimmer(apq5)'] = std_dev_features[8]
    log['std_shimmer(apq11)'] = std_dev_features[9]
    log['std_shimmer(dda)'] = std_dev_features[10]
    log['std_AC'] = std_dev_features[11]
    log['std_NTH'] = std_dev_features[12]
    log['std_HTN'] = std_dev_features[13]
    log['std_median_pitch'] = std_dev_features[14]
    log['std_mean_pitch'] = std_dev_features[15]
    log['std_std_dev'] = std_dev_features[16]
    log['std_min_pitch'] = std_dev_features[17]
    log['std_max_pitch'] = std_dev_features[18]
    log['std_num_pulses'] = std_dev_features[19]
    log['std_num_periods'] = std_dev_features[20]
    log['std_mean_period'] = std_dev_features[21]
    log['std_std_dev_period'] = std_dev_features[22]
    log['std_frac_locallyunvoiced_frames'] = std_dev_features[23]
    log['std_num_voice_breaks'] = std_dev_features[24]
    log['std_degree_voicebreaks']= std_dev_features[25]
    log['iqr_jitter(local)'] = iqr_features[0]
    log['iqr_jitter(local,absolute)']= iqr_features[1]
    log['iqr_jitter(ppq5)'] = iqr_features[2]
    log['iqr_jitter(rap)'] = iqr_features[3]
    log['iqr_jitter(ddp)'] = iqr_features[4]
    log['iqr_shimmer(local)'] = iqr_features[5]
    log['iqr_shimmer(local,dB)'] = iqr_features[6]
    log['iqr_shimmer(apq3)'] = iqr_features[7]
    log['iqr_shimmer(apq5)'] = iqr_features[8]
    log['iqr_shimmer(apq11)'] = iqr_features[9]
    log['iqr_shimmer(dda)'] = iqr_features[10]
    log['iqr_AC'] = iqr_features[11]
    log['iqr_NTH'] = iqr_features[12]
    log['iqr_HTN'] = iqr_features[13]
    log['iqr_median_pitch'] = iqr_features[14]
    log['iqr_mean_pitch'] = iqr_features[15]
    log['iqr_std_dev'] = iqr_features[16]
    log['iqr_min_pitch'] = iqr_features[17]
    log['iqr_max_pitch'] = iqr_features[18]
    log['iqr_num_pulses'] = iqr_features[19]
    log['iqr_num_periods'] = iqr_features[20]
    log['iqr_mean_period'] = iqr_features[21]
    log['iqr_std_dev_period'] = iqr_features[22]
    log['iqr_frac_locallyunvoiced_frames'] = iqr_features[23]
    log['iqr_num_voice_breaks'] = iqr_features[24]
    log['iqr_degree_voicebreaks']= iqr_features[25]
    log['mad_jitter(local)'] = mad_features[0]
    log['mad_jitter(local,absolute)']= mad_features[1]
    log['mad_jitter(ppq5)'] = mad_features[2]
    log['mad_jitter(rap)'] = mad_features[3]
    log['mad_jitter(ddp)'] = mad_features[4]
    log['mad_shimmer(local)'] = mad_features[5]
    log['mad_shimmer(local,dB)'] = mad_features[6]
    log['mad_shimmer(apq3)'] = mad_features[7]
    log['mad_shimmer(apq5)'] = mad_features[8]
    log['mad_shimmer(apq11)'] = mad_features[9]
    log['mad_shimmer(dda)'] = mad_features[10]
    log['mad_AC'] = mad_features[11]
    log['mad_NTH'] = mad_features[12]
    log['mad_HTN'] = mad_features[13]
    log['mad_median_pitch'] = mad_features[14]
    log['mad_mean_pitch'] = mad_features[15]
    log['mad_std_dev'] = mad_features[16]
    log['mad_min_pitch'] = mad_features[17]
    log['mad_max_pitch'] = mad_features[18]
    log['mad_num_pulses'] = mad_features[19]
    log['mad_num_periods'] = mad_features[20]
    log['mad_mean_period'] = mad_features[21]
    log['mad_std_dev_period'] = mad_features[22]
    log['mad_frac_locallyunvoiced_frames'] = mad_features[23]
    log['mad_num_voice_breaks'] = mad_features[24]
    log['mad_degree_voicebreaks']= mad_features[25]




    return log



data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')
#X_train = data[:]
new_data = open('new_dataset.csv','w')
cols_to_add =['subject_id','mean_jitter(local)','mean_jitter(local,absolute)','mean_jitter(ppq5)','mean_jitter(rap)','mean_jitter(ddp)',
              'mean_shimmer(local)','mean_shimmer(local,dB)','mean_shimmer(apq3)','mean_shimmer(apq5)','mean_shimmer(apq11)','mean_shimmer(dda)'
              ,'mean_AC','mean_NTH','mean_HTN','mean_median_pitch','mean_mean_pitch','mean_std_dev','mean_min_pitch','mean_max_pitch','mean_num_pulses'
              ,'mean_num_periods','mean_mean_period','mean_std_dev_period','mean_frac_locallyunvoiced_frames','mean_num_voice_breaks','mean_degree_voicebreaks',
              'median_jitter(local)','median_jitter(local,absolute)','median_jitter(rap)','median_jitter(ppq5)','median_jitter(ddp)',
              'median_shimmer(local)','median_shimmer(local,dB)','median_shimmer(apq3)','median_shimmer(apq5)','median_shimmer(apq11)','median_shimmer(dda)'
              ,'median_AC','median_NTH','median_HTN','median_median_pitch','median_mean_pitch','median_std_dev','median_min_pitch','median_max_pitch','median_num_pulses'
              ,'median_num_periods','median_mean_period','median_std_dev_period','median_frac_locallyunvoiced_frames','median_num_voice_breaks','median_degree_voicebreaks'
              ,'trim10mean_jitter(local)','trim10mean_jitter(local,absolute)','trim10mean_jitter(ppq5)','trim10mean_jitter(rap)','trim10mean_jitter(ddp)',
              'trim10mean_shimmer(local)','trim10mean_shimmer(local,dB)','trim10mean_shimmer(apq3)','trim10mean_shimmer(apq5)','trim10mean_shimmer(apq11)','trim10mean_shimmer(dda)'
              ,'trim10mean_AC','trim10mean_NTH','trim10mean_HTN','trim10mean_median_pitch','trim10mean_mean_pitch','trim10mean_std_dev','trim10mean_min_pitch','trim10mean_max_pitch','trim10mean_num_pulses'
              ,'trim10mean_num_periods','trim10mean_mean_period','trim10mean_std_dev_period','trim10mean_frac_locallyunvoiced_frames','trim10mean_num_voice_breaks','trim10mean_degree_voicebreaks'
              ,'trim25mean_jitter(local)','trim25mean_jitter(local,absolute)','trim25mean_jitter(ppq5)','trim25mean_jitter(rap)','trim25mean_jitter(ddp)',
              'trim25mean_shimmer(local)','trim25mean_shimmer(local,dB)','trim25mean_shimmer(apq3)','trim25mean_shimmer(apq5)','trim25mean_shimmer(apq11)','trim25mean_shimmer(dda)'
              ,'trim25mean_AC','trim25mean_NTH','trim25mean_HTN','trim25mean_median_pitch','trim25mean_mean_pitch','trim25mean_std_dev','trim25mean_min_pitch','trim25mean_max_pitch','trim25mean_num_pulses'
              ,'trim25mean_num_periods','trim25mean_mean_period','trim25mean_std_dev_period','trim25mean_frac_locallyunvoiced_frames','trim25mean_num_voice_breaks','trim25mean_degree_voicebreaks'
              ,'std_jitter(local)','std_jitter(local,absolute)','std_jitter(ppq5)','std_jitter(rap)','std_jitter(ddp)',
              'std_shimmer(local)','std_shimmer(local,dB)','std_shimmer(apq3)','std_shimmer(apq5)','std_shimmer(apq11)','std_shimmer(dda)'
              ,'std_AC','std_NTH','std_HTN','std_median_pitch','std_mean_pitch','std_std_dev','std_min_pitch','std_max_pitch','std_num_pulses'
              ,'std_num_periods','std_mean_period','std_std_dev_period','std_frac_locallyunvoiced_frames','std_num_voice_breaks','std_degree_voicebreaks'
              ,'iqr_jitter(local)','iqr_jitter(local,absolute)','iqr_jitter(ppq5)','iqr_jitter(rap)','iqr_jitter(ddp)',
              'iqr_shimmer(local)','iqr_shimmer(local,dB)','iqr_shimmer(apq3)','iqr_shimmer(apq5)','iqr_shimmer(apq11)','iqr_shimmer(dda)'
              ,'iqr_AC','iqr_NTH','iqr_HTN','iqr_median_pitch','iqr_mean_pitch','iqr_std_dev','iqr_min_pitch','iqr_max_pitch','iqr_num_pulses'
              ,'iqr_num_periods','iqr_mean_period','iqr_std_dev_period','iqr_frac_locallyunvoiced_frames','iqr_num_voice_breaks','iqr_degree_voicebreaks'
              ,'mad_jitter(local)','mad_jitter(local,absolute)','mad_jitter(ppq5)','mad_jitter(rap)','mad_jitter(ddp)',
              'mad_shimmer(local)','mad_shimmer(local,dB)','mad_shimmer(apq3)','mad_shimmer(apq5)','mad_shimmer(apq11)','mad_shimmer(dda)'
              ,'mad_AC','mad_NTH','mad_HTN','mad_median_pitch','mad_mean_pitch','mad_std_dev','mad_min_pitch','mad_max_pitch','mad_num_pulses'
              ,'mad_num_periods','mad_mean_period','mad_std_dev_period','mad_frac_locallyunvoiced_frames','mad_num_voice_breaks','mad_degree_voicebreaks']
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