__author__ = 'Timardeep'
from sklearn.grid_search import GridSearchCV
from numpy import loadtxt,unique,count_nonzero,asarray
import numpy as np
from sklearn import linear_model,neighbors
from  sklearn.svm import NuSVC,SVC,LinearSVC
from natsort import natsorted
#-------------------------------------Loading the data-----------------------------------------------------------------

data = loadtxt('C:/Users/Admin/Documents/ML Project 4/COMP_598_A4/train_data.txt',delimiter=',')


'''Creating the output file which will store the values of
different setups for alpha, penalty and number of iterations'''

#result_file = open("result_output",'w')
result_file_SVM = open('result_output_SVM','w')

'''Starting grid search to find optimal values for above mentioned parameters for SGDC'''

'''penality = ['l1','l2','elasticnet']     #parameter for SGDC
alpha = [0.1,0.01,0.001,0.0001,0.00001,0.5,0.05,0.005,0.0005,0.00005] #parameter for SGDC
n_iter = [5,10,15,20,25,30,40,50,100] # parameter for SGDC
avg_final_results = []
for n_it in  n_iter:
    for pen in penality:
        for alp in alpha:
            print("Parameters under test" + "\n")
            print ("alpha =%f" %alp + "\t" + "num_iteration=%d" %n_it + "\t" + "penality=%s" %pen )

            X = data[:]
            num_patients = unique(X[:,0])
            final_results =[]

            for m in range(10):
                print "m is",m
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
                        valid_examples.append(valid_ex)
                        valid_labels.append(valid_la)
                    true_labels.append(int(valid_la))

                    for k in rest_patients:
                        train_ex = k[1:27]
                        train_la = k[28]
                        train_examples.append(train_ex)
                        train_labels.append(train_la)
                    #-----------------------Training the model for Classification------------------------------------------
                    clf = linear_model.SGDClassifier(loss = 'log',penalty = pen, alpha = alp, n_iter = n_it)
                    clf.fit(train_examples,train_labels)
                    #-----------------------Performing classification for Validation set------------------------------------
                    predict = clf.predict(valid_examples)
                    #-------------------------------Counting subjects having positive parkinson-----------------------------
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
                final_results.append(percent_correct)
            #print final_results
            avg_final_result = sum(final_results) / len(final_results)
            print avg_final_result
            avg_final_results.append(avg_final_result)
            print avg_final_results

            result_file.write("\n" + "Parameters under test" + "\n")
            result_file.write("alpha =%f" %alp + "\t" + "num_iteration=%d" %n_it + "\t" + "penality=%s" %pen + "\n")
            result_file.write("Accuracy=%f"%avg_final_result)
print avg_final_results
nat = natsorted(avg_final_results,reverse=True)
print nat[0:5]
result_file.write("The final list of tested parameters is" + "\n")
for i in nat:
    result_file.write("%f"%i + "\t")'''


'''gamma =[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.5,0.05,0.005,0.0005,0.00005,0,000005]
C = [1,2,5,10,15,20,25,30,50,100,200,500,1000,2000,5000,10000]
avg_final_results=[]
for ga in gamma:
    for c in C:
        print("Parameters under test" + "\n")
        print ( "gamma=%f" %ga + "\t" + "Regularization=%d"%c )

        X = data[:]
        num_patients = unique(X[:,0])
        final_results = []

        for m in range(5):
            results = []
            true_labels = []
            for i in num_patients:
        #-------------------------------Dividing the dataset into Training and Validation set-----------------------
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
        #-----------------------Training the model for Classification------------------------------------------
                clf = SVC(gamma = ga,C=c)
                clf.fit(train_examples,train_labels)
            #-----------------------Performing classification for Validation set------------------------------------
                predict = clf.predict(valid_examples)
            #-------------------------------Counting subjects having positive parkinson-----------------------------
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
            final_results.append(percent_correct)
        avg_final_result = sum(final_results) / len(final_results)
        print avg_final_result
        avg_final_results.append(avg_final_result)
        print avg_final_results
        result_file_SVM.write("\n" + "Parameters under test" + "\n")
        result_file_SVM.write( "gamma=%f" %ga + "\t" + "Regularization=%d"%c )
        result_file_SVM.write("Accuracy=%f"%percent_correct)

print avg_final_results
nat = natsorted(avg_final_results,reverse=True)
print nat[0:5]
result_file_SVM.write("The final list of tested parameters is" + "\n")
for i in nat:
    result_file.write("%f"%i + "\t")'''



C = [1,2,5,10,15,20,25,30,50,100,200,500,1000,2000,5000,10000]
avg_final_results=[]
for c in C:
    print("Parameters under test" + "\n")
    print ("Regularization=%d"%c )

    X = data[:]
    num_patients = unique(X[:,0])
    final_results = []

    results = []
    true_labels = []
    for i in num_patients:
        #-------------------------------Dividing the dataset into Training and Validation set-----------------------
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
        clf = SVC(C=c,kernel='linear')
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
        results.append(result)
    asarray(results)
    asarray(true_labels)
    correct_results =  filter(lambda x : x[0] == x[1],zip(true_labels,results))
    print "Total correct predictions are %d"%len(correct_results)
    percent_correct = (float(len(correct_results))/len(results))*100
    print "Classifier is correct %f percent times"%percent_correct
    final_results.append(percent_correct)
avg_final_result = sum(final_results) / len(final_results)
print avg_final_result
avg_final_results.append(avg_final_result)
print avg_final_results
result_file_SVM.write("\n" + "Parameters under test" + "\n")
result_file_SVM.write( "Regularization=%d"%c )
result_file_SVM.write("Accuracy=%f"%percent_correct)
print avg_final_results
nat = natsorted(avg_final_results,reverse=True)
print nat[0:5]
result_file_SVM.write("The final list of tested parameters is" + "\n")
for i in nat:
    result_file.write("%f"%i + "\t")


