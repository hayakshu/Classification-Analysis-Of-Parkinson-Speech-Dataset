from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from numpy import loadtxt
import numpy as np
import pandas as pd
import sklearn

def logreg(X, Y):

    pen = ['l1','l2']     #parameter for Log Reg
    alpha = [10, 5,1, 0.1,0.01,0.001,0.0001,0.00001,0.5,0.05,0.005,0.0005,0.00005] #parameter for SGDC&Log Reg
    n_iter = [5,10,15,20,25,30,40,50,100] #parameter for SGDC
    loss_fun=['log'] #parameter for SGDC

    parameters = [{ 'C': alpha,'penalty' :pen}]
    dataLength = len(X)
    log_reg = LogisticRegression()
    lv = LeaveOneOut(dataLength)
    clf = GridSearchCV(log_reg, parameters, cv= lv)
    clf.fit(X, Y)
    
    # Obtaining Parameters from grid_scores
    accuracy = [p[1] for p in clf.grid_scores_]
    reg_coeff = [p[0]['C'] for p in clf.grid_scores_]
    reg_type = [p[0]['penalty'] for p in clf.grid_scores_]
    print accuracy
    
    print reg_coeff
    print reg_type
    accur_l1=[]
    regco_l1=[]
    accur_l2=[]
    regco_l2=[] 
    
    asarray(reg_type)
    for i in range(len(accuracy)):
        if i%2 == 0:
            accur_l1.append(accuracy[i])
            regco_l1.append(reg_coeff[i])
        else:
            accur_l2.append(accuracy[i])
            regco_l2.append(reg_coeff[i])
    print accur_l1
    print accur_l2
    print regco_l1
    print regco_l2
              
    asarray(accuracy)
    asarray(reg_coeff)
    
    accuracy = np.reshape(accuracy,(-1,2))
    reg_coeff= np.reshape(reg_coeff,(-1,2))
    
    asarray(accur_l1)
    asarray(accur_l2)
    asarray(regco_l1)
    asarray(regco_l2)
    
    #accur_l1 = np.reshape(accur_l1,(-1,2))
    #accur_l2 = np.reshape(accur_l2,(-1,2))
    #regco_l1 = np.reshape(regco_l1,(-1,2))
    #regco_l2 = np.reshape(regco_l2,(-1,2))    
    
    fig = plt.figure()
    

    plt.plot(regco_l2,accur_l2,'-o')
    plt.xlabel('Accuracy')
    plt.ylabel('Regularization coef. of l2 regularization')

    plt.show()
    
    print "Best Parameters: {}".format(clf.best_params_)
    print "Accuracy: {}".format(clf.best_score_)
    
def logreg_analysis(X, Y, a, p):

	print "Performing Cross Validation on Penalty: {}".format(a)
	dataLength = len(X)
	loo = LeaveOneOut(dataLength)
	predictions = []
	expected = []
	TP, FN, TN, FP = 0, 0, 0, 0
	Accuracy = 0
	for train_index, test_index in loo:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index][0]

		clf = LogisticRegression(penalty=p, C= a)
		clf.fit(X_train, Y_train)
		prediction = clf.predict(X_test)[0]
	
		predictions.append(prediction)
		expected.append(Y_test)

	print("Calculating.....")
	for i, prediction in enumerate(predictions):
		if(prediction == 1 and expected[i] == 1):
			TP += 1
		elif(prediction == 0 and expected[i] == 1):
			FN += 1
		elif(prediction == 0 and expected[i] == 0):
			TN += 1
		elif(prediction == 1 and expected[i] == 0):
			FP += 1
		else:
			pass

	Sensitivity = TP/float(TP + FN)
	Specificity = TN/float(TN + FP)
	Accuracy = (TP + TN)/float(TP + TN + FP + FN)

	print("Sensitivity of Prediction: {} @ Regularization: {} @ Penalty: {}".format(Sensitivity, a, p))
	print("Specificity of Prediction: {} @ Regularization: {} @ Penalty: {}".format(Specificity, a, p))
	print("Accuracy of Prediction: {} @ Regularization: {} @ Penalty: {}".format(Accuracy, a, p))
	print("Matthews Correlation Coeefficient Value: {}".format(matthews_corrcoef(predictions, expected)))
	print("Classification Report:")
	print(classification_report(predictions, expected))
	print("Confusion Matrix")
	print(confusion_matrix(predictions, expected))    
    
    
print "Performing Cross Validation on logistic regression SLOO Data"

trainName = "train.csv"
trainPatientTypeName = "type_train.csv"

# Load data from files by converting to Pandas DataFrame
trainData = pd.read_csv(trainName)
trainPatientType = pd.read_csv(trainPatientTypeName)


# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()

scaler = sklearn.preprocessing.StandardScaler()
trainData=scaler.fit_transform(trainData)
logreg(trainData, trainPatientType)
logreg_analysis(trainData, trainPatientType,0.5,'l2')

