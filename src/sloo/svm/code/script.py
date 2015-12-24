import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

X_train = "../dataset/pandas/train.csv"
X_test = "../dataset/pandas/test.csv"
Y_train = "../dataset/patient_type/train.csv"
Y_test = "../dataset/patient_type/test.csv"

X_train = pd.read_csv(X_train)
X_test = pd.read_csv(X_test)
Y_train = pd.read_csv(Y_train)
Y_test = pd.read_csv(Y_test)

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
X_train = X_train.iloc[:,1:]
X_train = np.array(X_train)
Y_train = Y_train.iloc[:,1:]
Y_train = np.array(Y_train).ravel()

X_test = X_test.iloc[:,1:]
X_test = np.array(X_test)
Y_test = Y_test.iloc[:,1:]
Y_test = np.array(Y_test).ravel()



# SVM Confusion Matrix Build
clf = SVC(kernel='rbf', C=1, gamma=1e-06)
Y_pred = clf.fit(X_train, Y_train).predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
plt.figure()
plot_confusion_matrix(cm, title='Confusion Matrix')