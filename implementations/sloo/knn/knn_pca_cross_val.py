
from sklearn.cross_validation import LeaveOneOut
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from numpy import loadtxt
import numpy as np
from numpy import asarray
import pandas as pd
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def knn(X, Y):

    neighbors = [1,2,3,4,5,6,7,8,9,10]
    weights = ['distance']
    n_components = [5,10,15,20,25,30]

    parameters = [{'k_nn__n_neighbors': neighbors, 'k_nn__weights': weights,'pca__n_components':n_components}]
    
    
    dataLength = len(X)
    lv = LeaveOneOut(dataLength)
    
    pipeline = Pipeline([
    ('pca', PCA()),
    ('k_nn', KNeighborsClassifier()),])  
              
            
    clf = GridSearchCV(pipeline, parameters, cv= lv)
    clf.fit(X, Y)
    # Obtaining Parameters from grid_scores
    accuracy = [p[1] for p in clf.grid_scores_]
    pca_components = [p[0]['pca__n_components'] for p in clf.grid_scores_]
    knn_neighbors = [p[0]['k_nn__n_neighbors'] for p in clf.grid_scores_]
    
    asarray(accuracy)
    asarray(pca_components)
    asarray(knn_neighbors)
    
    accuracy = np.reshape(accuracy,(-1,2))
    pca_components= np.reshape(pca_components,(-1,2))
    knn_neighbors= np.reshape(knn_neighbors,(-1,2))
    
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(pca_components,knn_neighbors,accuracy, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Number of neigbours')
    ax.set_zlabel('Accuracy')

    plt.show()
    
    print "Best Parameters: {}".format(clf.best_params_)
    print "Accuracy: {}".format(clf.best_score_)

    
def knn_analysis(X, Y, n, k):

    print "Performing Cross Validation on Penalty: {}".format(k)
    dataLength = len(X)
    loo = LeaveOneOut(dataLength)
    predictions = []
    expected = []
    TP, FN, TN, FP = 0, 0, 0, 0
    Accuracy = 0
    for train_index, test_index in loo:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index][0]

        pca = PCA(n_components=n)
        pca.fit(trainData)
        clf = KNeighborsClassifier(n_neighbors= k, weights = 'distance')
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

    print("Sensitivity of Prediction: {} @ Neighbours: {} @ Components: {}".format(Sensitivity, k, n))
    print("Specificity of Prediction: {} @ Neighbours: {} @ Components: {}".format(Specificity, k, n))
    print("Accuracy of Prediction: {} @ Neighbours: {} @ Components: {}".format(Accuracy, k, n))
    print("Matthews Correlation Coeefficient Value: {}".format(matthews_corrcoef(predictions, expected)))
    print("Classification Report:")
    print(classification_report(predictions, expected))
    print("Confusion Matrix")
    print(confusion_matrix(predictions, expected))    
    
        
    
    
print "Performing Cross Validation on knn SLOO Data"

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

knn(trainData, trainPatientType)
#knn_analysis(trainData, trainPatientType,10,8)

