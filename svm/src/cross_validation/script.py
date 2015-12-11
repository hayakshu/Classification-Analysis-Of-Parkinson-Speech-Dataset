from cross_validation import *

# Cross Validation for pca
# File path for training set - preprocessed via PCA and not
trainName = "../dataset/pca_train.csv"
trainPatientTypeName = "../dataset/train_patientType.csv"

# Load data from files by converting to Pandas DataFrame
trainData = pd.read_csv(trainName)
trainPatientType = pd.read_csv(trainPatientTypeName)

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()

linear(trainData, trainPatientType)
rbf(trainData, trainPatientType)

# Cross Validation on Non PCA Method
trainName = "../dataset/train.csv"

# Remove header and patient index numbers from Pandas DataFrame
# and convert to Numpy Array
trainData = trainData.iloc[:,1:]
trainData = np.array(trainData)
trainPatientType = trainPatientType.iloc[:,1:]
trainPatientType = np.array(trainPatientType).ravel()

linear(trainData, trainPatientType)
rbf(trainData, trainPatientType)



