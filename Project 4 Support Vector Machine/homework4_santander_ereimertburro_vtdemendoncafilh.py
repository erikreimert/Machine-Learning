import sklearn.svm as svm
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import numpy as np
import pandas as pd

##########################################Results

# 0.8557191054102763
# 0.8284909995178955

#################################################

# Load data
d = pd.read_csv('santander-customer-transaction-prediction/train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

def bagPredict(model, Xbags, Ybags, xTest):
    predict = np.empty((0, xTest.shape[0]))

    for xbag, ybag in  zip(Xbags, Ybags):

        tmodel = model
        tmodel.fit(xbag, ybag)
        prediction = tmodel.decision_function(xTest)
        predict = np.append(predict, [prediction], axis = 0)
        avgPredict = np.mean(predict, axis = 0)

        return avgPredict

def bagData(train, test, n):

    return np.split(train, n), np.split(test, n)

# Split into train/test folds
xTr, xTe, yTr, yTE = model_selection.train_test_split(X, y, test_size = 0.5, random_state = 0)

xTrBags, yTrBags = bagData(xTr, yTr, 50)

# Linear SVM
linearSVM =  svm.LinearSVC(random_state = 0, dual = False)

# Non-linear SVM (polynomial kernel)
polynomialSVM = svm.SVC(random_state = 0, kernel = 'poly')

# Apply the SVMs to the test set
yhat1 = bagPredict(linearSVM, xTrBags, yTrBags, xTe)
yhat2 = bagPredict(polynomialSVM, xTrBags, yTrBags, xTe)

# Compute AUC
auc1 = metrics.roc_auc_score(yTE, yhat1)
auc2 = metrics.roc_auc_score(yTE, yhat2)

print(auc1)
print(auc2)
