import sklearn.svm as svm
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import numpy as np
import pandas as pd


# Load data
d = pd.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

def bagData(train, test, n):
    return np.split(train, n), np.split(test, n)

def bagPredict(model, X_bags, y_bags, x_test):
    predictions = np.empty((0, x_test.shape[0]))
    for x_bag, y_bag in  zip(X_bags, y_bags):
        temp_model = model
        temp_model.fit(x_bag, y_bag)
        prediction = temp_model.decision_function(x_test)
        predictions = np.append(predictions, [prediction], axis = 0)
    average_predictions = np.mean(predictions, axis = 0)
    return average_predictions



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
