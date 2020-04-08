import numpy as np
from PIL import Image

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    ones = np.ones((1, 5000))
    faces =np.reshape(faces, (2304, 5000))
    return np.vstack((faces, ones))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    return np.mean((Xtilde.transpose().dot(w) - y)**2)/2


# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    return np.mean((np.dot(Xtilde.transpose(), w) - y)).dot(Xtilde) + alpha*(w**2)/2

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    ones = np.ones((1,2304))
    Xtilde = np.vstack((Xtilde, ones))
    Wtilde = np.linalg.solve(Xtilde.dot(np.transpose(Xtilde)), Xtilde.dot(y))
    loss = fMSE(w,Xtilde,y)
    print('Wtilde: ', Wtilde,"Loss: ", loss)
    return (Wtilde, loss)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    #####IMPLEMENT Gradient descent
    return gradientDescent(Xtilde, y, alpha = ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    w = np.random.randn(1,2304) * 0.01
    loss = 0
    for i in range(0,T):
        w = w - EPSILON*gradfMSE(w, Xtilde, y)
        loss = lossArray + gradfMSE(w, Xtilde, y)
    return (w,loss)

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    # ...
