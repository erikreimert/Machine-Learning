import numpy as np
from PIL import Image

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    ones = np.ones((1, len(faces[:,0,0])))
    faces =np.reshape(faces, (2304, len(faces[:,0,0])))
    return np.vstack((faces, ones))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.

###########################################################Is it np.dot() or the multiplication
def fMSE (w, Xtilde, y):
    return np.mean((Xtilde.transpose().dot(w) -y)**2)/2
###########################################################

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    print("Xtilde shape: ", Xtilde.shape, "w shape: ", w.shape, "y shape: ", y.shape)
    return    np.mean((Xtilde.transpose().dot(w) - y).dot(Xtilde)) + alpha * np.mean(w.transpose().dot(w))/2

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    print(Xtilde.shape)
    #ones = np.ones((1,2304))
    #Xtilde =  np.vstack((Xtilde, ones))
    Wtilde = np.linalg.solve(Xtilde.dot(Xtilde.transpose()), Xtilde.dot(y))
    print('Wtilde: ', Wtilde, "shape: ", Wtilde.shape )
    return Wtilde


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    #Choose random starting position
    w = np.random.randn(2305) * 0.01
    gradientDescent(Xtilde, y)
# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    #####IMPLEMENT Gradient descent
    w = np.random.randn(1,2305) * 0.01
    gradientDescent(Xtilde, y, alpha = ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    print("Xtilde shape: ",Xtilde.shape, "y shape: ", y.shape)
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    w = np.random.randn(2305) * 0.01


    for i in range(0,T):
        w = w - EPSILON*gradfMSE(w, Xtilde, y)


    return w

######TO DO WORK
#Get W shape (1,2304)
#Reshape to 48x48
# US plt.imshow()
#Get 5 most gruesome errors
def vizualize(w):
    w = w[:-1]
    w.reshape((48,48))
    plt.imshow(w)


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


    im = Image.open('img.jpg')
    im.show('image',im)
