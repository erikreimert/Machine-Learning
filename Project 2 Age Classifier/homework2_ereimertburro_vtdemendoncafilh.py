import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    #print("Xtilde shape: ", Xtilde.shape[1], "w shape: ", w.shape, "y shape: ", y.shape)
    return ((Xtilde.dot((Xtilde.transpose().dot(w) - y)))/Xtilde.shape[1]) + (alpha * np.sum(w**2)/Xtilde.shape[1])

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    #print(Xtilde.shape)
    Wtilde = np.linalg.solve(Xtilde.dot(Xtilde.transpose()), Xtilde.dot(y))
    #print('Wtilde: ', Wtilde, "shape: ", Wtilde.shape, "bias ", Wtilde[-1] )
    return Wtilde


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    #Choose random starting position
    w = np.random.randn(2305) * 0.01
    return gradientDescent(Xtilde, y)
# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    #####IMPLEMENT Gradient descent
    w = np.random.randn(1,2305) * 0.01
    return gradientDescent(Xtilde, y, alpha = ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    #print("Xtilde shape: ",Xtilde.shape, "y shape: ", y.shape)
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
def visualize(w, pictitle):
    wimg = np.reshape(w[:2304],(48,48))
    plt.title(pictitle)
    plt.imshow(wimg)
    plt.savefig((pictitle+ ".png"))

# def worstVal(worst):
#     shits = sorted(worst, key = abs).reverse() # worst values ordered by absolute value high to low
#     for x in range(0,4):
#         shitnum = str(x)
#         visualize(shits[x], ("Egregious", shitnum))
#     return shits[:5]

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")



    w1 = method1(Xtilde_tr, ytr)
    loss1tr = fMSE(w1,Xtilde_tr, ytr)
    bias1tr = w1[-1]
    print("Loss for W1: ", loss1tr)
    print("Bias for W1: ", bias1tr)
    visualize(w1, 'Method 1')

    w1test = method1(Xtilde_te, yte)
    loss1te = fMSE(w1test, Xtilde_te, yte)
    bias1te = w1test[-1]
    print("\nLoss for W1 Test: ", loss1te)
    print("Bias for W1 Test: ", bias1te)
    visualize(w1test, 'Method 1 Test')

    w2 = method2(Xtilde_tr, ytr)
    loss2tr = fMSE(w2,Xtilde_tr, ytr)
    bias2tr = w2[-1]
    print("\nLoss for W2: ",loss2tr)
    print("Bias for W2: ", bias2tr)
    visualize(w2, 'Method 2')

    w2test = method2(Xtilde_te, yte)
    loss2te = fMSE(w2test, Xtilde_te, yte)
    bias2te = w2test[-1]
    print("\nLoss for W2 Test: ", loss2te)
    print("Bias for W2 Test: ", bias2te)
    visualize(w2test, 'Method 2 Test')

    w3 = method3(Xtilde_tr, ytr)
    loss3tr = fMSE(w3,Xtilde_tr, ytr)
    bias3tr = w2[-1]
    print("\nLoss for W3: ",loss3tr)
    print("Bias for W3: ",bias3tr)
    visualize(w2, 'Method 3')

    w3test = method3(Xtilde_te, yte)
    loss3te = fMSE(w3test, Xtilde_te, yte)
    bias3te = w2test[-1]
    print("\nLoss for W3 Test: ", loss3te)
    print("Bias for W3 Test: ", bias3te)
    visualize(w3test, 'Method 3 Test')
    # theShits = worstVal(w3test)
    # Print("These are the worst values... ew: ", theShits)


    im = Image.open('img.jpg')
    im.show('image',im)
