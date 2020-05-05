import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import copy
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import random
import math


epochs = 0 ;
batchSize = 0;
learningRate= 0;

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40 # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    W1 = w[:NUM_INPUT * NUM_HIDDEN].reshape((NUM_INPUT, NUM_HIDDEN))
    b1 = w[NUM_INPUT * NUM_HIDDEN: NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN].reshape((NUM_HIDDEN))
    W2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN:NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT].reshape((NUM_HIDDEN, NUM_OUTPUT))
    b2 = w[NUM_INPUT * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUT:].reshape(NUM_OUTPUT)

    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    w = []
    W1vect = W1.flatten()
    b1vect = b1.flatten()
    W2vect = W2.flatten()
    b2vect = b2.flatten()
    w = np.concatenate((W1vect, b1vect, W2vect, b2vect))

    return w


# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))

    return images, labels


#Calculation Process, will be helper later for training
def fPropagation(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = X.dot(W1) + b1
    h1 = ReLU(z1)
    z2 = h1.dot(W2) + b2
    yhat = softMax(z2)

    return z1,h1,z2,yhat


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    z1, h1, z2, yhat = fPropagation(X, w)

    cost = np.mean(-1 * np.sum(Y * np.log(yhat.transpose()), axis = 1))

    return cost


# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    z1, h1, z2, yhat = fPropagation(X, w)

    db2 = np.mean(yhat.transpose() - Y, axis = 0)
    dW2 = np.atleast_2d(yhat.transpose() - Y).transpose().dot(h1).transpose()
    g = ((yhat.transpose() - Y).dot(W2.transpose())) * derivReLU(z1)
    db1 = np.mean(g, axis = 0)
    dW1 = X.transpose().dot(np.atleast_2d(g))

    grad = pack(dW1, db1, dW2, db2)

    return grad


#Useful from HW3
#Permutation to to SGD
def permutationData(X, y):
    permute = np.random.permutation(X.shape[0])
    Xpermute = X[permute]
    ypermute = y[permute]

    return Xpermute, ypermute


#Create minibatch for SGD
def miniBatch(X, y, batchSize):
    num_batches = math.floor(X.shape[0]/batchSize)
    # print(num_batches)
    # print(X.shape[0])
    X_batches = np.split(X, num_batches)
    y_batches = np.split(y, num_batches)

    return np.array(X_batches), np.array(y_batches)


def softMax(z):
    Zexp = np.exp(z)
    sumZexp = np.atleast_2d(Zexp).sum(axis = 1)

    #Normalize and convert back to column vector
    normalizedZexp = (Zexp.transpose() / sumZexp)

    return normalizedZexp


def ReLU(z):
    z[z<=0] = 0

    return z


def derivReLU(z):
    z[z <= 0] = 0
    z[z > 0] = 1

    return z


def OHdecode(y):

    return np.array([np.argmax(i) for i in y])


def percentAccuracy(y, yhat):
    y = OHdecode(y)
    yhat = OHdecode(yhat)

    return np.mean(y==yhat)


# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
#Use same idea from hw 3
#Get the best hiperparameters and compare
def train (trainX, trainY, testX, testY, w, epch, batch, learn_rate, printvar = True):
    epochs = epch
    batchSize = batch
    learningRate = learn_rate
    b = 0.1
    W1,b1,W2,b2 = unpack(w)

    ws = []
    costArray = []
    for i in range(epochs):
        X, y = permutationData(trainX, trainY)
        X_batches, y_batches = miniBatch(X, y, batchSize)

        #Same idea as last 3
        for X_batch, y_batch in zip(X_batches, y_batches):
            grad = w * b + gradCE(X_batch, y_batch, w) * (1 - b)
            w = w - (learningRate * grad)

        cost = fCE(X,y,w)
        costArray.append(copy.deepcopy(cost))
        ws.append(copy.deepcopy(w))

        #get print variable
        if printvar and epochs - i <= 20:
            print("Epoch: ", i + 1, "Loss: ", np.mean(cost))

    __, __, __, yhat = fPropagation(testX, w)
    accuracy = percentAccuracy(testY, yhat.transpose())
    loss = fCE(trainX, trainY, w)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    return ws, loss, accuracy


#finds the best best Hyperparameters
def findBestHyperparameters(trainX, trainY, testX, testY, w):
    epochTrain = [10, 20, 30, 40, 50, 75, 100]
    batchSizeTrain = [10, 20,25, 50, 100]
    learningRateTrain = [.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    optimal = {'Epoch': 0, 'Batch': 0, 'learnRate': 0.0, 'Loss': 8.0, 'Accuracy': 0.0}

    for x in range(1,10):
        epochs = random.choice(epochTrain)
        learningRate = random.choice(learningRateTrain)
        batchSize = random.choice(batchSizeTrain)
        ws, loss, accuracy = train(trainX, trainY, testX, testY, w, epochs, batchSize, learningRate, printvar = False)
        print("Hyper Parameters in run ",x, "\nEpochs: ", epochs, "\nLearning Rate: ", learningRate, "\nBatch Size: ", batchSize)

        if loss < optimal.get('Loss') and accuracy > optimal.get('Accuracy'):
            optimal.update(Epoch = epochs, Batch = batchSize, learnRate = learningRate, Loss = loss, Accuracy = accuracy)

    print(optimal)

    return optimal.get('Epoch'), optimal.get('batchSize'), optimal.get('learnRate')



if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))

    #findBestHyperparameters(trainX, trainY, testX, testY, w)
    #Train the network and obtain the sequence of w's obtained using SGD.
    ws, loss, accuracy = train(trainX, trainY, testX, testY, w, 100, 50, 0.001, printvar = True)
