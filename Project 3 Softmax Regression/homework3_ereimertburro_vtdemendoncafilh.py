import numpy as np
import matplotlib.pyplot as plt

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    # z = X.transpose().dot(w), where w is all 10 emotions
    z = trainingImages.transpose().dot()
    epsilon =  - tx.dot(w)
    pass

#stochastic gradient descent
def SGD():
    pass


#augment the data set by changing each item a bit
#translation, rotation, scaling,random noise
def augmentation(trainingImages):
    # do a rotation
    # trainingImages concatenate rotation
    # new training labels = training labels concatenate training labels
    # do a translation
    #trainingImages concatenate translation
    #new training labels = new training labels concatenate training labels
    pass

#loss function
def fce(n, ytilde, y): ####################is this good?!?!?!
    return -1*(1/n)*np.sum(np.sum(y.dot(np.log(ytilde)), initial = 0), initial = 1)

def fPC(y, yhat):
    return np.mean(y == yhat)

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")

    # Append a constant 1 term to each example to correspond to the bias terms
    # ...

    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)

    # Visualize the vectors
    # ...


############
def compute_stoch_gradient(y, tx, w):
    #Compute a stochastic gradient for batch data.
    e = y - tx.dot(w)
    return (-1/y.shape[0])*tx.transpose().dot(e)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma):
    #"""Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_epochs):
        for minibatch_y,minibatch_x in batch_iter(y,tx,batch_size):
            w = ws[n_iter] - gamma * compute_stoch_gradient(minibatch_y,minibatch_x,ws[n_iter])
            ws.append(np.copy(w))
            loss = y - tx.dot(w)
            losses.append(loss)

    return losses, ws
