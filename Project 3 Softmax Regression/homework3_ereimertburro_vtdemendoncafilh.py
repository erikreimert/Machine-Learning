import numpy as np
import matplotlib.pyplot as plt
import skimage
import random
from random import randrange
from skimage.transform import rotate, rescale, warp, AffineTransform
from skimage.util import random_noise
# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.


#Add bias, same as last time. Adjusted to size of matrix
def reshapeAndAppend1s (faces):
    ones = np.ones((faces.shape[0],1))
    faces =np.append(faces, ones, axis = 1)
    return faces


#Perutation to to SGD
def permutationData(X, y):
    permute = np.random.permutation(X.shape[0])
    Xpermute = X[permute]
    ypermute = y[permute]
    return Xpermute, ypermute


#Stochastic Gradient Descent with SoftMax Problem
def SGD(X, y, batchSize = 100, epsilon = .1):
    #Initiate W like HW 2, adjust dimensions
    w =  np.random.randn(X.shape[1], 10) * .1

    #Epochs with randomization
    epochs = 200
    for i in range(epochs):
        X, y = permutationData(X, y)
        X_batches, y_batches = miniBatch(X, y, 100)

        #Softmax problem
        #Same idea as last hw
        for X_batch, y_batch in zip(X_batches, y_batches):
            z = softMax(X_batch, w)
            loss = fCE(y_batch, z)
            w = w - (epsilon * gradfCE(X_batch, y_batch, w))


            print("Epoch: ", i, "Loss: ", np.mean(loss), "Weights: ", w)

    return w


#Logistic Regression aka SoftMax
def softMax(X, w):
    #Row vector 1x10
    z = np.dot(X, w)
    Zexp = np.exp(z)
    sumZexp = Zexp.sum(axis = 1)

    #Normalize and convert back to column vector
    normalizedZexp = (Zexp.transpose() / sumZexp).transpose()

    return normalizedZexp


#Create minibatch for SGD
def miniBatch(X, y, batchSize):
    num_batches = np.floor(X.shape[0]/batchSize)
    X_batches = np.split(X, num_batches)
    y_batches = np.split(y, num_batches)

    return np.array(X_batches), np.array(y_batches)


#Cross Entropy
def fCE(y, yhat):
    crossEntropy =  -1 / y.shape[0] * np.sum(y * np.log(yhat), axis = 1)
    print(crossEntropy)

    return crossEntropy


#Gradient Cross Entropy
def gradfCE(X, y, w):
    yhat = softMax(X, w)
    grad = -1 / np.shape(X)[0] * np.dot(X.transpose(), y - yhat)

    return grad



#Make a guess picking highest probability
def classify(X, y, w):
    z = X.dot(w)
    yhat = np.zeros(np.shape(z))
    encoded_idx = z.argmax(axis = 1)
    yhat[np.arange(np.shape(z)[0]), encoded_idx] = 1
    print(yhat)
    return yhat



#Check accuracy of guesses
def percentAccuracy(X,y, yhat, w):
    newY = [np.where(i == 1)[0][0] for i in y]
    newYhat = [np.where(i == 1)[0][0] for i in yhat]

    return np.mean(newY == newYhat)





##############################################################
#Data augmentation section

#shifts image
#this works
def shift(image):
    if (randint(1,4) >= 2):
        changex = random.randrange(-5,-1)
        changey = random.randrange(-5,-1)
    else:
        changex = random.randrange(1,5)
        changey = random.randrange(1,5)

    transform = AffineTransform(translation=(changex,changey))
    return warp(image, transform, mode = "wrap")

#45 degree rotation
#this works
def r1 (image):
    return rotate(x, angle = 20)

#-45 degree rotation
#this works
def r2(image):
    return rotate(x, angle = -20)

#adds noise to the image
# this works
def noise(image):
    return random_noise(x)

#yes worky
# rescales testingImages
def scale(image):
    emptyimg = np.zeros_like(image)
    factor = [.75, .82142857142857142857142857142857, 1.1071428571428571428571428571429, 1.2142857142857142857142857142857]
    selection = random.choice(factor)
    if (selection > 1): #if true then crop selection
        image = rescale(image, selection, anti_aliasing =True)
        size = 28
        location = random.randrange(1,3)
        final = location+size
        emptyimg = image[location:final,location:final]
        return emptyimg
    else: #if false superimpose in random part of 28x28 array
        image = rescale(image, selection, anti_aliasing =True)
        size = int(28*selection)
        location = random.randrange(1,7)
        final = location+size
        emptyimg[location:final,location:final] = image
        return emptyimg

def augment(trainingImages):
    #dictionary with the augmenting functions
    trans = {1: r1, 2: r2, 3: shift, 4: noise, 5: scale}
    Xaug = np.array(784,500) #make (784,5000) array
    for x in trainingImages:
        newimg = trans[randint(1,5)](x)
        #change new img to (784, )
        Xaug = np.append(Xaug, newimg)
    return Xaug
###############################################################
    # where i get the stuff for the pdf
def pdfstuff(x , trainingImages):
    if(x == 0):
        plt.title("Original")
        plt.imshow(trainingImages)
        plt.savefig(("Original"+ ".png"))
    if(x == 1):
        plt.title("Rotation")
        rotate = r1(trainingImages)
        plt.imshow(rotate)
        plt.savefig(("Rotation"+ ".png"))
    if(x == 2):
        plt.title("Shift")
        shifty = shift(trainingImages)
        plt.imshow(shifty)
        plt.savefig(("Shift"+ ".png"))
    if(x==3):
        plt.title("Noise")
        noisey = noise(trainingImages)
        plt.imshow(noisey)
        plt.savefig(("Noise"+ ".png"))
    if(x==4):
        plt.title("Scale")
        scaley = scale(trainingImages)
        plt.imshow(scaley)
        plt.savefig(("Scale"+ ".png"))



if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")


    reshapeimages = trainingImages.reshape(5000, 28, 28)


    # Append a constant 1 term to each example to correspond to the bias terms
    Xaug = augment(reshapeimages)
    # X_tr = reshapeAndAppend1s(trainingImages)
    # X_te = reshapeAndAppend1s(testingImages)

    #
    # i = 0
    # for x  in reshapeimages:
    #     pdfstuff(i, x)
    #     if (i == 5):
    #         break
    #     i += 1

    # yaug = trainingLabels #given they are inputted in order and return only one copy the order for yaug should be the same

    # W = SGD(X_tr, trainingLabels)
    #
    #
    # yhat = classify(X_te, testingLabels, W)
    # accuracy = percentAccuracy(testingLabels, yhat)
    #
    # print("Testing Loss: ", (fCE(testingLabels, softMax(X_te, W))))
    # print("Testing Accuracy: ", accuracy)

    # Visualize the vectors
    # ...
