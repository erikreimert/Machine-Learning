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
    faces =np.hstack((faces, ones))
    return faces


#Perutation to to SGD
def permutationData(X, y):
    permute = np.random.permutation(X.shape[0])
    Xpermute = X[permute]
    ypermute = y[permute]

    return Xpermute, ypermute


#Stochastic Gradient Descent with SoftMax Problem
#Stochastic Gradient Descent with SoftMax Problem
def SGD(X, y, batchSize = 100, epsilon = .1):
    #Initiate W like HW 2, adjust dimensions
    w =  np.random.randn(X.shape[1], 10) * .1

    #Epochs with randomization
    epochs = 300
    for i in range(epochs):
        X, y = permutationData(X, y)
        X_batches, y_batches = miniBatch(X, y, 100)

        #Softmax problem
        #Same idea as last hw
        #get print variable
        printvar = True
        for X_batch, y_batch in zip(X_batches, y_batches):
            z = softMax(X_batch, w)
            loss = fCE(y_batch, z)
            w = w - (epsilon * gradfCE(X_batch, y_batch, w))

            if printvar and i >= 280:
                printvar = False
                print("Epoch: ", i + 1, "Loss: ", np.mean(loss))

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
    #print(crossEntropy)

    return crossEntropy


#Gradient Cross Entropy
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

    return yhat



#Check accuracy of guesses
def percentAccuracy(y, yhat):
    accuracy = 0.0

    for i in range(0,y.shape[0]):
        if (y[i] == yhat[i]).all():
            accuracy = accuracy + 1
        else:
            continue

    return accuracy/y.shape[0]





##############################################################
#Data augmentation section

#this works
#shifts image
def shift(image):
    factor = [-5,-4,-3,-2,-1,1,2,3,4,5]
    changex = random.choice(factor)
    changey = random.choice(factor)

    transform = AffineTransform(translation=(changex,changey))
    return warp(image, transform, mode = "wrap")

#this works
#45 degree rotation
def r1 (image):
    return rotate(image, angle = 20)

#this works
#-45 degree rotation
def r2(image):
    return rotate(image, angle = -20)

# this works
#adds noise to the image
def noise(image):
    return random_noise(image)

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

def augment(trainingImages, ahhh):
    #dictionary with the augmenting functions
    trans = {1: r1, 2: r2, 3: shift, 4: noise, 5: scale}
    hold = ahhh
    i = 0
    # location = 0
    # location2 = 784
    for x in trainingImages:
        i +=1
        if (i==4999):
            print("Done augmenting data")
        # print(i)
        newimg = trans[random.randrange(1,5)](x)
        hold = np.append(hold, newimg)
    hold = hold.reshape(5000,784)
    return hold
###############################################################
    # where i get the stuff for the pdf
def pdfstuff(x , trainingImages):
    if(x == 1):
        plt.title("Rotation")
        rotate = r1(trainingImages)
        plt.imshow(rotate)
        plt.savefig(("Rotation"+ ".png"))

        plt.title("Original 1")
        plt.imshow(trainingImages)
        plt.savefig(("Original 1"+ ".png"))
    if(x == 2):
        plt.title("Shift")
        shifty = shift(trainingImages)
        plt.imshow(shifty)
        plt.savefig(("Shift"+ ".png"))

        plt.title("Original 2")
        plt.imshow(trainingImages)
        plt.savefig(("Original 2"+ ".png"))
    if(x==3):
        plt.title("Noise")
        noisey = noise(trainingImages)
        plt.imshow(noisey)
        plt.savefig(("Noise"+ ".png"))

        plt.title("Original 3")
        plt.imshow(trainingImages)
        plt.savefig(("Original 3"+ ".png"))
    if(x==4):
        plt.title("Scale")
        scaley = scale(trainingImages)
        plt.imshow(scaley)
        plt.savefig(("Scale"+ ".png"))

        plt.title("Original 4")
        plt.imshow(trainingImages)
        plt.savefig(("Original 4"+ ".png"))



if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")


    reshapeimages = trainingImages.reshape(5000, 28, 28)
    # print(reshapeimages.shape)
    empty = np.zeros((0,28,28))
    Xaug = augment(reshapeimages, empty)
    Yaug = testingLabels
    # print(trainingImages)
    # print(Xaug)

    # Append a constant 1 term to each example to correspond to the bias terms
    X_tr = reshapeAndAppend1s(trainingImages)
    X_te = reshapeAndAppend1s(testingImages)

    ##########for printing the images, not necessary to run
    # i = 0
    # for x  in reshapeimages:
    #     pdfstuff(i, x)
    #     if (i == 4):
    #         break
    #     i += 1


    W = SGD(X_tr, trainingLabels)
    yhat = classify(X_te, testingLabels, W)
    accuracy = percentAccuracy(testingLabels, yhat)

    print("Testing Loss: ", np.mean(fCE(testingLabels, softMax(X_te, W))))
    print("Testing Accuracy: ", accuracy)
    


    # Visualize the vectors
    for i in range(0,9):
        plotW = np.reshape(W[:784,i], (28,28))
        plt.imshow(plotW)
        plt.savefig("Weight{}.png".format(i))
