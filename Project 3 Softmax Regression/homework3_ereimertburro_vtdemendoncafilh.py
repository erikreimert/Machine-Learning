import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rotate, rescale
from skimage.util import random_noise
# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.


#Add bias, same as last time. Adjusted to size of matrix
def reshapeAndAppend1s (faces):
    ones = np.ones((1, len(faces[:,0,0])))
    faces =np.reshape(faces, (784, len(faces[:,0,0])))
    return np.vstack((faces, ones))


#Perutation to to SGD
def permutationData(X, y):
    permute = np.random.permutation(X.shape[0])
    Xpermute = X[permute]
    ypermute = y[permute]
    return Xpermute, ypermute


#Stochastic Gradient Descent with SoftMax Problem
def SGD(X, y, batchSize = 100, epsilon = .1):
    #Sigma in the formula, also used last time
    sigma = .1
    w =  np.random.randn(X.shape[1], 10) * sigma

    #Epochs with randomization
    epochs = X.shape[1]/batchSize - 1
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

    return yhat

#Check accuracy of guesses
def percentAccuracy(y, yhat):
    newY = [np.where(i == 1)[0][0] for i in y]
    newYhat = [np.where(i == 1)[0][0] for i in yhat]

    return np.mean(newY == newYhat)

##############################################################
#Data augmentation section

#shifts image
def shift(image):
    transform = AffineTransform(translation=(-50,0))
    return warp(image, transform, mode = "wrap")

#45 degree rotation
def r1 (image):
    return rotate(x, angle = 45)

#-45 degree rotation
def r2(image):
    return rotate(x, angle = -45)

#adds noise to the image
def noise(image):
    return random_noise(x)

#rescales testingImages
def scale(image):
    return rescale(image, 0.25, anti_aliasing =False)


def augment(trainingImages):
    #dictionary with the augmenting functions
    trans = {1: r1, 2: r2, 3: shift, 4: noise, 5: scale}
    Xaug = np.array()  ########create an array to store the new augmented pics, not clear wtf im doing here tbh
    for x in trainingImages:
        newimg = trans[randint(1,5)](x)
        Xaug = np.append(Xaug, newimg)
    return Xaug
###############################################################
    # where i get the stuff for the pdf
def pdfstuff(x , trainingImages):
    if(x == 0):
        plt.title("Original")
        plt.imshow(x)
        plt.savefig(("Original"+ ".png"))
    if(x == 1):
        plt.title("Rotation")
        rotate = rotate(x, angle = 45)
        plt.imshow(rotate)
        plt.savefig(("Rotation"+ ".png"))
    if(x == 2):
        plt.title("Scale")
        scale = shifter(x)
        plt.imshow(scale)
        plt.savefig(("Scale"+ ".png"))
    if(x==3):
        plt.title("Noise")
        noise = random_noise(x)
        plt.imshow(noise)
        plt.savefig(("Noise"+ ".png"))
    if(x==4):
        plt.title("Scale")
        scale = rescale(image, 0.25, anti_aliasing =False)
        plt.imshow(scale)
        plt.savefig(("Scale"+ ".png"))



if __name__ == "__main__":
    # Load data
    trainingImages = np.load("small_mnist_train_images.npy")
    trainingLabels = np.load("small_mnist_train_labels.npy")
    testingImages = np.load("small_mnist_test_images.npy")
    testingLabels = np.load("small_mnist_test_labels.npy")



    # Append a constant 1 term to each example to correspond to the bias terms
    X_tr = reshapeAndAppend1s(trainingImages)
    X_te = reshapeAndAppend1s(testingImages)

    Xaug = augment(trainingImages)

    int i = 0
    for x  in trainingImages:
        pdfstuff(i, trainingImages)
        i++

    yaug = trainingLabels #given they are inputted in order and return only one copy the order for yaug should be the same

    W = SGD(X_tr, trainingLabels)



    yhat = predictClassifications(X_te, testingLabels, W)
    accuracy = percentAccuracy(testingLabels, yhat)

    print("Testing Loss: ", np.mean(fCE(testingLabels, softMax(X_te, W))))
    print("Testing Accuracy: ", accuracy)

    # Visualize the vectors
    # ...
