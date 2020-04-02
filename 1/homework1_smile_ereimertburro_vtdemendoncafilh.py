import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# import import ipdb; ipdb.set_trace()

def fPC (y, yhat): #this takes in a vector of ground-truth labels and corresponding vector of guesses, and then computes the accuracy (PC). The implementation (in vectorized form) should only take 1-line.
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y): #this takes in a set of predictors, a set of images to run it on, as well as the ground-truth labels of that set. For each image in the image set, it runs the ensemble to obtain a prediction. Then, it computes and returns the accuracy (PC) of the predictions w.r.t. the ground-truth labels.

 #Initialize counters as 0 in shape of y(true value)
    counter = np.zeros(y.shape)

    # print (predictors)
    for x in predictors:
        r1, c1, r2, c2 = x

        #Compare if pixel (r1,c1) is brighter than (r2,c2)
        x_diff = X[:,r1,c1] - X[:,r2,c2]

        #Success if (r1,c1) greater than (r2,c2)
        x_diff[x_diff > 0] = 1
        x_diff[x_diff <= 0] = 0

        #increase counte rin case of success
        counter = counter + x_diff

    #define the mean
    mean = counter / len(predictors)

    #Check if prediction is above 0.5 and classify it
    mean[mean > 0.5] = 1
    mean[mean <= 0.5]= 0


    #Calculate the accuracy using fPC between true values y and our mean (slide lecture 2)
    return fPC(y, mean)


def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):

    predictors = []
    maxAccuracy = []

    for m in range(0,4):
        maxAccuracy = 0
        bestPixelPair = None
        for r1 in range(0,23):
            for c1  in range(0,23):
                for r2  in range(0,23):
                    for c2  in range(0,23):
                        #skip iteration if theyre the same
                        if (r1,c1) == (r2,c2):
                            continue
                        #skip iteration if theyre already in the set
                        if (r1,c1,r2,c2) in predictors:
                            continue

                        newPredictors = predictors + list(((r1,c1,r2,c2),))
                        measuredAccuracy = measureAccuracyOfPredictors(newPredictors, trainingFaces, trainingLabels)

                        if measuredAccuracy > maxAccuracy:
                            maxAccuracy = measuredAccuracy
                            bestPixelPair = (r1,c1,r2,c2)

        predictors.append(bestPixelPair)

    r1,c1,r2,c2 = bestPixelPair
    print('best Pixel Pair ', bestPixelPair, 'with training Accuracy ',maxAccuracy, "and testing accuracy", measuredAccuracy, "\n")

        # print(predictors)
    return predictors

#i spent two hours trying to figure out how to do this and then i saw that you wrote it in the other function, i laughed and cried a bit
def visualize(predictors, testingFaces):
    # Show an arbitrary test image in grayscale
    im = testingFaces[0,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')

    for x in predictors:
        r1,c1,r2,c2 = x
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Display the merged result
    plt.show()


#analyze training/testing accuracy changes as a function of number of examples n ∈ {400, 800, 1200, 1600, 2000} (implement this in a for-loop)
def analyze (trainingFaces, trainingLabels, testingFaces, testingLabels):

    test = [400, 800, 1200, 1600, 2000]

    predictorsStorage = []
    accuracyStorage = []
    for x in test:

        #prints best accuracy, best pixel for test with n amount of samples
        print("Testing for n =", x, "\n")
        predictorsStorage = stepwiseRegression(trainingFaces[:x],trainingLabels[:x], testingFaces[:x], testingLabels[:x])
        #get each element break down the tuple of 4 into two tuples of two.
        #make r1c1 green and r2c2 red
    visualize(predictorsStorage, testingFaces)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    analyze(trainingFaces,trainingLabels, testingFaces, testingLabels)
    # stepwiseRegression(trainingFaces,trainingLabels, testingFaces, testingLabels)
