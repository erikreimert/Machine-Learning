import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat): #this takes in a vector of ground-truth labels and corresponding vector of guesses, and then computes the accuracy (PC). The implementation (in vectorized form) should only take 1-line.
    pass

def measureAccuracyOfPredictors (predictors, X, y): #this takes in a set of predictors, a set of images to run it on, as well as the ground-truth labels of that set. For each image in the image set, it runs the ensemble to obtain a prediction. Then, it computes and returns the accuracy (PC) of the predictions w.r.t. the ground-truth labels.
    pass

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = False
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
