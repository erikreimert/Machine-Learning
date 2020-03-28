import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A, B) - C

def problem3 (A, B, C):
    return A * B + C.getT()

def problem4 (x, y):
    return np.inner(x, y)

def problem5 (A):
    return np.zeros_like(A)

def problem6 (A):
    return np.ones_like(A)

def problem7 (A, alpha):
    return A + alpha.dot(np.eye(A.shape)  #fix

def problem8 (A, i, j):
    return A.item(j,i)

def problem9 (A, i): #probably wrong
    B = np.sum(A, axis = 1, keepdims = True)
    return B.item(0, i)

def problem10 (A, c, d):
    return ...

def problem11 (A, k):
    return ...

def problem12 (A, x):
    return ...

def problem13 (A, x):
    return ...
