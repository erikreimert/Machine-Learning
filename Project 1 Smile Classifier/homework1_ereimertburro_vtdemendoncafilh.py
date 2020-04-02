import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A, B) - C

def problem3 (A, B, C):
    return A * B + C.T()

def problem4 (x, y):
    return np.inner(x, y)

def problem5 (A):
    return np.zeros_like(A)

def problem6 (A):
    return np.ones((A.shape[0],1))

def problem7 (A, alpha):
    return A + (alpha* np.eye(A.shape))

def problem8 (A, i, j):
    return A[i][j]

def problem9 (A, i):
    B = np.sum(A, axis = 1, keepdims = True)
    return B.item(0, i)

def problem10 (A, c, d):
    if c > d:
        return np.mean(A[np.nonzero(A <= c and A >=d)])
    elif d > c:
        return np.mean(A[np.nonzero(A >= c and A <=d)])

def problem11 (A, k):
    eigValues = np.linalg.eig(A)[1]
    resultCol = A - k
    return eigValues[:, resultCol:]

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    resultTranspose = np.linalg.solve(A.T, x.T)
    return resultTranspose.T
