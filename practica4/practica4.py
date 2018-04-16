#!/opt/local/bin/python2.7
# -*- coding : utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import exp

def linear_kernel(x,l):
    # Initialize some useful values
    rx,cx=x.shape
    rl,cl=l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx,rl),dtype=float)

# ====================== YOUR CODE HERE ======================
# Instructions: return K(x,l) using a linear kernel
#               linear: < x, l>
    result = np.dot(x,l.T)
# =========================================================================
    return result

def polynomial_kernel(x,l, gamma=1,d=2,r=3):
    # Initialize some useful values
    rx,cx=x.shape
    rl,cl=l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx,rl),dtype=float)

# ====================== YOUR CODE HERE ======================
# Instructions: return K(x,l) using a polynomial kernel
#               polynomial: (gamma * < x, l > + r)^d
    result = pow((gamma * np.dot(x,l.T) + r),d)
# =========================================================================
    return result

def sigmoid_kernel(x,l, gamma=1,r=2):
    # Initialize some useful values
    rx,cx=x.shape
    rl,cl=l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx,rl),dtype=float)

# ====================== YOUR CODE HERE ======================
# Instructions: return K(x,l) using a sigmoid kernel
#               sigmoid: tanh(gamma * <x,l> + r)
# Hint: you may find useful numpy.tanh
    result = np.tanh(gamma * np.dot(x,l.T) + r)
# =========================================================================
    return result

def rbf_kernel(x,l, gamma = 3):
    # Initialize some useful values
    rx,cx=x.shape
    rl,cl=l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx,rl),dtype=float)

# ====================== YOUR CODE HERE ======================
# Instructions: return K(x,l) using an rbf kernel
#               rbf: exp(-gamma * |x-l|^2)
# Hint: use one loop

    for i,x in enumerate(x):
        for j,y in enumerate(l):
            result[i,j] = np.exp(-gamma * np.linalg.norm(x-y)**2)

# =========================================================================
    return result

def datasetParams(X, y):
    # You need to return the following variables correctly
    # try all values in [0.01, 0.03, 0.1, 0.3, 10, 30]
    optC = 0
    optSigma = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set
#               (see sklearn.cross_validation) with predefined values of
#               parameters.
#   Hint: you can use svm.score to compute the prediction error
    values = [0.01, 0.03, 0.1, 0.3, 10, 30]
    optC = 1
    optSigma = 1
    maxscore = -1
    for c in values:
        for g in values:
     		clf = svm.SVC(c, kernel='rbf', gamma=g)
     		clf.fit(X, y)
     		err = clf.score(X, y)
     		if err > maxscore:
     			optC = c
     			optSigma = g
     			maxscore = err

# =========================================================================
    return optC,optSigma
