import math
import numpy as np
from scipy import linalg

#   normalEqn(X,y) computes the closed-form solution to linear
#   regression using the normal equations.
def normalEqn(X, y):
    # Initialize some useful values
    m,p=X.shape
    theta = np.zeros((p,1))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#
    theta = np.dot(linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))

# ============================================================

    return theta

#   featureNormalize(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.
#   Set mu and sigma to a column vector containing the mean and
#   standard variation of each feature in X, respectively
#   (you may want to check the documentation of numpy.std and numpy.mean)

def featureNormalize(X):
    # Initialize some useful values
    m,p=X.shape
    # You need to set these values correctly
    X_norm = X
    mu = np.zeros((p,))
    sigma = np.ones((p,))
# ====================== YOUR CODE HERE ======================
    mu = np.mean(X,0)
    sigma = np.std(X,0)

    X_norm = (X-mu)/sigma


# ============================================================
    return X_norm, mu, sigma


#   computeCost(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
def computeCost(X,y,theta):
    # Initialize some useful values
    m,p=X.shape
    # You need to return the following variable correctly
    J=0.0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.

    h = X.dot(theta)
    J = (1.0 / (2 * m)) * (h - y).T.dot((h-y))


# =========================================================================
    return J


#   gradientDescent(x, y, theta, alpha, num_iters) updates theta by
#   taking iterations gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, iterations):
    # Initialize some useful values
    m,p=X.shape
    # ====================== YOUR CODE HERE ======================
    xTraspuesta = X.T
    for i in range(0,iterations):
        h = np.dot(X, theta) - y
        j = np.dot(xTraspuesta, h) / m
        theta = theta - alpha * j
    # ============================================================

    return theta

#   predict(newdata, mu, sigma) predicts the value of the variable y for
#   previously unseen data (in general, an m by p numpy.array)

def predict(newdata,mu,sigma,theta):
    m,p=newdata.shape
    value=np.zeros((m,))


    # ====================== YOUR CODE HERE ======================
    one = np.ones((m,1))
    newdata = (newdata-mu)/sigma
    newdata = np.concatenate([one,newdata],1)
    value = newdata.dot(theta)

    # ============================================================

    return value
