import numpy as np
import math
from sigmoidpredict import sigmoid,predict
from scipy import linalg


#   computeCostLogReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for logistic regression using regularization.
def computeCostLogReg(theta, X, y,lambda1):
    # Initialize some useful values
    m,n = X.shape
    # You need to return the following variable correctly
    J = 0.0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. You may find useful numpy.log
#               and the sigmoid function.
#

    h = sigmoid(X.dot(theta))
    J1 = -y.T.dot(np.log(h))
    J2 = (1-y.T).dot(np.log(1 - h))
    J =  (1.0/m) * ((J1 - J2) + ((lambda1/2*m)*(np.dot(theta.T,theta)-(theta[0]*theta[0]))))

# =============================================================

    return J

#   gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,n = X.shape

    # ====================== YOUR CODE HERE ======================
    xTraspuesta = X.T
    for i in range(0,iterations):
        h = sigmoid(np.dot(X, theta)) - y
        j = np.dot(xTraspuesta, h)
        auxTheta = theta[0]
        theta = theta - ((alpha * (j + np.dot(lambda1,theta))) / m)
        theta[0] += alpha * (1/m) * lambda1 * auxTheta
    # ============================================================

    return theta

#   computeCostLinReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for linear regression using regularization.
def computeCostLinReg(theta, X, y,lambda1):
    # Initialize some useful values
    m,n = X.shape
    # You need to return the following variable correctly
    J = 0.0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
    h = X.dot(theta)
    J = (1.0 / (2 * m)) * ((h - y).T.dot((h-y)) + ((lambda1/2*m)*(np.dot(theta.T,theta)-(theta[0]*theta[0]))))
# =============================================================

    return J

#   gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,n = X.shape

    # ====================== YOUR CODE HERE ======================
    xTraspuesta = X.T
    for i in range(0,iterations):
        h = np.dot(X, theta) - y
        j = np.dot(xTraspuesta, h)
        auxTheta = theta[0]
        theta = theta - ((alpha * (j + np.dot(lambda1,theta))) / m)
        theta[0] += alpha * (1/m) * lambda1 * auxTheta

    # ============================================================

    return theta


#   normalEqn(X,y) computes the closed-form solution to linear
#   regression using the normal equations with regularization.
def normalEqnReg(X, y,lambda1):
    # Initialize some useful values
    m,n = X.shape
    # You need to return the following variable correctly
    theta = np.zeros((n,1))


# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression with regularization and put the result in theta.
#
    identidad = np.identity(n)
    identidad[0][0] = 0
    lambda1 = np.dot(lambda1,identidad)
    theta = np.dot(linalg.inv(np.dot(X.T,X) + lambda1),np.dot(X.T,y))

# ============================================================

    return theta
