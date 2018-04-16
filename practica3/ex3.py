## Exercise 3: Regularization
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the exercise
#  which covers regularization with linear and logistic regression.
#
#  You will need to complete the following functions in this exericse in the file practica3.py:
#     costFunctionLogReg
#     gradientDescentLogReg
#     costFunctionLinReg
#     gradientDescentLinReg
#     normalEqnReg
#
#  For this exercise, you will not need to change any code in this file.

def loadData(filename):
    print 'Loading data ...\n'
    separator=','
    data=[]
    for line in open(filename,'r'):
        data.append([float(w) for w in line.split(separator)])
    X = np.array([[1]+list(line[:-1]) for line in data],dtype=float)
    y = np.array([[line[-1]] for line in data],dtype=float)
    return X,y

def plotData(X,y):
    m,n=X.shape
    pos=[i for i in range(len(y)) if y[i]==1]
    neg=[i for i in range(len(y)) if y[i]==0]
    plt.plot(X[pos,1],X[pos,2],'+')
    plt.plot(X[neg,1],X[neg,2],'o')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('y=1, y=0')
    plt.show()



def mapFeature(X1,X2):
    degree = 6
    m,n=X1.shape
    out = np.ones((m,1))
    for i in range(degree):
        for j in range(i+2):
            out = np.concatenate((out,np.power(X1,i+1-j) * np.power(X2,j)),1)
    return out


def plotDecisionBoundary(theta, X, y):
    m,n=X.shape
    if n <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:,1])-2,  max(X[:,1])+2]
        # Calculate the decision boundary line
        plot_y = [(-theta[1,0]*i - theta[0,0])/theta[2,0] for i in plot_x]
        # Plot
        plt.plot(plot_x, plot_y)
    else:
        print 'Printing contour plot...'
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = np.dot(mapFeature(np.array([[u[i]]]), np.array([[v[j]]])),theta)
        z = z.T # important to transpose z before calling contour
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, 0)
        print 'Done printing'


import numpy as np
from sigmoidpredict import sigmoid,predict
from practica3 import computeCostLogReg, gradientDescentLogReg,computeCostLinReg, gradientDescentLinReg,normalEqnReg
import matplotlib.pyplot as plt

## ==================== Loading data ====================

X,y = loadData('ex3data.txt')

## ==================== Plotting data ====================


plotData(X, y)


# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled


X = mapFeature(X[:,1:2], X[:,2:3])


iterations = 5000
alpha = 0.03
theta= np.zeros((28,1))
theta = gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1=1)
cost = computeCostLogReg(theta, X, y,lambda1=1)

# Print theta to screen
print 'Cost at theta found by gradient descent ', cost
print theta

plotDecisionBoundary(theta,X,y)
plotData(X, y)






