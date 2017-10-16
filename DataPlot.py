import matplotlib . pyplot as plt
import numpy as np

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data_internal (X, y):
    x_min , x_max = X[ : , 0 ]. min () - .5 , X[ : , 0 ]. max () + .5
    y_min , y_max = X[ : , 1 ]. min () - .5 , X[ : , 1 ]. max () + .5
    xx , yy = np. meshgrid (np. linspace (x_min , x_max , 100) ,
                            np. linspace (y_min , y_max , 100))
    plt. figure ()
    plt. xlim (xx.min () , xx. max ())
    plt. ylim (yy.min () , yy. max ())
    ax = plt. gca ()
    ax. plot (X[y == 0 , 0] , X[y == 0 , 1] , 'ro ', label = 'Class 1')
    ax. plot (X[y == 1 , 0] , X[y == 1 , 1] , 'bo ', label = 'Class 2')
    plt. xlabel ('X1 ')
    plt. ylabel ('X2 ')
    plt. title ('Plot data ')
    plt. legend (loc = 'upper left', scatterpoints = 1 , numpoints = 1)
    return xx , yy

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def plot_data (X, y):
    xx , yy = plot_data_internal (X, y)
    plt. show ()

##
# ll: 1d array with the average likelihood per data point
#

def plot_ll (ll ):
    plt. figure ()
    ax = plt. gca ()
    plt. xlim (0 , len(ll) + 2)
    plt. ylim (min(ll) - 0.1 , max(ll) + 0.1)
    ax. plot (np. arange (1 , len(ll) + 1) , ll , 'r-')
    plt. xlabel ('Steps ')
    plt. ylabel ('Average log - likelihood ')
    plt. title ('Plot Average Log - likelihood Curve ')
    plt. show ()

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict : function that recives as input a feature matrix and returns a 1d
# vector with the probability of class 1.

def plot_predictive_distribution (X, y, predict, weights):
    xx , yy = plot_data_internal (X, y)
    ax = plt. gca ()
    X_predict = np. concatenate (( xx. ravel (). reshape (( -1 , 1)) ,
    yy. ravel (). reshape (( -1 , 1))) , 1)
    Z = predict(X_predict, weights)
    Z = Z. reshape (xx. shape )
    cs2 = ax. contour (xx , yy , Z, cmap = 'RdBu', linewidths = 2)
    plt. clabel (cs2, fmt='%2.1f', colors='k', fontsize = 14)
    plt. show ()

##
# l: hyper - parameter for the width of the Gaussian basis functions
# Z: location of the Gaussian basis functions
# X: points at which to evaluate the basis functions

def expand_inputs (l, X, Z):
    X2 = np. sum (X**2 , 1)
    Z2 = np. sum (Z**2 , 1)
    ones_Z = np. ones (Z. shape [ 0 ])
    ones_X = np. ones (X. shape [ 0 ])
    r2 = np. outer (X2 , ones_Z ) - 2 * np. dot(X, Z.T) + np. outer (ones_X , Z2)
    return np.exp ( -0.5 / l**2 * r2)

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict : function that recives as input a feature matrix and returns a 1d
# vector with the probability of class 1.

def plot_predictive_distribution_expand(X, y, predict, expand_input, l, X_train, weights):
    xx , yy = plot_data_internal (X, y)
    ax = plt. gca ()
    X_predict = np. concatenate (( xx. ravel (). reshape (( -1 , 1)) ,
    yy. ravel (). reshape (( -1 , 1))) , 1)
    X_predict = expand_input(l, X_predict, X_train)
    Z = predict(X_predict, weights)
    Z = Z. reshape (xx. shape )
    cs2 = ax. contour (xx , yy , Z, cmap = 'RdBu', linewidths = 2)
    plt. clabel (cs2, fmt='%2.1f', colors='k', fontsize = 14)
    plt. show ()