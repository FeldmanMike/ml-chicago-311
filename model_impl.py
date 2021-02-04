'''
Model implementations to use during training process

Note: this code was made as part of a Machine Learning course, where the focus
was on learning the mathematical foundations of common ML models. On this
project we were instructed to write our own implementations of these models,
when possible. (In practice this meant writing code to implement ridge and
principal components regression, but using sk-learn to implement an SVM.)
'''
import math
import numpy as np
from numpy import linalg as la

def estimate_weights_using_svd(v, s, u_t, y, sing_vals_to_keep=None,
                               lamb=None, model='trunc_svd_ls'):
    '''
    Estimate weights for regression model using the SVD
  
    Inputs:
        v, s, u: V, Sigma, and U transpose matrices found via
            the SVD on X
        y: vector of labels for X
        sing_vals_to_keep: first n singular values to keep non-zero
        lamb: ridge regularization parameter
        model: which model to estimated; options include 'trunc_svd_ls',
            'svd_ls', and 'ridge'
    '''
    # Ridge regression
    if model == 'ridge':
        I = np.identity(s.shape[0])
        return v @ la.inv((s.T @ s) + (lamb * I)) @ s.T @ u_t @ y

    # Principal components regression
    if model == 'trunc_svd_ls':
        s_trunc = np.copy(s)
        s_trunc[:, sing_vals_to_keep:] = 0
        s_inv = la.pinv(s_trunc)
        return v @ s_inv @ u_t @ y

    # Ordinary least squares
    s_inv = la.pinv(s)
    return v @ s_inv @ u_t @ y


def root_mean_squared_error(true_y, predicted_y):
    '''
    Calculated root mean-squared error
    '''
    return math.sqrt(np.sum((true_y - predicted_y)**2) / len(true_y))
