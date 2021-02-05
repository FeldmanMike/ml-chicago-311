'''
Model implementations to use during training process

Note: this code was written as part of an ML course, where the focus
was on learning the mathematical foundations of common ML models. On this
project we were encouraged to write our own implementations of these models,
when possible. (In practice this meant writing code to implement ridge and
principal components regression, but using sk-learn to implement an SVM.)
'''
import math
import numpy as np
from numpy import linalg as la

def estimate_weights_using_svd(v, s, u_t, y, param=None, model='trunc_svd'):
    '''
    Estimate weights for regression model using the SVD
  
    Inputs:
        v, s, u: V, Sigma, and U transpose matrices found via
            the SVD on X
        y: vector of labels for X
        param: Hyperparameter for model
        model: which model to estimated; options include 'trunc_svd_ls',
            'svd_ls', and 'ridge'
    '''
    # Ridge regression
    if model == 'ridge':
        I = np.identity(s.shape[0])
        return v @ la.inv((s.T @ s) + (param * I)) @ s.T @ u_t @ y

    # Principal components regression
    if model == 'trunc_svd':
        s_trunc = np.copy(s)
        s_trunc[:, param:] = 0
        s_inv = la.pinv(s_trunc)
        return v @ s_inv @ u_t @ y

    # Ordinary least squares
    s_inv = la.pinv(s)
    return v @ s_inv @ u_t @ y
