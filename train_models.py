'''
Train models and find model with best performance on validation set

Note: this code was written as part of an ML course, where the focus
was on learning the mathematical foundations of common ML models. The goal of
the project was to demonstrate knowledge learned in the course, and so we were
encouraged to prioritize implementing models that were focused on.

Due to these instructions and time constraints, we were not exhaustive in our
model selection process below. If we had more time and computational resources,
we would have also tried logistic regression, Naive Bayes, and random forest
classifiers.
'''
import pandas as pd
import pipeline as pl
import numpy as np
from sklearn.svm import LinearSVC
from model_impl import estimate_weights_using_svd



def calc_model_performance(model_params, v, s, u_t, val_feat, val_targ):
    '''
    Calculate performance of models on validation set.
    
    Inputs:
        model_params: Dictionary with models to train as the keys and the
        hyperparameters to use as the values
        v, s, u_t: V, Sigma, and U transpose matrices calculated by performing
        SVD on the training set
        val_feat, val_targ: Features and target from validation set

    Output:
        Numpy array with model, hyperparameter, and validation error
    '''
    results = np.zeros((len(model_params), 2))

    # NEED TO ADD INPUT FOR REGRESSION VS CLASSIFIATION!!!!!!!
    i = 0
    for model, param in model_params.items():
        if model == 'ridge':
            weights = estimate_weights_using_svd(v, s, u_t, 




def go():
    '''
    Train models and return model that minimizes error on validation set
    '''
    # Read in dataframe from pickle file
    chi_311 = pd.read_pickle("../pickle_files/chi_311.pkl")

    # Split into train and test sets (80/20) using random seed (which is
    # defined in the wrapper function)
    train_311, test_311 = pl.create_train_test_split(chi_311)

    # Normalize num_children column
    cols_to_normalize = ['NUM_CHILDREN']
    train_311_norm, scaler = pl.normalize(train_311, cols_to_normalize)
    test_311_norm, scaler = pl.normalize(test_311, cols_to_normalize, scaler)

    # Split training set into model training and model validation sets (80/20
    # split)
    train_311_norm, val_311_norm = pl.create_train_test_split(train_311_norm)

    # Define feature and target columns in train, validation, and target sets
    train_feat = np.array(train_311_norm.iloc[:, :-1], dtype=np.float)
    train_targ = np.array(pd.DataFrame(train_311_norm.iloc[:, -1]),
                          dtype=np.float)
    test_feat = np.array(test_311_norm.iloc[:, :-1], dtype=np.float)
    test_targ = np.array(pd.DataFrame(test_311_norm.iloc[:, -1]),
                         dtype=np.float)
    val_feat = np.array(val_311_norm.iloc[:, :-1],
                        dtype=np.float)
    val_targ = np.array(pd.DataFrame(val_311_norm.iloc[:, -1]), dtype=np.float)

    # Run SVD on training set
    u, s, v_t = la.svd(train_feat, full_matrices=False)




if __name__ == '__main__':
    go()
