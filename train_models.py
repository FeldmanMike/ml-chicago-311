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


def calc_model_performance(model_params, v, s, u_t, train_targ, val_feat,
                            val_targ, task='regr'):
    '''
    Calculate performance of models on validation set.
    
    Inputs:
        model_params: Dictionary with models to train as the keys and the
        hyperparameters to use as the values
        v, s, u_t: V, Sigma, and U transpose matrices calculated by performing
            SVD on the training set
        train_targ: Target from training set
        val_feat, val_targ: Features and target from validation set
        task: Modeling task to be performed. Either 'regr' for regression or
            'class' for classification.

    Output:
        pandas DataFrame with model, hyperparameter, and validation error,
            sorted from lowest to highest validation error
    '''
    results = np.zeros((len(model_params), 4))
    train_feat = u_t.T @ s @ v.T

    i = 0
    for model, param in model_params.items():
        results[i, 1] = model
        results[i, 2] = param
        results[i, 3] = task

        print('Estimating training weights for...', model)
        print('Hyperparam is...', param)
        if model in ['ridge', 'trunc_svd']:
            weights = estimate_weights_using_svd(v, s, u_t, train_targ,
                                                 param=param, model=model)

            print('Evaluating performance...')
            if task == 'regr':
                results[i, 3] = pl.root_mean_squared_error(val_targ,
                                                           val_feat @ weights)
            if task == 'class':
                results[i, 3] = pl.classification_err(val_targ,
                                                      val_feat @ weights)

        elif model == 'svm_linear' and task == 'class':
            svc = LinearSVC(C=param)
            svc.fit(train_feat, train_targ.reshape(-1))

            print('Evaluating performance...')
            results[i, 4] = pl.classification_err(val_targ,
                                                  svc.predict(val_feat))

    print('Converting results to sorted dataframe...')
    # NEED TO PRINT BEST MODEL AND ADD OPTION TO PICKLE IT!!
    return pd.DataFrame(data=results, columns=['model', 'hyporparam',
                                               'validation_err']).sort.values(
                        by=['validation_err'], inplace=True)


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
