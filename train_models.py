'''
Train models and find model with best performance on validation set

Note: this code was written as part of an ML course, where the focus
was on learning the mathematical foundations of common ML models. The goal of
the project was to demonstrate knowledge learned in the course, and so we were
encouraged to prioritize implementing models that were focused on.

Due to these instructions and time constraints, we were not exhaustive in our
model selection process below. If we had more time and computational resources,
we would have also tried logistic regression, Naive Bayes, and random forest
classifiers, as well as some polynomial feature transformations.
'''
import pandas as pd
import numpy as np
import pickle
from numpy import linalg as la
from sklearn.svm import LinearSVC
from model_impl import estimate_weights_using_svd
import pipeline as pl

MODEL_PARAMS_REGR = {
        'ridge': [0, 1, 5, 10, 15, 20, 25, 50, 75, 100, 1000, 10000],
        'trunc_svd': [1, 2, 5, 10, 50, 75, 100, 125, 150, 175, 189]
        }

MODEL_PARAMS_CLSF = {
        'ridge': [0, 1, 5, 10, 15, 20, 25, 50, 75, 100, 1000, 10000],
        'trunc_svd': [1, 2, 5, 10, 50, 75, 100, 125, 150, 175, 189],
        'svm_linear': [1, 10, 100, 1000]
        }

HOURS_IN_WEEK = 168.0

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
            'clsf' for classification.

    Output:
        Name of model with lowest validation error, hyperparameter of best
            model, and pandas DataFrame with model, hyperparameter, and
            validation error,sorted from lowest to highest validation error
    '''
    # results = np.zeros((len(model_params), 4))
    results = {
        'model': [],
        'hyperparam': [],
        'task': [],
        'validation_err': []
    }
    train_feat = u_t.T @ s @ v.T

    for model, param_list in model_params.items():
        for param in param_list:
            # results[i, 1] = model
            # results[i, 2] = param
            # results[i, 3] = task
            results['model'].append(model)
            results['hyperparam'].append(param)
            results['task'].append(task)

            print('Estimating training weights for...', model)
            print('Hyperparam is...', param)
            if model in ['ridge', 'trunc_svd']:
                weights = estimate_weights_using_svd(v, s, u_t, train_targ,
                                                     param=param, model=model)

                print('Evaluating performance...')
                if task == 'regr':
                    results['validation_err'].append(pl.root_mean_squared_error(
                                                        val_targ,
                                                        val_feat @ weights))
                if task == 'clsf':
                    results['validation_err'].append(pl.classification_err(
                                                         val_targ,
                                                         val_feat @ weights))

            elif model == 'svm_linear' and task == 'clsf':
                svc = LinearSVC(C=param)
                svc.fit(train_feat, train_targ.reshape(-1))

                print('Evaluating performance...')
                results['validation_err'].append(pl.classification_err(val_targ,
                                                     svc.predict(val_feat)))

    print('Converting results to sorted dataframe...')
    results_df = pd.DataFrame(data=results).sort_values(
                        by=['validation_err'], ignore_index=True)

    print('Results dataframe is...', results_df)

    best_model = results_df.loc[0, 'model']
    best_model_param = results_df.loc[0, 'hyperparam']

    return best_model, best_model_param, results_df


def go():
    '''
    Train models and return model that minimizes error on validation set
    '''
    # Read in dataframe from pickle file
    print('Reading in pickle data file...')
    chi_311 = pd.read_pickle("./pickle_files/chi_311.pkl")

    # Split into train and test sets (80/20) using random seed (which is
    # defined in the wrapper function)
    print('Creating train and test sets...')
    train_311, test_311 = pl.create_train_test_split(chi_311)

    # Normalize num_children column
    print('Normalizing data...')
    cols_to_normalize = ['NUM_CHILDREN']
    train_311_norm, scaler = pl.normalize(train_311, cols_to_normalize)
    test_311_norm, scaler = pl.normalize(test_311, cols_to_normalize, scaler)

    # Split training set into model training and model validation sets (80/20
    # split)
    train_311_norm, val_311_norm = pl.create_train_test_split(train_311_norm)

    # Define feature columns in train, validation, and target sets
    print('Defining feature columns...')
    train_feat = np.array(train_311_norm.iloc[:, :-1], dtype=np.float)
    test_feat = np.array(test_311_norm.iloc[:, :-1], dtype=np.float)
    val_feat = np.array(val_311_norm.iloc[:, :-1],
                        dtype=np.float)

    # Targets for regression tasks
    print('Defining target columns for regression tasks...')
    train_targ_regr = np.array(pd.DataFrame(train_311_norm.iloc[:, -1]),
                               dtype=np.float)
    test_targ_regr = np.array(pd.DataFrame(test_311_norm.iloc[:, -1]),
                              dtype=np.float)
    val_targ_regr = np.array(pd.DataFrame(val_311_norm.iloc[:, -1]),
                             dtype=np.float)

    # Targets for classification tasks
    print('Defining target columns for classification tasks...')
    train_targ_clsf = np.where(train_targ_regr <= HOURS_IN_WEEK, 1, -1)
    test_targ_clsf = np.where(test_targ_regr <= HOURS_IN_WEEK, 1, -1)
    val_targ_clsf = np.where(val_targ_regr <= HOURS_IN_WEEK, 1, -1)

    # Run SVD on training set
    print('Running SVD on training set...')
    u, s, v_t = la.svd(train_feat, full_matrices=False)

    # Grid search for regression task
    print('Running grid search for regression task...')
    best_model_regr, \
    best_param_regr, \
    results_df_regr = calc_model_performance(MODEL_PARAMS_REGR, v_t.T,
                                             np.diag(s), u.T, train_targ_regr,
                                             val_feat, val_targ_regr,
                                             task='regr')

    print('Best model for regression is...', best_model_regr)
    print('Its hyperparameter is...', best_param_regr)
    print('Retraining best model to pickle and calc test error...')

    best_weights = estimate_weights_using_svd(v_t.T, np.diag(s),
                                              u.T, train_targ_regr,
                                              param=best_param_regr,
                                              model=best_model_regr)

    test_error_regr = pl.root_mean_squared_error(test_targ_regr,
                                                 test_feat @ best_weights)

    print('Its test error is...', test_error_regr)

    regr_model_info = {
        'model_task': 'regr',
        'model_type': best_model_regr,
        'model_weights': best_weights,
        'model_test_error': test_error_regr
    }
    
    print('Sending to pickle...')
    with open('pickle_files/best_regr_model.pkl', 'wb') as f:
        pickle.dump(regr_model_info, f)

    print('Running grid search for classification task...')
    best_model_clsf, \
    best_param_clsf, \
    results_df_clsf = calc_model_performance(MODEL_PARAMS_CLSF, v_t.T,
                                             np.diag(s), u.T, train_targ_clsf,
                                             val_feat, val_targ_clsf,
                                             task='clsf')

    print('Best model for classification is...', best_model_clsf)
    print('Its hyperparameter is...', best_param_clsf)
    print('Retraining best model and calculating test error...')
    if best_model_clsf == 'svm_linear':
        svc = LinearSVC(C=best_param_clsf)
        svc.fit(train_feat, train_targ_clsf.reshape(-1))
        best_weights_clsf = None
        model_obj = svc
        test_error_clsf = pl.classification_err(test_targ_clsf,
                                                svc.predict(test_feat))
    else:
        best_weights_clsf = estimate_weights_using_svd(v_t.T, np.diag(s), u.T,
                                                       train_targ_clsf,
                                                       param=best_param_clsf,
                                                       model=best_model_clsf)
        model_obj = None
        test_error_clsf = pl.classification_err(test_targ_clsf,
                                                test_feat @ best_weights_clsf)

    print('Its test error is...', test_error_clsf)

    clsf_model_info = {
        'model_task': 'clsf',
        'model_type': best_model_clsf,
        'model_obj': model_obj,
        'model_weights': best_weights_clsf,
        'model_test_error': test_error_clsf
    }

    print('Sending to pickle...')
    with open('pickle_files/best_clsf_model.pkl', 'wb') as f:
        pickle.dump(clsf_model_info, f)
    
    print('Done!')

    
if __name__ == '__main__':
    go()
