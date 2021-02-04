'''
Functions to automate ML pipeline
'''
import datetime as dt
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def read_data(path):
    '''
    Read a dataset into a pandas dataframe

    Inputs:
        path (string): A string representing the file path to a CSV file

    Outputs:
        df: A pandas dataframe
    '''
    return pd.read_csv(path)


def explore_df(df):
    '''
    Get summary stats and sample of dataframe, by calling
        explore_df_summary_stats and explore_df_sample functions
    '''
    explore_df_summary_stats(df)
    print("--------------------------------------------------------------")
    explore_df_sample(df)


def explore_df_summary_stats(df):
    '''
    Given a dataframe, print its shape and various summary statistics

    Inputs:
        df: A pandas dataframe

    Outputs:
        - Sentence stating # of rows and columns in dataframe
        - A dataframe with summary statistics for all quantitative columns
        - A listing of all columns with NA or Null values (and how many)
        - A listing of all columns with negative minimum values (and what
            that value is)
    '''
    rows, columns = df.shape
    print("The dataframe has {:,} rows and {:,} columns.".format(rows,
                                                                 columns))
    print("--------------------------------------------------------------")

    df_stats = df.describe()
    print("Detailed descriptive statistics of quantitative columns:")
    display(df_stats)

    print("--------------------------------------------------------------")

    print("Quantitative columns with null/NA values:")
    for col in df_stats.columns:
        num_null = rows - df_stats[col]['count']
        if num_null > 0:
            print("\nColumn: {}".format(col))
            print("Number of null/NA values: {}".format(num_null))

    print("--------------------------------------------------------------")

    print("Quantitative columns with negative minimum values:")
    for col in df_stats.columns:
        min_val = df_stats[col]['min']
        if min_val < 0:
            print("\nColumn: {}".format(col))
            print("Min value: {:,}".format(min_val))


def explore_df_sample(df):
    '''
    Given a dataframe, print the types of each column, along with several
        rows of the actual dataframe

    Input:
        df: A pandas dataframe

    Output:
        A pandas Series with the type of each column, and a smaller version
            of the input dataframe (with the first 5 rows)
    '''
    print("Column types:\n")
    print(df.dtypes)

    print("--------------------------------------------------------------")

    print("First five rows of dataframe:")
    display(df.head())


def explore_correlations(df):
    '''
    Display the correlations between the features in tabular and graphical
        form.

    Input:
        df: A pandas dataframe

    Outputs:
        A dataframe of correlations between features, and a plot visualizing
            those correlations
    '''
    df_corrs = df.corr()
    print("Correlations between variables:")
    display(df_corrs)

    print("Plot displaying correlations (brighter box = higher correlation):")
    plt.matshow(df_corrs)


def explore_time_series(df, date_col, y_var_col, count_plot=True, xlabel=None,
                        ylabel=None, xlim=None, ylim=None, title=None,
                        title_font=16):
    '''
    Plot the change in a variable over time.

    Inputs:
        df: A pandas dataframe
        date_col (string): Column in dataframe that includes dates to
            plot against.
        y_var_col (string): Column in dataframe to plot as y-variable
        count_plot (boolean): When True, this function plots the frequency of
            the y-var against time. When False, it plots the sum of the y-var
            against time.
        xlabel/y_label (strings): labels for x and y axes
        xlim/ylim (tuples): Min and max values for axes
        title (string): Title of plot

    Output:
        A matplotlib line plot
    '''
    df_gb = df.groupby(df[date_col].dt.date)
    if count_plot:
        to_plot = df_gb[y_var_col].size()
    else:
        to_plot = df_gb[y_var_col].sum()

    ax = to_plot.plot(x=date_col, y=y_var_col)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.set_title(label=title, fontsize=title_font);


def create_train_test_split(df, test_size=0.2, train_size=None,
                            random_state=100, shuffle=True):
    '''
    Create train and test sets

    Inputs:
        df: A pandas dataframe
        test_size (float or int): Proportion of dataset to include in test
            split
        train_size (float or int): Proportion of dataset to include in train
            split
        random_state (int): Random seed
        shuffle (boolean): Whether or not to shuffle data before splitting

    Outputs:
        Tuple including training dataset and testing dataset
    '''
    df_train, df_test = train_test_split(df, test_size=test_size,
                                         random_state=random_state,
                                         train_size=train_size,
                                         shuffle=shuffle)

    return df_train, df_test


def impute_missing_values(df, impute_col, df_to_impute_from=None,
                          missing_value_list=None):
    '''
    Impute missing values in a column using the median value of the column
        from the training set

    Inputs:
        df: A pandas dataframe
        impute_col (string): Column for which we will be imputing missing
            values
        df_to_impute_from: pandas dataframe we will be using as our dataframe
            to impute from. If one is not provided, this is assumed to be the
            first df provided
        missing_value_list (list): Values for which we consider data to be
            missing. If this is not provided, only NaN values are treated
            as missing

    Outputs:
        pandas Series with missing values imputed
    '''
    if df_to_impute_from is None:
        col_median = df[impute_col].median()
    else:
        col_median = df_to_impute_from[impute_col].median()

    df = df.copy(deep=True)
    df[impute_col] = df[impute_col].fillna(value=col_median)

    if missing_value_list:
        df[impute_col] = df[impute_col].replace(to_replace=missing_value_list,
                                                value=col_median)

    return df[impute_col]


def normalize(df, cols, scaler=None):
    '''
    Use scaler to normalize columns of dataset.

    Inputs:
        df: a pandas dataframe
        cols (list): names of columns in dataset to be normalized
        scaler: a StandardScaler object

    Outputs:
        A dataframe with normalized features and a scaler object
    '''
    # If scaler not passed to function, normalize columns based on their
    # own mean and SD
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(pd.DataFrame(df[cols]))
    else:
        normalized_features = scaler.transform(pd.DataFrame(df[cols]))

    other_col_names = [col for col in df.columns if col not in cols]
    other_cols = df[other_col_names]

    normalized_df = pd.DataFrame(normalized_features, columns=cols,
                                 index=df.index)
    normalized_df[other_col_names] = other_cols

    return normalized_df, scaler

# Adapted from:
#https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e
def one_hot_encode(df, cols_to_encode, prefix_sep="__", train_dummy_cols=None,
                   train_processed_cols=None):
    '''
    Perform one-hot encoding of categorical variables from dataframe

    Inputs:
        df: A pandas dataframe included the columns you would like to
            one-hot encode
        cols_to_encode (list): Column names from dataframe that you want
            to one-hot encode
        train_dummy_cols (list): Dummy columns from your training dataframe.
            If this is provided, then any column not in this list will be
            removed from the final dataframe (and any column not in your final
            dataframe that is in this list will be added to the final
            dataframe)
        train_processed_cols (list): Column names of your training dataframe.
            If this is provided, then the column order of the final dataset
            will be changed to reflect this list.

    Outputs:
        A three-element tuple with the following components:
            - A pandas dataframe with categorical columns converted to columns
            with dummy variables;
            - A list of the dummy columns from the final dataframe; and
            - A list of all columns from the final dataframe
    '''
    df_with_dummies = pd.get_dummies(df, columns=cols_to_encode,
                                     prefix_sep=prefix_sep)

    # Dummy columns in test dataset should match dummy columns from training
    # dataset
    if train_dummy_cols:
        for col in df_with_dummies.columns:
            if ("__" in col) and (col.split("__")[0] in cols_to_encode) \
                    and col not in train_dummy_cols:
                df_with_dummies.drop(col, axis=1, inplace=True)

        for col in train_dummy_cols:
            if col not in df_with_dummies.columns:
                df_with_dummies[col] = 0

    # Reorder columns in dataset to match order of columns from training set
    if train_processed_cols:
        df_with_dummies = df_with_dummies[train_processed_cols]

    return df_with_dummies, \
        [col for col in df_with_dummies if "__" in col
         and col.split("__")[0] in cols_to_encode], \
        list(df_with_dummies.columns[:])


def discretize_feature(col, bins, labels, include_lowest=True,
                       right=True):
    '''
    Convert continuous feature to discrete

    Inputs:
        col_name (pandas Series): column with data to be discretized
        feature (pandas Series): data to be converted from continuous to
            discrete
        bins (list of ints or floats): lowest/highest values for discrete
            categories
        labels (list of strings): labels for bins
        include_lowest (boolean): whether the first interval should be left-
            inclusive or not
        right (boolean): whether bins includes the rightmost edge or not
    '''
    return pd.cut(col, bins=bins, labels=labels, include_lowest=include_lowest,
                  right=right)


def build_model(features, target, model, params=None):
    '''
    Fit a machine learning model to a dataset and print time required
        to train it.

    Inputs:
        features: a pandas dataframe with model features
        target: a pandas Series with data the model is predicting
        model: model object you'd like to fit
        params (dict): parameters you'd like to set for the model
        time_to_fit (boolean): If True, function prints out the amount of
            time it takes to fit model

    Outputs:
        Fitted model object
    '''
    if len(features.columns) == 1:
        features = np.array(features).reshape(-1, 1)

    model.set_params(**params)

    start = dt.datetime.now()
    fitted_model = model.fit(features, target)
    stop = dt.datetime.now()

    print("Time Elapsed to Train: {:.4f} seconds".format(
        (stop - start).total_seconds()))

    return fitted_model


def evaluate_classifier_accuracy(fitted_model, test_features, test_target):
    '''
    Calculate the accuracy of a model based on a test set

    Input:
        fitted_model: A fitted model object
        test_features: A pandas dataframe with model features
        target: a pandas Series with data the model is predicting

    Output:
        A float - the fraction of correctly classified samples from the
            test set
    '''
    if len(test_features.columns) == 1:
        test_features = np.array(test_features).reshape(-1, 1)

    return accuracy_score(test_target,
                          fitted_model.predict(test_features))


def root_mean_squared_error(true_y, predicted_y):
    '''
    Calculated root mean-squared error
    '''
    return math.sqrt(np.sum((true_y - predicted_y)**2) / len(true_y))


def classification_err(y_true, y_hat):
    '''
    Calculate classification error

    Inputs:
        y_true: 1d numpy array with true target values
        y_hat: 1d numpy array with predicted target values
    '''
    y_hat = np.sign(y_hat).reshape(-1)
    test_matches = np.equal(y_hat, y_true.reshape(-1))
    return (y_hat.size - np.sum(test_matches))/y_hat.size


def perform_grid_search(models, param_grid, train_features, train_target,
                        test_features, test_target):
    '''
    Fit 1 or more machine learning models to a dataset, and evaluate on a
        test set

    Inputs:
        models (dict): model names as keys, and model objects as values
        param_grid (dict): the model names (from models dict) are the keys and
            each value is another dictionary; for that dictionary, each key is
            a model parameter name, and each value is a list of parameters
            values that will be used in the grid search
        train_features: a pandas dataframe with model features used to train
            model
        train_target: a pandas Series with data the model is predicting (to
            train model)
        test_features: A pandas dataframe with model features to test fitted
            model
        target: a pandas Series with data the model is predicting (to
            evaluate model)

    Output:
        pandas Dataframe with model names, parameters, and
            accuracy score (with rank by accuracy score)
    '''
    results_list = []
    for model_key in models:
        for params in ParameterGrid(param_grid[model_key]):
            print("Training model:", model_key, "|", params)

            fitted_model = build_model(train_features, train_target,
                                       models[model_key],
                                       params=params)

            model_accuracy = evaluate_classifier_accuracy(fitted_model,
                                                          test_features,
                                                          test_target)

            results = {"model_name": model_key,
                       "test_accuracy": model_accuracy,
                       "parameters": params}

            results_list.append(results)

    results_df = pd.DataFrame(results_list).sort_values(by=['test_accuracy'],
                                                        ascending=False)

    results_df['rank'] = np.arange(len(results_df)) + 1

    return results_df


# Get the best model from the grid above
def retrain_best_grid_model(grid_df, features, target):
    '''
    Get best model from grid search

    Inputs:
        grid_df: Dataframe

    '''
    return []


def create_coefficient_df(fitted_model, train_features):
    '''
    Return a pandas dataframe with a column with the feature names for a fitted
        model, and a column with their associated coefficient values.
        Dataframe is sorted high to low by absolute coefficient value

    Inputs:
        fitted_model: A model object that has been fitted to data
        train_features (pandas df): the features used to train the model

    Output:
        a pandas dataframe with feature names and coefficient values, sorted
            high-to-low by absolute coefficient value
    '''
    coef_df = pd.DataFrame(data={"feature_names": train_features.columns,
                                 "slope_coefficients": fitted_model. \
                                                       coef_[0].tolist()})

    return coef_df.iloc[coef_df['slope_coefficients'].abs().argsort()][::-1]
