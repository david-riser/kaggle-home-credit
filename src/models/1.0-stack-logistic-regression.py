#
# 1.0-stack-logistic-regression.py
# Author: David Riser
# Date: July 21, 2018
#
# Using the predictions from the train models
# that are stored in data/predictions logistic regression
# is trained as a stacker.

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def build_name(path, version, model):
    ''' Setup the naming scheme used in this project
    to access files.
    '''
    return str(path + version + '-' + model + '.csv')

def load_model_data(path_to_train, path_to_test, version, model):
    ''' Load the predictions from the model specified that
    are from the training script. This function also renames the
    prediction column in testing to the model name to match the
    training data.
    '''

    train = pd.read_csv(build_name(path_to_train, version, model))
    test = pd.read_csv(build_name(path_to_test, version, model))

    # Type correctly the ID numbers.
    train['SK_ID_CURR'] = train['SK_ID_CURR'].astype('int32')
    test['SK_ID_CURR'] = test['SK_ID_CURR'].astype('int32')

    # The column for prediction should be named
    # with the name of the model.  For submission
    # it is just called TARGET.
    test.rename(columns={'TARGET':model}, inplace=True)
    return train, test

def combine_models(train, test):

    ''' This function creates a dataframe that contains all model
    predictions for testing and training based on two dictionaries
    that have as keys the model names (these have to match).
    '''

    # The models have to be the same
    # in both training and testing.
    assert(train.keys() == test.keys())

    # Retrieve the names
    models = train.keys()

    all_train = train[train.keys()[0]]
    all_test  = test[test.keys()[0]]

    for model in train.keys()[1:]:
        all_train = pd.merge(all_train, train[model], on=['SK_ID_CURR', 'TARGET'])
        all_test = pd.merge(all_test, test[model], on='SK_ID_CURR')

    return all_train, all_test

def stack():

    # Define constants and configurations.
    path_to_test   = '../../data/submissions/'
    path_to_output = '../../data/submissions/'
    path_to_train  = '../../data/predictions/'
    version        = '1.1'
    SEED           = 8675309

    train = {}
    test  = {}

    models = ['xgboost', 'lightgbm']
    for model in models:
        train[model], test[model] = load_model_data(path_to_train,
                                                    path_to_test,
                                                    version, model)

        print('Loaded model %s with train shape %s and test shape %s.'
              % (model, train[model].shape, test[model].shape))

    # Combine the models
    all_train, all_test = combine_models(train, test)
    print('Training shape after merge: ', all_train.shape)
    print('Testing shape after merge: ', all_test.shape)

    # At this point we could introduce features from the
    # original training/testing dataset to improve score.

    # Drop SK_ID_CURR because it may contain some
    # information that we don't want to assume will
    # apply in the testing set.  These will be appended
    # later for the submission.
    train_ids, test_ids = all_train['SK_ID_CURR'], all_test['SK_ID_CURR']
    all_train.drop(columns=['SK_ID_CURR'], inplace=True)
    all_test.drop(columns=['SK_ID_CURR'], inplace=True)
    print('Training shape after dropping ID: ', all_train.shape)
    print('Testing shape after dropping ID: ', all_test.shape)

    # Setup empties for preds.
    oof_preds = np.zeros(len(all_train))
    sub_preds = np.zeros(len(all_test))

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, valid_index in kf.split(all_train):
        x_train, y_train = all_train.iloc[train_index][models].values, all_train.iloc[train_index]['TARGET'].values
        x_valid, y_valid = all_train.iloc[valid_index][models].values, all_train.iloc[valid_index]['TARGET'].values

        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        
        oof_preds[valid_index] = clf.predict_proba(x_valid)[:,1]
        sub_preds += clf.predict_proba(all_test)[:,1] / n_folds

    # Summarize
    print('Total ROC AUC = %.4f' % roc_auc_score(all_train['TARGET'].values, oof_preds))

    # Predict the test set for submission.
    submission = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':sub_preds})
    submission.to_csv(path_to_output + version + '-stack-logistic-regression.csv', index=False)

if __name__ == '__main__':
    stack()
