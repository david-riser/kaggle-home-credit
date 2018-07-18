#
# File:   1.0-train.py 
# Author: David Riser 
# Date:   July 18, 2018
#
# I am defining a common format for 
# the ML models used in this competition
# based on the kernel below. 
# https://www.kaggle.com/eliotbarr/stacking-test-sklearn-xgboost-catboost-lightgbm

import numpy as np
import pandas as pd
import lightgbm 
import xgboost 

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold



class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        self.clf = clf(**params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.clf.fit(x_train, y_train, 
                     eval_set=[(x_train, y_train), (x_valid, y_valid)]
                     )

    def predict(self, x):
        return self.clf.predict_proba(x, num_iteration=self.clf.best_iteration_)[:,1]

class XgboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['seed'] = seed
        self.clf = clf(**params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.clf.fit(x_train, y_train, 
                     eval_set=[(x_train, y_train), (x_valid, y_valid)],
                     early_stopping_rounds=50,
                     verbose=20)

    def predict(self, x):
        return self.clf.predict_proba(x, ntree_limit=self.clf.best_ntree_limit)[:,1]


def load_features(path_to_data, version, sample_size=10000):

    # Load both training and testing
    train = pd.read_csv(path_to_data + version +'-features-train.csv', nrows=sample_size, compression='gzip')
    test = pd.read_csv(path_to_data + version + '-features-test.csv', nrows=sample_size, compression='gzip')

    # Drop
    labels = train['TARGET']

    # Save for predictions
    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']

    # The way I constructed the testing set in the
    # feature building notebook leaves it with an empty TARGET column.
    train.drop(columns=['test', 'SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    test.drop(columns=['test', 'SK_ID_CURR', 'TARGET'], axis=1, inplace=True)

    return train, labels, test, train_ids, test_ids

if __name__ == '__main__':

    # Define constants and configurations.
    path_to_data = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    path_to_preds = '../../data/predictions/'
    version = '1.1'
    sample_size = 10000
    SEED = 8675309

    train, labels, test, train_ids, test_ids = load_features(path_to_data, version, sample_size)
    oof_preds = np.zeros(len(train))
    sub_preds = np.zeros(len(test))

    # Best set of parameters for XGBoostClassifier
    xgb_params = {
        'learning_rate':0.1,
        'n_estimators':10000,
        'max_depth':4,
        'min_child_weight':5,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'objective':'binary:logistic',
        'nthread':4,
        'seed':SEED,
        'scale_pos_weight':2.5,
        'reg_alpha':1.2,
        'early_stopping_rounds':50,
        'verbose':20,
        'eval_metric':'auc'
    }

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, val_index in kf.split(train):
        x_train, y_train = train.iloc[train_index].values, labels[train_index]
        x_valid, y_valid = train.iloc[val_index].values, labels[val_index]

        clf = XgboostWrapper(xgboost.XGBClassifier, seed=SEED, params=xgb_params)

        clf.fit(x_train, y_train, x_valid, y_valid)

        oof_preds[val_index] = clf.predict(x_valid)
        sub_preds += clf.predict(test.values)

    # Predict the test dataset
    print('Total ROC AUC = %.4f' % roc_auc_score(labels, oof_preds))

