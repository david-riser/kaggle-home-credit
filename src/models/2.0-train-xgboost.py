#
# 2.0-train-xgboost.py
# Author: David Riser
# Date: July 19, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, xgboost is trained.

import numpy as np
import pandas as pd
import utils

from xgboost import XGBClassifier

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

def train():

    path_to_data   = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    path_to_preds  = '../../data/predictions/'

    version = '1.1'
    random_seed = 8675309
    sample_size = 50000
    n_folds = 5

    xgb_params = {
        'learning_rate':0.1,
        'n_estimators':10000,
        'max_depth':4,
        'min_child_weight':5,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'objective':'binary:logistic',
        'nthread':4,
        'seed':random_seed,
        'scale_pos_weight':2.5,
        'reg_alpha':1.2,
        'early_stopping_rounds':50,
        'verbose':20,
        'eval_metric':'auc'
    }

    train, labels, test, train_ids, test_ids = utils.load_features(path_to_data, version, sample_size)
    oof_train, oof_test = utils.kfold(classifier_builder=XgboostWrapper,
                                      base_classifier=XGBClassifier,
                                      classifier_params=xgb_params,
                                      train=train,
                                      labels=labels,
                                      test=test,
                                      n_folds=n_folds,
                                      random_seed=random_seed)

    df_oof_train = pd.DataFrame({'SK_ID_CURR':train_ids, 'TARGET':labels, 'xgboost':oof_train})
    df_oof_train['SK_ID_CURR'] = df_oof_train['SK_ID_CURR'].astype('int32')

    df_oof_test = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':oof_test})
    df_oof_test['SK_ID_CURR'] = df_oof_test['SK_ID_CURR'].astype('int32')

    df_oof_train.to_csv(path_to_preds + version + '-xgboost.csv', index=False)
    df_oof_test.to_csv(path_to_output + version + '-xgboost.csv', index=False)

if __name__ == '__main__':
    train()