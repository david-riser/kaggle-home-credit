#
# 1.0-stack-lightgbm.py
# Author: David Riser
# Date: July 17, 2018
#
# Using the predictions from the train models
# that are stored in data/predictions lightgbm
# is trained as a stacker.

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def stack():

    # Define constants and configurations.
    path_to_data = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    path_to_preds = '../../data/predictions/'
    version = '1.1'
    sample_size = 10000
    SEED = 8675309

    # Load inputs
    xgb  = pd.read_csv(path_to_preds + version + '-xgboost.csv')
    lgbm = pd.read_csv(path_to_preds + version + '-lightgbm.csv')

    # Predictions should be the same shape.
    assert(xgb.shape == lgbm.shape)

    # Aggregate the training data
    train = xgb.merge(lgbm[['SK_ID_CURR', 'LGBM']], on='SK_ID_CURR')
    train['SK_ID_CURR'] = train['SK_ID_CURR'].astype('int32')

    # Load and combine testing data
    xgb_test = pd.read_csv(path_to_output + version + '-xgboost.csv')
    lgbm_test = pd.read_csv(path_to_output + version + '-lightgbm.csv')
    test = xgb_test.merge(lgbm_test, on='SK_ID_CURR')
    test['SK_ID_CURR'] = test['SK_ID_CURR'].astype('int32')


    # Setup empties for preds.
    oof_preds = np.zeros(len(train))
    sub_preds = np.zeros(len(train))

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, val_index in kf.split(train):
        x_train, y_train = train.iloc[train_index][['XGB', 'LGBM']].values, train.iloc[train_index]['TARGET'].values
        x_valid, y_valid = train.iloc[val_index][['XGB', 'LGBM']].values, train.iloc[val_index]['TARGET'].values

        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1
            )

        clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='auc',
                verbose=100, early_stopping_rounds=200)

        oof_preds[val_index] = clf.predict_proba(x_valid, num_iteration=clf.best_iteration_)[:,1]
        sub_preds += clf.predict_proba(test, num_iteration=clf.best_iteration_)[:,1] / n_folds

    # Predict the test dataset
    print('Total ROC AUC = %.4f' % roc_auc_score(labels, oof_preds))

    submission = pd.DataFrame({'SK_ID_CURR':ids, 'TARGET':sub_preds})
    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
    submission.to_csv(path_to_output+'1.1-stack.csv', index=False)


if __name__ == '__main__':
    stack()
