#
# 1.0-train-lightgbm.py
# Author: David Riser
# Date: July 2, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, LightGBM is trained.

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

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

def train():

    # Define constants and configurations.
    path_to_data   = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    path_to_preds  = '../../data/predictions/'
    version        = '1.1'
    sample_size    = None
    SEED           = 8675309

    # Setup the feature datasets and prediction containers
    train, labels, test, train_ids, test_ids = load_features(path_to_data, version, sample_size)
    oof_preds = np.zeros(len(train))
    sub_preds = np.zeros(len(test))

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, val_index in kf.split(train):
        x_train, y_train = train.iloc[train_index].values, labels[train_index]
        x_valid, y_valid = train.iloc[val_index].values, labels[val_index]

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

    submission = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':sub_preds})
    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
    submission.to_csv(path_to_output + version + '-lightgbm.csv', index=False)

    # Save out of fold predictions for model stacking.
    oof = pd.DataFrame({'SK_ID_CURR':train_ids, 'lightgbm':oof_preds, 'TARGET':labels})
    oof.to_csv(path_to_preds + version + '-lightgbm.csv', index=False)


if __name__ == '__main__':
    train()
