# 
# File: utils.py 
# 
# Author: David Riser 
# Date:   July 19, 2018 
#
# Utilities for loading data, removing nan 
# values for some models, and training are 
# provided for the training scripts. 

import numpy as np 
import pandas as pd 
import time

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE 

def load_features(path_to_data, version, sample_size=10000):

    # Load both training and testing
    train = pd.read_csv(path_to_data + version +'-features-train.csv', nrows=sample_size, compression='gzip')
    test = pd.read_csv(path_to_data + version + '-features-test.csv', nrows=sample_size, compression='gzip')
    print('Loaded training shape: ', train.shape)
    print('Loaded test shape: ', test.shape)
    
    #    train.dropna(subset=['SK_ID_CURR'], axis=0, inplace=True) 
    #    print('Dropped nan training shape: ', train.shape)

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


def kfold(classifier_builder, classifier_params, base_classifier,
          train, labels, test, n_folds=5, random_seed=0, use_smote=False):

    # Replace nan
    if use_smote:
        train.fillna(0, inplace=True)
        test.fillna(0, inplace=True)

    # Testing and training out of fold
    # predictions.
    oof_test  = np.zeros(len(test))
    oof_train = np.zeros(len(train))

    # Setup kfolds and train.
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    for train_index, val_index in kf.split(train):
        start_time = time.time()

        x_train, y_train = train.iloc[train_index].values, labels[train_index]
        x_valid, y_valid = train.iloc[val_index].values, labels[val_index]

        if use_smote:
            sm = SMOTE(k_neighbors=5)
            x_train, y_train = sm.fit_sample(x_train, y_train)
        
        # This is just a wrapper that has fit and predict
        # functionality.
        clf = classifier_builder(base_classifier,
                                 seed=random_seed,
                                 params=classifier_params)
        clf.fit(x_train, y_train, x_valid, y_valid)

        # Out of fold predictions for training and testing sets.
        oof_train[val_index] = clf.predict(x_valid)
        oof_test += clf.predict(test.values) / n_folds

        elapsed_time = time.time() - start_time
        print('Finished fold in %s seconds.' % elapsed_time)

    # Predict the test dataset
    print('Total ROC AUC = %.4f' % roc_auc_score(labels, oof_train))
    return oof_train, oof_test
