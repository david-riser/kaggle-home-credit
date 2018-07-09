#
# 1.0-train-random-forest.py
# Author: David Riser
# Date: July 2, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, a random forest is trained.

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer


def main():

    # Define constants and configurations.
    path_to_data = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    sample_size  = None
    SEED         = 8675309

    # Load both training and testing
    train = pd.read_csv(path_to_data + '1.1-features-train.csv', nrows=sample_size, compression='gzip')
    test = pd.read_csv(path_to_data + '1.1-features-test.csv', nrows=sample_size, compression='gzip')

    # Drop 
    y_train = train.TARGET 
    y_test = test.TARGET 

    # Save for predictions
    test_ids = test.SK_ID_CURR
    train.drop(columns=['test', 'SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    test.drop(columns=['test', 'SK_ID_CURR'], axis=1, inplace=True)
    
    # Impute missing values.  This 
    # will transform our dataframe
    # to numpy.ndarray. 
    imp   = Imputer()
    train = imp.fit_transform(train)
    test  = imp.fit_transform(test)

    oof_preds = np.zeros(len(train))
    sub_preds = np.zeros(len(test))

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, val_index in kf.split(train):
        rf = RandomForestClassifier()
        rf.fit(train[train_index], y_train[train_index])

        y_pred = rf.predict_proba(train[val_index])[:,1]
        val_score = roc_auc_score(y_train[val_index], y_pred)
        print('Validation AUC = %.4f' % val_score)

        oof_preds[val_index] = y_pred
        sub_preds += rf.predict_proba(test)[:,1] / n_folds

    # Predict the test dataset 
    print('Total ROC AUC = %.4f' % roc_auc_score(y_train, oof_preds))

    submission = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':sub_preds})
    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
    submission.to_csv(path_to_output+'1.0-random-forest.csv', index=False)

if __name__ == '__main__':
    main()
