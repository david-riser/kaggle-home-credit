#
# 1.0-train-naive-bayes.py
# Author: David Riser
# Date: July 10, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, a naive-bayes is trained.

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer


def main():

    # Define constants and configurations.
    path_to_data   = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    sample_size    = None
    SEED           = 8675309

    # Load both training and testing
    train = pd.read_csv(path_to_data + '1.2-features-train.csv', nrows=sample_size, compression='gzip')
    test = pd.read_csv(path_to_data + '1.2-features-test.csv', nrows=sample_size, compression='gzip')
    labels = train['TARGET'].values 

    # Save for predictions
    test_ids = test['SK_ID_CURR'].values
    train.drop(columns=['test', 'SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    test.drop(columns=['test', 'SK_ID_CURR'], axis=1, inplace=True)

    imp   = Imputer()
    train_imp = imp.fit_transform(train)
    test_imp  = imp.transform(test)

    train_df = pd.DataFrame(train_imp)
    test_df = pd.DataFrame(test_imp)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, val_index in kf.split(train_df):
        x_train, y_train = train_df.iloc[train_index], labels[train_index]
        x_valid, y_valid = train_df.iloc[val_index], labels[val_index]
        #        x_train, y_train = train.iloc[train_index].values, labels[train_index]
        #        x_valid, y_valid = train.iloc[val_index].values, labels[val_index]

        # Define model
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)

        y_pred = gnb.predict_proba(x_valid)[:,1]
        val_score = roc_auc_score(y_valid, y_pred)
        print('Validation AUC = %.4f' % val_score)

        oof_preds[val_index] = y_pred
        preds = gnb.predict_proba(test_df)[:,1] / n_folds
        print('Shape of model prediction: ', preds.shape)

        sub_preds += preds

    # Predict the test dataset 
    print('Total ROC AUC = %.4f' % roc_auc_score(labels, oof_preds))

    submission = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':sub_preds})
    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
    submission.to_csv(path_to_output+'1.2-naive-bayes.csv', index=False)

if __name__ == '__main__':
    main()
