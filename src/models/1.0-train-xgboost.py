#
# 1.0-train-xgboost.py
# Author: David Riser
# Date: July 16, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, xgboost is trained.

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import KFold

def main():

    # Define constants and configurations.
    path_to_data   = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    sample_size    = None
    SEED           = 8675309

    # Load both training and testing
    train = pd.read_csv(path_to_data + '1.1-features-train.csv', nrows=sample_size, compression='gzip')
    test = pd.read_csv(path_to_data + '1.1-features-test.csv', nrows=sample_size, compression='gzip')

    # Drop
    labels = train.TARGET

    # Save for predictions
    test_ids = test.SK_ID_CURR

    # The way I constructed the testing set in the
    # feature building notebook leaves it with an empty TARGET column.
    train.drop(columns=['test', 'SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    test.drop(columns=['test', 'SK_ID_CURR', 'TARGET'], axis=1, inplace=True)

    oof_preds = np.zeros(len(train))
    sub_preds = np.zeros(len(test))

    # Setup kfolds and train.
    n_folds = 5
    kf = KFold(n_splits=n_folds, random_state=SEED)
    for train_index, val_index in kf.split(train):
        x_train, y_train = train.iloc[train_index].values, labels[train_index]
        x_valid, y_valid = train.iloc[val_index].values, labels[val_index]

        #        x_train.columns = train.columns
        #        x_valid.columns = train.columns

        clf = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=10000,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            nthread=4,
            seed=SEED,
            scale_pos_weight=2.5,
            reg_alpha=1.2
        )

        clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='auc',
                verbose=20, early_stopping_rounds=50)

        oof_preds[val_index] = clf.predict_proba(x_valid, ntree_limit=clf.best_ntree_limit)[:,1]
        sub_preds += clf.predict_proba(test.values, ntree_limit=clf.best_ntree_limit)[:,1] / n_folds

    # Predict the test dataset 
    print('Total ROC AUC = %.4f' % roc_auc_score(labels, oof_preds))

    submission = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':sub_preds})
    submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
    submission.to_csv(path_to_output+'1.1-xgboost.csv', index=False)

if __name__ == '__main__':
    main()
