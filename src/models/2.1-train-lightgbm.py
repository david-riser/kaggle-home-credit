#
# 2.1-train-lightgbm.py
# Author: David Riser
# Date: July 21, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, lightgbm is trained.  In this model, SMOTE is used to
# upsample the minority class. 


import lightgbm
import pandas as pd
import utils

class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        self.clf = clf(**params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.clf.fit(x_train, y_train,
                     eval_set=[(x_train, y_train), (x_valid, y_valid)],
                     eval_metric='auc',
                     verbose=50,
                     early_stopping_rounds=200)

    def predict(self, x):
        return self.clf.predict_proba(x, num_iteration=self.clf.best_iteration_)[:,1]

def train():

    path_to_data   = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    path_to_preds  = '../../data/predictions/'

    version = '1.1'
    random_seed = 8675309
    sample_size = 50000
    n_folds = 5

    params = {
        'nthread':8,
        'n_estimators':10000,
        'learning_rate':0.02,
        'num_leaves':34,
        'colsample_bytree':0.9497036,
        'subsample':0.8715623,
        'max_depth':8,
        'reg_alpha':0.041545473,
        'reg_lambda':0.0735294,
        'min_split_gain':0.0222415,
        'min_child_weight':39.3259775,
        'silent':-1,
        'verbose':-1
    }

    train, labels, test, train_ids, test_ids = utils.load_features(path_to_data, version, sample_size)
    oof_train, oof_test = utils.kfold(classifier_builder=LightGBMWrapper,
                                      base_classifier=lightgbm.LGBMClassifier,
                                      classifier_params=params,
                                      train=train,
                                      labels=labels,
                                      test=test,
                                      n_folds=n_folds,
                                      random_seed=random_seed,
                                      use_smote=True)

    df_oof_train = pd.DataFrame({'SK_ID_CURR':train_ids, 'TARGET':labels, 'lightgbm':oof_train})
    #    df_oof_train['SK_ID_CURR'] = df_oof_train['SK_ID_CURR'].astype('int32')

    df_oof_test = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':oof_test})
    #    df_oof_test['SK_ID_CURR'] = df_oof_test['SK_ID_CURR'].astype('int32')

    df_oof_train.to_csv(path_to_preds + version + '-lightgbm.csv', index=False)
    df_oof_test.to_csv(path_to_output + version + '-lightgbm.csv', index=False)

if __name__ == '__main__':
    train()
