#
# 2.0-train-sklearn.py
# Author: David Riser
# Date: July 20, 2018
#
# Using the output (in data/processed/) from the feature building scripts
# in src/features, sklearn models are trained.


import pandas as pd
import utils

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        # params['random_state'] = seed
        self.clf = clf(**params)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

def train():

    path_to_data   = '../../data/processed/'
    path_to_output = '../../data/submissions/'
    path_to_preds  = '../../data/predictions/'

    version = '1.1'
    random_seed = 8675309
    sample_size = None
    n_folds = 5

    rf_params = {
        'n_jobs':-1,
        'n_estimators':10
    }

    et_params  = {}
    nb_params = {}

    train, labels, test, train_ids, test_ids = utils.load_features(path_to_data, version, sample_size)

    # Handle NaN values.
    # This converts pandas.DataFrame to numpy.ndarray.
    imp = Imputer()
    train = imp.fit_transform(train)
    test = imp.transform(test)

    # Cast to pandas.dataframe for kfold method.
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    # Check format 
    print('Summary of training nulls: ', train.isnull().sum().sum())
    print('Summary of testing nulls: ', test.isnull().sum().sum())

    # ------------------------------------------------------------------------
    #    Start training models. 
    # ------------------------------------------------------------------------

    # Start with RandomForest
    oof_train, oof_test = utils.kfold(classifier_builder=SklearnWrapper,
                                      base_classifier=RandomForestClassifier,
                                      classifier_params=rf_params,
                                      train=train,
                                      labels=labels,
                                      test=test,
                                      n_folds=n_folds,
                                      random_seed=random_seed)

    df_oof_train = pd.DataFrame({'SK_ID_CURR':train_ids, 'TARGET':labels, 'random-forest':oof_train})
    df_oof_train['SK_ID_CURR'] = df_oof_train['SK_ID_CURR'].astype('int32')

    df_oof_test = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':oof_test})
    df_oof_test['SK_ID_CURR'] = df_oof_test['SK_ID_CURR'].astype('int32')

    df_oof_train.to_csv(path_to_preds + version + '-random-forest.csv', index=False)
    df_oof_test.to_csv(path_to_output + version + '-random-forest.csv', index=False)
    del oof_test, oof_train, df_oof_test, df_oof_train

    # Extra trees
    oof_train, oof_test = utils.kfold(classifier_builder=SklearnWrapper,
                                      base_classifier=ExtraTreesClassifier,
                                      classifier_params=et_params,
                                      train=train,
                                      labels=labels,
                                      test=test,
                                      n_folds=n_folds,
                                      random_seed=random_seed)

    df_oof_train = pd.DataFrame({'SK_ID_CURR': train_ids, 'TARGET': labels, 'extra-trees': oof_train})
    df_oof_train['SK_ID_CURR'] = df_oof_train['SK_ID_CURR'].astype('int32')

    df_oof_test = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': oof_test})
    df_oof_test['SK_ID_CURR'] = df_oof_test['SK_ID_CURR'].astype('int32')

    df_oof_train.to_csv(path_to_preds + version + '-extra-trees.csv', index=False)
    df_oof_test.to_csv(path_to_output + version + '-extra-trees.csv', index=False)
    del oof_test, oof_train, df_oof_test, df_oof_train

    # Naive Bayes
    oof_train, oof_test = utils.kfold(classifier_builder=SklearnWrapper,
                                      base_classifier=GaussianNB,
                                      classifier_params=nb_params,
                                      train=train,
                                      labels=labels,
                                      test=test,
                                      n_folds=n_folds,
                                      random_seed=random_seed)

    df_oof_train = pd.DataFrame({'SK_ID_CURR':train_ids, 'TARGET':labels, 'naive-bayes':oof_train})
    df_oof_train['SK_ID_CURR'] = df_oof_train['SK_ID_CURR'].astype('int32')

    df_oof_test = pd.DataFrame({'SK_ID_CURR':test_ids, 'TARGET':oof_test})
    df_oof_test['SK_ID_CURR'] = df_oof_test['SK_ID_CURR'].astype('int32')

    df_oof_train.to_csv(path_to_preds + version + '-naive-bayes.csv', index=False)
    df_oof_test.to_csv(path_to_output + version + '-naive-bayes.csv', index=False)


if __name__ == '__main__':
    train()
