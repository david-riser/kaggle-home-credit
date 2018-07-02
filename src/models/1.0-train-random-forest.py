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
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer

def main():

    # Define constants and configurations.
    path_to_data = '../../data/processed/'
    sample_size  = 10000
    SEED         = 8675309

    # Load both training and testing
    data = pd.read_csv(path_to_data + '1.0-features.csv', nrows=sample_size, compression='gzip')

    # Split into training and testing
    # and remove data.
    train = data.loc[data.test == 0]
#    test  = data.loc[data.test == 1]
    del data

    
    # Setup columns for training
    features = list(train.columns)
    features.remove('test')
    features.remove('SK_ID_CURR')
    features.remove('TARGET')

    # Impute missing values
    imp   = Imputer()
    train = imp.fit_transform(train)
#    test  = imp.fit_transform(test)

    # Setup kfolds and train.
    kf = KFold(n_splits=5, random_state=SEED)
    for train_index, val_index in kf.split(train):
        print('Training...')
        rf = RandomForestClassifier()
        rf.fit(train[features].iloc[train_index], train.TARGET.iloc[train_index])



if __name__ == '__main__':
    main()
