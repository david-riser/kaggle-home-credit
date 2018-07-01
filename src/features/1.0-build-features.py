#
# 1.0-build-features.py
# Author: David Riser
# Date: Jun. 27, 2018
#
# Template for aggregating data from all tables provided
# as part of this kaggle challenge, and saving the processed
# data into data/processed/.
#

import logging
import numpy as np
import pandas as pd
import time

from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder

# Taken from https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
# and modified to log instead of print. 

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    log.info("{} - done in {:.0f}s".format(title, time.time() - t0))

def rename_column(table_name, variable, aggregation):
    '''
    Naming scheme for the columns is all caps and
    has the format. 

    (DATA_SOURCE)_(VARIABLE)_(AGGREGATION_TYPE)
    '''
    return table_name + '_' + variable + '_' + aggregation.upper()

def encode_categoricals(data):
    '''
    Take a dataframe and label encode all categorical 
    variables inplace. 
    '''
    cat_cols = data.select_dtypes('object').columns
    for cat in cat_cols:
        encoder = LabelEncoder()
        data[cat] = encoder.fit_transform(data[cat])

def process_application(path_to_data='', sample_size=1000):
    '''
    Load and process the main dataset.  Merge the test and
    training samples together using the dummy variable test.
    '''

    # Read application data, this is the main dataset.
    app_train = pd.read_csv(path_to_data + 'application_train.csv', nrows=sample_size)
    app_test = pd.read_csv(path_to_data + 'application_test.csv', nrows=sample_size)
    log.info('Read application testing table with shape %s', app_train.shape)

    # Concatenate training and testing data, after providing a dummy label.
    app_train['test'] = np.repeat(0, len(app_train))
    app_test['test'] = np.repeat(1, len(app_test))
    app = pd.concat(list([app_train, app_test]), axis=0)
    log.info('Concatenated training/testing data with shape %s', app.shape)

    # Perform label encoding on categorical variables.
    encode_categoricals(app)

    return app

def process_installment(path_to_data='', sample_size=1000):
    '''
    Process installments dataset.
    '''

    # Read supplementary tables
    install_data = pd.read_csv(path_to_data + 'installments_payments.csv', nrows=sample_size)
    log.info('Loaded installment data with shape %s', install_data.shape)

    # Add features before aggregation
    install_data['PAYMENT_DIFF'] = install_data.AMT_PAYMENT - install_data.AMT_INSTALMENT
    install_data['PAYMENT_PERCENT'] = install_data.AMT_PAYMENT / install_data.AMT_INSTALMENT
    install_data['DAYS_BEFORE_DUE'] = install_data.DAYS_INSTALMENT - install_data.DAYS_ENTRY_PAYMENT
    install_data['LATE_PAYMENT'] = np.zeros(len(install_data))
    install_data['LATE_PAYMENT'].loc[install_data.DAYS_BEFORE_DUE < 0] = 1

    # Simple aggregations for all columns.
    install_aggregated = install_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    # Fix names of columns
    install_aggregated.columns = [rename_column('INSTALL', col[0], col[1])
                                  for col in list(install_aggregated.columns)]

    # Add the number of payments for this customer.
    install_aggregated['INSTALL_COUNT'] = install_data.groupby('SK_ID_CURR').size()

    log.debug('Aggregated installment dataframe has columns %s', install_aggregated.columns)

    return install_aggregated

def process_creditcard(path_to_data='', sample_size=1000):
    '''
    Process credit card dataset.
    '''

    # Load and encode.
    credit_card_data = pd.read_csv(path_to_data + 'credit_card_balance.csv', nrows=sample_size)
    encode_categoricals(credit_card_data)

    # Add features before aggregation.
    credit_card_data['BALANCE_PERCENT'] = credit_card_data.AMT_BALANCE / credit_card_data.AMT_CREDIT_LIMIT_ACTUAL
    credit_card_data['PAYMENT_MIN_PAYMENT_DIFF'] = credit_card_data.AMT_PAYMENT_TOTAL_CURRENT - credit_card_data.AMT_INST_MIN_REGULARITY
    credit_card_data['OVERPAYMENT_LIMIT_PERCENT'] = credit_card_data.PAYMENT_MIN_PAYMENT_DIFF / credit_card_data.AMT_CREDIT_LIMIT_ACTUAL

    # Perform basic aggregations.
    credit_card_aggregated = credit_card_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    # Fix names of columns
    credit_card_aggregated.columns = [rename_column('CREDIT_CARD', col[0], col[1])
                                  for col in list(credit_card_aggregated.columns)]

    # Add the number of payments for this customer.
    credit_card_aggregated['CREDIT_CARD_COUNT'] = credit_card_data.groupby('SK_ID_CURR').size()
    log.debug('Aggregated credit card dataframe has columns %s', credit_card_aggregated.columns)

    return credit_card_aggregated

def process_bureau(path_to_data='', sample_size=1000):
    '''
    Process both bureau and bureau balance datasets.
    '''

    # Load data and encode.
    bureau_data = pd.read_csv(path_to_data + 'bureau.csv', nrows=sample_size)
    bureau_balance_data = pd.read_csv(path_to_data + 'bureau_balance.csv', nrows=sample_size)
    encode_categoricals(bureau_data)
    encode_categoricals(bureau_balance_data)

    # Add features before aggregation.


    # Simple aggregations
    bureau_aggregated = bureau_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    bureau_aggregated.columns = [rename_column('BUREAU', col[0], col[1])
                                 for col in list(bureau_aggregated.columns)]

    bureau_balance_aggregated = bureau_balance_data.groupby('SK_ID_BUREAU').aggregate(
        ['min', 'max', 'mean', 'var','nunique']
    )
    bureau_balance_aggregated.columns = [rename_column('BUREAU_BALANCE', col[0], col[1])
                                  for col in list(bureau_balance_aggregated.columns)]

    # Join tables
    data = bureau_aggregated.join(bureau_balance_aggregated, how='left')

    return data

def process_cash(path_to_data='', sample_size=1000):
    '''
    Processing cash dataset and making aggregations.
    '''

    cash_data = pd.read_csv(path_to_data + 'POS_CASH_balance.csv', nrows=sample_size)
    encode_categoricals(cash_data)

    # Add features here, then aggregate.
    cash_aggregated = cash_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    # Fix naming
    cash_aggregated.columns = [rename_column('CASH', col[0], col[1])
                                 for col in list(cash_aggregated.columns)]
    log.debug('Aggregated cash dataframe has columns %s', cash_aggregated.columns)
    return cash_aggregated

def process_previous(path_to_data='', sample_size=1000):
    '''
    Processing cash dataset and making aggregations.
    '''

    previous_data = pd.read_csv(path_to_data + 'previous_application.csv', nrows=sample_size)
    encode_categoricals(previous_data)

    # Add features here.

    # Perform aggregations
    previous_aggregated = previous_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    # Fix naming
    previous_aggregated.columns = [rename_column('PREV', col[0], col[1])
                                 for col in list(previous_aggregated.columns)]
    log.debug('Aggregated previous dataframe has columns %s', previous_aggregated.columns)
    return previous_aggregated


###################################################
def build_features():

    # Constants for loading of data.
    # Placing None as the sample size
    # will run the complete dataset.
    sample_size    = None
    path_to_data   = '../../data/raw/'
    path_to_output = '../../data/processed/'

    with timer('Processing application testing/training'):
        app = process_application(path_to_data, sample_size)

    with timer('Processing installment dataset'):
        installment = process_installment(path_to_data, sample_size)

    with timer('Processing credit card dataset'):
        credit_card = process_creditcard(path_to_data, sample_size)

    with timer('Processing bureau datasets'):
        bureau = process_bureau(path_to_data, sample_size)

    with timer('Processing cash dataset'):
        cash = process_cash(path_to_data, sample_size)

    with timer('Processing previous dataset'):
        previous = process_previous(path_to_data, sample_size)

    # Create merged table
    dataset = app.join(installment, how='left', on='SK_ID_CURR')
    log.info('Added installment data with shape %s', dataset.shape)

    dataset = dataset.join(credit_card, how='left', on='SK_ID_CURR')
    log.info('Added credit card data with shape %s', dataset.shape)

    dataset = dataset.join(bureau, how='left', on='SK_ID_CURR')
    log.info('Added bureau data with shape %s', dataset.shape)

    dataset = dataset.join(cash, how='left', on='SK_ID_CURR')
    log.info('Added cash data with shape %s', dataset.shape)

    dataset = dataset.join(previous,how='left', on='SK_ID_CURR')
    log.info('Added previous data with shape %s', dataset.shape)

    # Save compressed version of aggregated dataset.
    with timer('Finishing and writing to file'):
        dataset.to_csv(path_to_output + '1.0-features.csv', compression='gzip')

if __name__ == '__main__':
    build_features()