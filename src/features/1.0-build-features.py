#
# 1.0-build-features.py
# Author: David Riser
# Data: Jun. 27, 2018
#
# Template for aggregating data from all tables provided
# as part of this kaggle challenge, and saving the processed
# data into data/processed/.
#

import logging
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import LabelEncoder

def rename_column(table_name, variable, aggregation):
    '''

    :param table_name: Name of table passed
    :param variable: Name of variable being aggregated
    :param aggregation: Name of aggregation method
    :return: string with new column name
    '''
    return table_name + '_' + variable + '_' + aggregation.upper()

def encode_categoricals(data):
    '''

    :param data: A pandas dataframe that contains categorical variables.
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

    start_time = time.time()

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

    runtime = time.time() - start_time
    log.info('Application data processed in %s seconds.', runtime)
    return app

def process_installment(path_to_data='', sample_size=1000):
    '''
    Process installments dataset.
    '''

    start_time = time.time()

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

    runtime = time.time() - start_time
    log.info('Application data processed in %s seconds.', runtime)
    return install_aggregated

def process_creditcard(path_to_data='', sample_size=1000):
    '''
    Process credit card dataset.
    '''

    start_time = time.time()

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

    runtime = time.time() - start_time
    log.info('Credit card data processed in %s seconds.', runtime)
    return credit_card_aggregated

def process_bureau(path_to_data='', sample_size=1000):
    '''
    Process both bureau and bureau balance datasets.
    '''

    start_time = time.time()

    # Load data and encode.
    bureau_data = pd.reac_csv(path_to_data + 'bureau.csv', nrows=sample_size)
    bureau_balance_data = pd.reac_csv(path_to_data + 'bureau_balance.csv', nrows=sample_size)
    encode_categoricals(bureau_data)
    encode_categoricals(bureau_balance_data)

    # Add features before aggregation.


    # Simple aggregations
    bureau_aggregated = bureau_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    bureau_aggregated.columns = [rename_column('BUREAU', col[0], col[1])
                                  for col in list(bureau_aggregated.columns)]

    bureau_balance_aggregated = bureau_balance_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var','nunique']
    )
    bureau_balance_aggregated.columns = [rename_column('BUREAU_BALANCE', col[0], col[1])
                                  for col in list(bureau_balance_aggregated.columns)]

    # Join tables
    data = bureau_aggregated.join(bureau_balance_aggregated, how='left', on='SK_ID_CURR')

    runtime = time.time() - start_time
    log.info('Bureau data processed in %s seconds.', runtime)
    return data

###################################################
###################################################
###################################################

# Setup logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)

# Constants for loading of data.
sample_size = 5000
path_to_data = '../../data/raw/'
path_to_output = '../../data/processed/'

app = process_application(path_to_data, sample_size)
installment = process_installment(path_to_data, sample_size)
credit_card = process_creditcard(path_to_data, sample_size)
bureau = process_creditcard(path_to_data, sample_size)

# Create merged table
dataset = app.join(installment, how='left', on='SK_ID_CURR')
log.info('Added installment data with shape %s', dataset.shape)

dataset = dataset.join(credit_card, how='left', on='SK_ID_CURR')
log.info('Added credit card data with shape %s', dataset.shape)

dataset = dataset.join(bureau, how='left', on='SK_ID_CURR')
log.info('Added bureau data with shape %s', dataset.shape)

# Save compressed version of aggregated dataset.
dataset.to_csv(path_to_output + '1.0-features.csv', compression='gzip')
