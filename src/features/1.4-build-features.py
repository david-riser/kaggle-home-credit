#
# 1.4-build-features.py
# Author: David Riser
# Date: Aug. 11, 2018
#
# Template for aggregating data from all tables provided
# as part of this kaggle challenge, and saving the processed
# data into data/processed/.
#
# In this version I am returning to label
# encoding, and applying partial aggs.

import logging
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__file__)

# Taken from https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
# and modified to log instead of print.
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

def aggregate_over_k(data, agg_func, k):
    ''' We're going to take the data and apply an aggregation 
    to the just the last k installments. 
    '''
    return agg_func(data[:k])

def min_max_diff(data):
    return (data.max()-data.min())

# This is a block of aggregation functions.  Perhaps there
# is a better way to do this.  Creating lambdas doesn't work
# because the name is the same.
def mean_over_6(data):
    return aggregate_over_k(data, np.mean, 6)

def mean_over_12(data):
    return aggregate_over_k(data, np.mean, 12)

def mean_over_18(data):
    return aggregate_over_k(data, np.mean, 18)

def mean_over_24(data):
    return aggregate_over_k(data, np.mean, 24)

def std_over_6(data):
    return aggregate_over_k(data, np.std, 6)

def std_over_12(data):
    return aggregate_over_k(data, np.std, 12)

def std_over_18(data):
    return aggregate_over_k(data, np.std, 18)

def std_over_24(data):
    return aggregate_over_k(data, np.std, 24)

def mmd_over_6(data):
    return aggregate_over_k(data, min_max_diff, 6)

def mmd_over_12(data):
    return aggregate_over_k(data, min_max_diff, 12)

def mmd_over_18(data):
    return aggregate_over_k(data, min_max_diff, 18)

def mmd_over_24(data):
    return aggregate_over_k(data, min_max_diff, 24)

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
    app_test['test']  = np.repeat(1, len(app_test))
    app = pd.concat(list([app_train, app_test]), axis=0)
    log.info('Concatenated training/testing data with shape %s', app.shape)

    # Basic cleaning to change place holders to nan.
    app['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
    app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    app['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    app['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    app['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

    # Perform label encoding on categorical variables.
    encode_categoricals(app)

    # Add features 
    app['DAYS_EMPLOYED_PERCENT'] = app.DAYS_EMPLOYED / app.DAYS_BIRTH
    app['CREDIT_TERM'] = app.AMT_ANNUITY / app.AMT_CREDIT
    app['CREDIT_INCOME_PERCENT'] = app.AMT_CREDIT / app.AMT_INCOME_TOTAL
    app['ANNUITY_INCOME_PERCENT'] = app.AMT_ANNUITY / app.AMT_INCOME_TOTAL
    app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    app['EXT_SOURCE_STD'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    app['CREDIT_GOODS_RATIO'] = app.AMT_CREDIT / app.AMT_GOODS_PRICE

    # Additional features from neptune-ml open solution.
    app['car_to_birth_ratio'] = app['OWN_CAR_AGE'] / app['DAYS_BIRTH']
    app['car_to_employ_ratio'] = app['OWN_CAR_AGE'] / app['DAYS_EMPLOYED']
    app['children_ratio'] = app['CNT_CHILDREN'] / app['CNT_FAM_MEMBERS']
    app['credit_to_annuity_ratio'] = app['AMT_CREDIT'] / app['AMT_ANNUITY']
    app['credit_to_goods_ratio'] = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']
    app['income_per_child'] = app['AMT_INCOME_TOTAL'] / (1 + app['CNT_CHILDREN'])
    app['income_per_person'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']
    app['phone_to_birth_ratio'] = app['DAYS_LAST_PHONE_CHANGE'] / app['DAYS_BIRTH']
    app['phone_to_employ_ratio'] = app['DAYS_LAST_PHONE_CHANGE'] / app['DAYS_EMPLOYED']
    app['external_sources_weighted'] = app.EXT_SOURCE_1 * 2 + app.EXT_SOURCE_2 * 3 + app.EXT_SOURCE_3 * 4
    app['cnt_non_child'] = app['CNT_FAM_MEMBERS'] - app['CNT_CHILDREN']
    app['child_to_non_child_ratio'] = app['CNT_CHILDREN'] / app['cnt_non_child']
    app['income_per_non_child'] = app['AMT_INCOME_TOTAL'] / app['cnt_non_child']
    app['credit_per_person'] = app['AMT_CREDIT'] / app['CNT_FAM_MEMBERS']
    app['credit_per_child'] = app['AMT_CREDIT'] / (1 + app['CNT_CHILDREN'])
    app['credit_per_non_child'] = app['AMT_CREDIT'] / app['cnt_non_child']
    app['short_employment'] = (app['DAYS_EMPLOYED'] < -2000).astype(int)
    app['young_age'] = (app['DAYS_BIRTH'] < -14000).astype(int)

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
    credit_card_data['AMT_DRAWINGS_ATM_CURRENT'][credit_card_data['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    credit_card_data['AMT_DRAWINGS_CURRENT'][credit_card_data['AMT_DRAWINGS_CURRENT'] < 0] = np.nan
    encode_categoricals(credit_card_data)

    # Add features before aggregation.
    credit_card_data['BALANCE_PERCENT'] = credit_card_data.AMT_BALANCE / credit_card_data.AMT_CREDIT_LIMIT_ACTUAL
    credit_card_data['PAYMENT_MIN_PAYMENT_DIFF'] = credit_card_data.AMT_PAYMENT_TOTAL_CURRENT - credit_card_data.AMT_INST_MIN_REGULARITY
    credit_card_data['OVERPAYMENT_LIMIT_PERCENT'] = credit_card_data.PAYMENT_MIN_PAYMENT_DIFF / credit_card_data.AMT_CREDIT_LIMIT_ACTUAL
    credit_card_data['AMT_ATM_RATIO'] = credit_card_data.AMT_DRAWINGS_ATM_CURRENT / credit_card_data.AMT_DRAWINGS_CURRENT
    credit_card_data['AMT_POS_RATIO'] = credit_card_data.AMT_DRAWINGS_POS_CURRENT / credit_card_data.AMT_DRAWINGS_CURRENT
    credit_card_data['AMT_OTHER_RATIO'] = credit_card_data.AMT_DRAWINGS_OTHER_CURRENT / credit_card_data.AMT_DRAWINGS_CURRENT

    # Perform basic aggregations.
    credit_card_aggregated = credit_card_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    # Fix names of columns
    credit_card_aggregated.columns = [rename_column('CREDIT_CARD', col[0], col[1])
                                  for col in list(credit_card_aggregated.columns)]

    # Add the number of payments for this customer.
    credit_card_aggregated['CREDIT_CARD_COUNT'] = credit_card_data.groupby('SK_ID_CURR').size()

    # Perform partial aggregations 
    partial_aggregation_functions = [mean_over_6, mean_over_12, mean_over_18, mean_over_24,
                                     std_over_6, std_over_12, std_over_18, std_over_24,
                                     mmd_over_6, mmd_over_12, mmd_over_18, mmd_over_24
    ]
    partial_aggregations = {
        'AMT_BALANCE':partial_aggregation_functions,
        'BALANCE_PERCENT':partial_aggregation_functions,
        'SK_DPD':partial_aggregation_functions
    }
    credit_card_partially_aggregated = credit_card_data.sort_values(
        ['SK_ID_CURR', 'MONTHS_BALANCE']).groupby('SK_ID_CURR').agg(partial_aggregations)
    credit_card_partially_aggregated.columns = [rename_column('CREDIT_CARD', col[0], col[1])
                                  for col in list(credit_card_partially_aggregated.columns)]
    
    credit_card_aggregated = credit_card_aggregated.join(credit_card_partially_aggregated, how='left', on='SK_ID_CURR')
    log.debug('Aggregated credit card dataframe has columns %s', credit_card_aggregated.columns)

    return credit_card_aggregated

def process_bureau(path_to_data='', sample_size=1000):
    '''
    Process both bureau and bureau balance datasets.
    '''

    # Load data and encode.
    bureau_data = pd.read_csv(path_to_data + 'bureau.csv', nrows=sample_size)
    bureau_balance_data = pd.read_csv(path_to_data + 'bureau_balance.csv', nrows=sample_size)

    bureau_data['DAYS_CREDIT_ENDDATE'][bureau_data['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    bureau_data['DAYS_CREDIT_UPDATE'][bureau_data['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    bureau_data['DAYS_ENDDATE_FACT'][bureau_data['DAYS_ENDDATE_FACT'] < -40000] = np.nan

    encode_categoricals(bureau_data)
    encode_categoricals(bureau_balance_data)

    # Add features before aggregation.
    # This introduces infinites, just pointing that out.
    # For now it's okay. 
    bureau_data['CREDIT_RATIO'] = bureau_data.AMT_CREDIT_SUM / bureau_data.AMT_CREDIT_SUM_LIMIT
    bureau_data['OVERDUE_RATIO'] = bureau_data.AMT_CREDIT_SUM_OVERDUE / bureau_data.AMT_CREDIT_SUM_LIMIT

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

    # Perform partial aggregations 
    partial_aggregation_functions = [mean_over_6, mean_over_12, mean_over_18, mean_over_24,
                                     std_over_6, std_over_12, std_over_18, std_over_24,
                                     mmd_over_6, mmd_over_12, mmd_over_18, mmd_over_24
    ]
    partial_aggregations = {
        'SK_DPD':partial_aggregation_functions,
        'SK_DPD_DEF':partial_aggregation_functions
    }
    cash_partially_aggregated = cash_data.sort_values(
        ['SK_ID_CURR', 'MONTHS_BALANCE']).groupby('SK_ID_CURR').agg(partial_aggregations)
    cash_partially_aggregated.columns = [rename_column('CASH', col[0], col[1])
                                  for col in list(cash_partially_aggregated.columns)]
    
    cash_aggregated = cash_aggregated.join(cash_partially_aggregated, how='left', on='SK_ID_CURR')
    

    log.debug('Aggregated cash dataframe has columns %s', cash_aggregated.columns)
    return cash_aggregated

def process_previous(path_to_data='', sample_size=1000):
    '''
    Processing cash dataset and making aggregations.
    '''

    previous_data = pd.read_csv(path_to_data + 'previous_application.csv', nrows=sample_size)
    encode_categoricals(previous_data)

    # Add features here.
    previous_data['DOWNPAYMENT_CREDIT_RATIO'] = previous_data.AMT_DOWN_PAYMENT / previous_data.AMT_CREDIT
    previous_data['CREDIT_APPLICATION_RATIO'] = previous_data.AMT_ANNUITY / previous_data.AMT_APPLICATION

    # Perform aggregations
    previous_aggregated = previous_data.groupby('SK_ID_CURR').aggregate(
        ['min', 'max', 'mean', 'var']
    )

    # Fix naming
    previous_aggregated.columns = [rename_column('PREV', col[0], col[1])
                                 for col in list(previous_aggregated.columns)]

    previous_aggregated['PREV_COUNT'] = previous_aggregated.groupby('SK_ID_CURR').size()
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

    # Sometimes this number is used to represent NaN
    dataset.replace(365243, np.nan, inplace= True)

    # There is no point in having columns that are 
    # empty.  Let's get rid of those. 
    percentage_threshold = 0.9
    missing = dataset.isnull().sum().sort_values(ascending=False) / len(dataset)
    drop_cols = [col for col in missing.index if missing[col] > percentage_threshold]
    dataset.drop(columns=drop_cols, inplace=True)
    log.info('Dropped columns with no information %s', drop_cols)

    # Print the types of columns that are still
    # present in our dataset before moving on.
    log.info('Column types: %s', dataset.dtypes.value_counts())

    # Save compressed testing and training set. 
    with timer('Writing training data'):
        train = dataset.loc[dataset.test == 0]
        train.to_csv(path_to_output + '1.4-features-train.csv', compression='gzip')

    with timer('Writing testing data'):
        test = dataset.loc[dataset.test == 1]
        test.drop(columns=['TARGET'], inplace=True)
        test.to_csv(path_to_output + '1.4-features-test.csv', compression='gzip')

if __name__ == '__main__':
    build_features()
