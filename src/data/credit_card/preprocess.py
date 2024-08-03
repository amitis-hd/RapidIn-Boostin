"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
import util


def main(test_size=0.2, seed=1):

    # create logger
    logger = util.get_logger('log.txt')
    logger.info('timestamp: {}'.format(datetime.now()))

    # retrieve dataset
    start = time.time()
    df = pd.read_excel('default of credit card clients.xls', header=1)
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # split data into train and test
    train_df, test_df = train_test_split(df, test_size=test_size,
                                         random_state=seed, stratify=df['default payment next month'])

    # get features
    columns = list(train_df.columns)

    # remove select columns
    remove_cols = ['ID']
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['default payment next month']
    features['numeric'] = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
                           'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    X_train, y_train, X_test, y_test, feature = util.preprocess(train_df, test_df, features, logger=logger)

    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'feature': feature, 'train_df': train_df, 'test_df': test_df}

    logger.info(f'train (head): {X_train[:5]}, {y_train[:5]}')
    logger.info(f'test (head): {X_test[:5]}, {y_test[:5]}')
    logger.info(f'feature (head): {feature[:5]}')
    logger.info(f'X_train.shape: {X_train.shape}')
    logger.info(f'X_test.shape: {X_test.shape}')
    logger.info(f'y_train.shape: {y_train.shape}, y_train.sum: {y_train.sum()}')
    logger.info(f'y_test.shape: {y_test.shape}, y_test.sum: {y_test.sum()}')

    # save to numpy format
    np.save(os.path.join('data.npy'), data)


if __name__ == '__main__':
    main()
