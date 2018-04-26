######################################################################
# Script to generate count features for different column combinations using booth train and test data
# Author: Mohsin Hasan Khan
######################################################################

import pandas as pd
#import ray.dataframe as pd
import numpy as np
import pickle
import os
from collections import Counter

import multiprocessing as mp

import itertools as IT

from functools import reduce
import gc
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('time_feature_generation.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def group_diff(garr, tarr, na_val=7200):
    #sidx = np.argsort(group_arr)
    #garr = group_arr[sidx]
    #tarr = target_arr[sidx]
    #print(tarr)
    grp_change = (garr[1:] != garr[:-1]) 
    #same_val = np.concatenate(([True], (tarr[1:] != tarr[:-1])))
    
    tarr_diff = np.diff(tarr)
    tarr_diff[grp_change] = na_val
    #print(tarr_diff)
    tarr_diff = np.concatenate(([na_val], tarr_diff))
    #print(np.argsort(sidx))
    return tarr_diff #tarr_diff[np.argsort(sidx)]
    
def group_agg(data):
    unq, ix, tags, count= np.unique(data, axis=0, 
                                    return_inverse=True, return_counts=True, return_index=True)
    logger.info("Getting counts for aggregator")
    cc = np.bincount(tags)
    logger.info("Getting diffs for groupby epoch time")
    diffs = group_diff(unq[:,0], unq[:, 1], 10**6)
    arr = np.c_[unq, cc, diffs][tags]
    return arr[:,2], arr[:,3]

def get_test_from_supp(test, test_supp, match_cols=[0,1], map_cols=[2,3]):
    unq, tags,  = np.unique(test[:,match_cols],  return_inverse=True, axis=0)
    unq2, idx2 = np.unique(test_supp[:, match_cols], return_index=True, axis=0)
    unq2 = np.c_[unq2, test_supp[idx2, :][:,map_cols]]
    
    test_unq = pd.merge( pd.DataFrame(unq, columns=match_cols),
              pd.DataFrame(unq2, columns=match_cols+map_cols), on=match_cols, how='left')
    test = test_unq.values[tags]
    return test

def read_prepare_data(input_path, col):
    df = pd.read_csv(input_path, usecols=[col, 'click_time'], dtype=dtypes)
    df['epoch_time'] = ((pd.to_datetime(df['click_time']) - pd.to_datetime("2017-11-06 14:00:00"))/10**9).astype(np.int64)
    del df['click_time']
    return df

def get_time_feats(TRAIN_PATH, TEST_PATH, TEST_SUPPLEMENT_PATH, col):
    logger.info("Reading train file")
    train = read_prepare_data(TRAIN_PATH, col)
    logger.info("Reading test supplement file")
    test_supp = read_prepare_data(TEST_SUPPLEMENT_PATH, col)
    logger.info("Reading test file")
    test = read_prepare_data(TEST_PATH, col)
    
    return train, test_supp, test
        
def process_time_feat(col, dtype=np.uint32, 
                    train_filepath = "../input/train_base.csv", 
                    test_filepath = "../input/test_base.csv",
                    test_supp_filepath = "../input/test_supplement_base.csv",
                    out_filepath = "../output"
                     ):
    """
    Gets counts over all unique tuple of given columns in both train and test(supplement) data
    """
    #Use known dtypes to reduce memory footprint during loading of dataframe chunks


    train, test_supp, test = get_time_feats(train_filepath, test_filepath, test_supp_filepath, col=col)
    logger.info("Shape are {} {} {} ".format(train.shape, test.shape, test_supp.shape))
    gc.collect()
    
    data = np.vstack((train[[col,'epoch_time']].values, test_supp[[col,'epoch_time']].values))
    ar1, ar2 = group_agg(data)
    
    train[col+'_timecount'] = ar1[:len(train)]
    test_supp[col+'_timecount'] = ar1[len(train):]

    train[col+'_timediff'] = ar2[:len(train)]
    test_supp[col+'_timediff'] = ar2[len(train):]

    logger.info("Shapes after adding new columns for train and test supp {} and {}".format(train.shape, test_supp.shape))
    
    gc.collect()
    
    test = get_test_from_supp(test.values, test_supp.values)
    test = pd.DataFrame(test, columns=train.columns)
    logger.info("Test shape {}".format(test.shape))
    print(test.head())
    
    feats = [col+'_timecount', col+'_timediff']
    train[feats].to_pickle(os.path.join(out_filepath, "train_time_"+col + ".pkl"))
    test[feats].to_pickle(os.path.join(out_filepath, "test_time_"+col + ".pkl"))

    del train
    del test
    del test_supp


if __name__ == "__main__":

    TRAIN_FILEPATH  = "../input/train_base.csv"
    TEST_FILEPATH = "../input/test_base.csv"
    TEST_SUPP_FILEPATH = "../input/test_supplement_base.csv"
    OUT_FILEPATH = "../output"

    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        'hourofday'     : 'uint8',
        'dayofweek'     : 'uint8',
        'ip_device_os'  : 'uint32',
        'ip_device_os_app': 'uint32',
        'ip_device_os_app_channel' : 'uint32'
        }
    start_time = time.time() #Keep a timer for each feature generation
    for col in [ #Choosing column combinations based on intituition
                'ip',
               'ip_device_os',
               'ip_device_os_app', 
               'ip_device_os_app_channel',
            ]:  
        process_time_feat(col=col, dtype=np.uint32,
                       train_filepath=TRAIN_FILEPATH,
                       test_filepath=TEST_FILEPATH,
                       test_supp_filepath=TEST_SUPP_FILEPATH,
                       out_filepath=OUT_FILEPATH,
                       )
        total_time = time.time() - start_time
        logger.info("Time feature generation for {} finished in {}".format(col, total_time))
        gc.collect()
        start_time = time.time()
