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
handler = logging.FileHandler('count_feature_generation.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def cntit(chunk, cols):
    """
    Given a chunk return a Counter object over tuples of given cols
    """
    return Counter(list(chunk[cols].itertuples(index=False, name=None)))
    #return list(zip(*reduce(lambda x,y : (x,y), [chunk[col] for col in cols])))

def gen_args(chunk, cols):
    """
    Helper function for Pool.starmap() to generate a iterable of arguments
    """
    for c in chunk:
        yield (c, cols)
        
def get_cnt_feature(cols, filename="", dtype=np.uint32, 
                    train_filepath = "../input/train.csv", 
                    test_filepath = "../input/test_supplement.csv",
                    out_filepath = "../output",
                    num_procs=10, 
                    chunksize=10**6):
    """
    Gets counts over all unique tuple of given columns in both train and test(supplement) data
    """
    #Use known dtypes to reduce memory footprint during loading of dataframe chunks
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


    results = Counter() # Save all the counter elements here
    
    with mp.Pool(num_procs) as pool:
        #Iterators for train and test files
        tr_iterator = pd.read_csv(train_filepath, dtype=dtypes, usecols=cols, chunksize=chunksize)
        te_iterator = pd.read_csv(test_filepath, dtype=dtypes, usecols=cols, chunksize=chunksize)
        
        run_cnt = 0
        #queue up chunks same as num_procs for parallel processing
        for chunks in iter(lambda: list(IT.islice(tr_iterator, num_procs)), []): 
            args = gen_args(chunks, cols)
            result = pool.starmap(cntit, args) #parallely process all chunks
            for r in result:
                results = results + r
            run_cnt += 1
            logger.info("Finished iter {}".format(run_cnt))
            
        for chunk in iter(lambda: list(IT.islice(te_iterator, num_procs)), []):
            args = gen_args(chunk, cols)
            result = pool.starmap(cntit, args) #parallely process all chunks
            for r in result:
                results = results + r
            run_cnt += 1
            logger.info("Finished iter {}".format(run_cnt))
            
    all_cnts = results
    gc.collect()
        
    logger.info(len(all_cnts)) #Check total unique keys in counter
    
    #Convert counter object to pandas series for easier integration with pandas dataframe for downstream tasks
    col_name = '_'.join(cols) + "_cnt"
    cnt_series = pd.Series(all_cnts, name='col_name', dtype=dtype)
    cnt_series.index.names = cols
    if len(cols) == 1:
        cnt_series.index = cnt_series.index.levels[0]
    
    logger.info(cnt_series.head())
    #If filename not present create one save series as pickled object
    if not(filename):
        filename = os.path.join(out_filepath, col_name + '.pkl')
    with open(filename, "wb") as f:
        pickle.dump(cnt_series, f)
    


if __name__ == "__main__":

    NUM_PROCS = 8
    CHUNKSIZE = 1 * 10 ** 7 #Make sure NUM_PROCS*CHUNKSIZE fits RAM
    TRAIN_FILEPATH  = "../input/train_base.csv"
    TEST_FILEPATH = "../input/test_supplement_base.csv"
    OUT_FILEPATH = "../output"

    start_time = time.time() #Keep a timer for each feature generation
    for cols in [ #Choosing column combinations based on intituition
               ['ip'], ['ip', 'app'], ['app'], ['app', 'channel'], ['device', 'os'],
               ['ip', 'app', 'hourofday'], ['device', 'os', 'app'], ['device', 'os', 'channel'],
               ['ip_device_os'], ['ip_device_os', 'hourofday'],
               ['ip_device_os_app'], ['ip_device_os_app_channel'],
            ]:
        if len(cols) > 3:
            NUM_PROCS = 2    
        get_cnt_feature(cols=cols, dtype=np.uint32,
                       train_filepath=TRAIN_FILEPATH,
                       test_filepath=TEST_FILEPATH,
                       out_filepath=OUT_FILEPATH,
                       num_procs=NUM_PROCS,
                       chunksize=CHUNKSIZE
                       )
        total_time = time.time() - start_time
        logger.info("Count feature generation for {} finished in {}".format(cols, total_time))
        gc.collect()
        start_time = time.time()
