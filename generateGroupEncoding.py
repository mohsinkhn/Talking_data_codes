######################################################################
# Script to generate categorical feature for combinations of "ip, device, os" and "ip, device, os, app"
# Author: Mohsin Hasan Khan
######################################################################

import pandas as pd
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
handler = logging.FileHandler('combination_feature_generation.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def setit(chunk, cols):
    """
    Given a chunk return a list of tuples for given cols
    """
    return list(chunk[cols].itertuples(index=False, name=None))
    
def gen_args(chunk, cols):
    """
    Helper function for Pool.starmap() to generate a iterable of arguments
    """
    for c in chunk:
        yield (c, cols)
        
def get_group_dict(cols, filename="", dtype=np.uint32, 
                    train_filepath = "../input/train.csv", 
                    test_filepath = "../input/test_supplement.csv",
                    out_filepath = "../output",
                    num_procs=10, 
                    chunksize=10**6):
    """
    Gets dictionary with all unique tuple of given columns in both train and test(supplement) as keys and a integer mapping as its value
    """
    #Use known dtypes to reduce memory footprint during loading of dataframe chunks
    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }


    results = set() # Save all the counter elements here
    
    with mp.Pool(num_procs) as pool:
        #Iterators for train and test files
        tr_iterator = pd.read_csv(train_filepath, dtype=dtypes, usecols=cols, chunksize=chunksize)
        te_iterator = pd.read_csv(test_filepath, dtype=dtypes, usecols=cols, chunksize=chunksize)
        
        run_cnt = 0
        #queue up chunks same as num_procs for parallel processing
        for chunks in iter(lambda: list(IT.islice(tr_iterator, num_procs)), []): 
            args = gen_args(chunks, cols)
            result = pool.starmap(setit, args) #parallely process all chunks
            results.update(*result)
            run_cnt += 1
            logger.info("Finished iter {}".format(run_cnt))
            
        for chunk in iter(lambda: list(IT.islice(te_iterator, num_procs)), []):
            args = gen_args(chunk, cols)
            result = pool.starmap(setit, args) #parallely process all chunks
            results.update(*result)
            run_cnt += 1
            logger.info("Finished iter {}".format(run_cnt))
            
    logger.info(len(results)) #Check total unique keys in dictionary
    
    #Convert set to dict with indices (integer encoding for keys)
    cols_dict = {k:i for i,k in enumerate(results)}
    col_name = '_'.join(cols)
    
    #If filename not present create one save dictionary as pickled object
    if not(filename):
        filename = os.path.join(out_filepath, col_name + '.pkl')
    with open(filename, "wb") as f:
        pickle.dump(cols_dict, f)
    


if __name__ == "__main__":

    NUM_PROCS = 5
    CHUNKSIZE = 10 ** 6 #Make sure NUM_PROCS*CHUNKSIZE fits RAM
    TRAIN_FILEPATH  = "../input/train.csv"
    TEST_FILEPATH = "../input/test_supplement.csv"
    OUT_FILEPATH = "../output"

    start_time = time.time() #Keep a timer for each dictionary generation
    for cols in [ #Choosing column combinations based on understanding of data so far
               ['ip', 'device', 'os'],
               ['ip', 'device', 'os', 'app'],
               ['ip', 'device', 'os', 'app', 'channel'],
            ]:  
        get_group_dict(cols=cols, dtype=np.uint32,
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
