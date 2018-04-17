######################################################################
# Script to generate count features for different column combinations using booth train and test data
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



def cntit(chunk, cols):
    """
    Given a chunk return a Counter object over tuples of given cols
    """
    return Counter(list(chunk[cols].itertuples(index=False, name=None)))
    
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
        'click_id'      : 'uint32'
        }


    results = [] # Save all the counter elements here
    
    with mp.Pool(num_procs) as pool:
        #Iterators for train and test files
        tr_iterator = pd.read_csv(train_filepath, dtype=dtypes, usecols=cols, chunksize=chunksize)
        te_iterator = pd.read_csv(test_filepath, dtype=dtypes, usecols=cols, chunksize=chunksize)
        
        run_cnt = 0
        #queue up chunks same as num_procs for parallel processing
        for chunks in iter(lambda: list(IT.islice(tr_iterator, num_procs)), []): 
            args = gen_args(chunks, cols)
            result = pool.starmap(cntit, args) #parallely process all chunks
            results.extend(result)
            run_cnt += 1
            print("Finished iter {}".format(run_cnt))
            
        for chunk in iter(lambda: list(IT.islice(te_iterator, num_procs)), []):
            args = gen_args(chunk, cols)
            result = pool.starmap(cntit, args) #parallely process all chunks
            results.extend(result)
            run_cnt += 1
            print("Finished iter {}".format(run_cnt))
            
    all_cnts = reduce(lambda x, y: x + y, results) #Add all counter objects to get aggregated counter
        
    print(len(all_cnts)) #Check total unique keys in counter
    
    #Convert counter object to pandas series for easier integration with pandas dataframe for downstream tasks
    col_name = '_'.join(cols) + "_cnt"
    cnt_series = pd.Series(all_cnts, name='col_name', dtype=dtype)
    cnt_series.index.names = cols
    if len(cols) == 1:
        cnt_series.index = cnt_series.index.levels[0]
    
    #If filename not present create one save series as pickled object
    if not(filename):
        filename = os.path.join(out_filepath, col_name + '.pkl')
    with open(filename, "wb") as f:
        pickle.dump(cnt_series, f)
    


if __name__ == "__main__":

    NUM_PROCS = 5
    CHUNKSIZE = 10 ** 6 #Make sure NUM_PROCS*CHUNKSIZE fits RAM
    TRAIN_FILEPATH  = "../input/train.csv"
    TEST_FILEPATH = "../input/test_supplement.csv"
    OUT_FILEPATH = "../output"

    start_time = time.time() #Keep a timer for each feature generation
    for cols in [ #Choosing column combinations based on intituition
               ['ip'], ['ip', 'device', 'os'], ['device', 'os', 'app'],
               ['app'], ['app', 'channel'], ['ip', 'app'], 
               ['device', 'os'], ['device', 'os', 'channel'],
               ['ip', 'device', 'os', 'app'],
               ['ip', 'device', 'os', 'app', 'channel'],
            ]:
    
        get_cnt_feature(cols=cols, dtype=np.uint32,
                       train_filepath=TRAIN_FILEPATH,
                       test_filepath=TEST_FILEPATH,
                       out_filepath=OUT_FILEPATH,
                       num_procs=NUM_PROCS,
                       chunksize=CHUNKSIZE
                       )
        total_time = time.time() - start_time
        print("Count feature generation for {} finished in {}".format(cols, total_time))
        gc.collect()
        start_time = time.time()

