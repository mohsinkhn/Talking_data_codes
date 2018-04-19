######################################################################
# Script to generate train and test with added columns for "ip, device, os", "ip, device, os, app", "ip, deivce, os, app, channel" groups
# Having a integer column for these groups will help in feature generation
# 
# Read all grouper dictionaries
# We will read train/test/test_supplement in chunks(and in parallel)
#    - Extract hourofday, dayofweek
#    - Map all grouper dictionaries by creating a tmp column
#
# Not using multiprocessing for fear of messing up indices
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
handler = logging.FileHandler('base_traintest_generation.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

#

def process_file(filename,                 #input filename
                 new_filename,             #output filename
                 dict_cols,                #list of grouper columns for each dictionary  
                 group_dicts,              #list of dictionaries
                 chunksize=10**6):
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

    #Features for our new base dataframe
    base_feats = ['ip', 'app', 'device', 'os', 'channel', 'is_attributed','click_time' ,'dayofweek', 'hourofday',  
                  'ip_device_os', 'ip_device_os_app', 'ip_device_os_app_channel']

    #Create a empty datframe and just write out columns; we will write data in append mode to same file
    df = pd.DataFrame(columns=base_feats)
    df.to_csv(new_filename, index=False)
    
    logger.info("processing file {}".format(filename))
    for i, chunk in enumerate(pd.read_csv(filename, dtype=dtypes, chunksize=chunksize)):
        logger.info("Processing chunk {}".format(i))
        
        #Get date related features
        chunk['click_time'] = pd.to_datetime(chunk['click_time'])
        chunk['dayofweek'] = chunk['click_time'].dt.dayofweek
        chunk['hourofday'] = chunk['click_time'].dt.hour
        
        #For each set of grouper cols, map integer based on loaded dictionaries
        for cols, di in zip(dict_cols, group_dicts):
            chunk['tmp'] = list(chunk[cols].itertuples(index=False, name=None))
            col_name = '_'.join(cols)
            logger.info("mapping column {}".format(col_name))
            chunk[col_name] = chunk['tmp'].map(di)  #This step can be REALLY SLOW if dictionaries cannot be loaded in RAM
            del chunk['tmp']
        
        #Write out chunk in append mode
        chunk[base_feats].to_csv(new_filename, mode='a', header=False, index=False)
    
if __name__ == "__main__":
    
    #Read all dictionaries
    with open("../output/ip_device_os.pkl", "rb") as f:
        dict1 = pickle.load(f)
        cols1 = ['ip', 'device', 'os']
        
    with open("../output/ip_device_os_app.pkl", "rb") as f:
        dict2 = pickle.load(f)
        cols2 = ['ip', 'device', 'os', 'app']
        
    with open("../output/ip_device_os_app_channel.pkl", "rb") as f:
        dict3 = pickle.load(f)
        cols3 = ['ip', 'device', 'os', 'app', 'channel']
        
    all_dicts = [dict1, dict2, dict3]
    all_cols = [cols1, cols2, cols3]
    
    #Process all files
    process_file("../input/train.csv", "../input/train_base.csv", all_cols, all_dicts, chunksize=10**6)
    process_file("../input/test.csv", "../input/test_base.csv", all_cols, all_dicts, chunksize=10**6)
    process_file("../input/test_supplement.csv", "../input/test_supplement_base.csv", all_cols, all_dicts, chunksize=10**6)


