
import pandas as pd
import numpy as np

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import *

import lightgbm as lgb
import os
import gc
import pickle
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('genTimeDiffs.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)   

if __name__ == "__main__":
    SEED = 786
    OUT_PATH = "../output"
    DTYPES = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            'hourofday'     : 'uint8',
            'dayofweek'     : 'uint8',
            'ip_device_os'     : 'uint32',
            'ip_device_os_app'     : 'uint32',
            'ip_device_os_app_channel' : 'uint32',

            }
    
    logger.info("Reading train and test")
    train = pd.read_csv("../input/train_base.csv", dtype=DTYPES, 
                       )
    test = pd.read_csv("../input/test_base.csv", dtype=DTYPES)
    test["is_attributed"] = np.nan
    
    logger.info("Get time details")
    train = time_details(train)
    test = time_details(test)
    
    logger.info("Break train into tr and val")
    cond = (train.dayofweek == 3) & (train.hourofday.isin([4,5,9,10,13,14]))
    cond2 = ((train.dayofweek == 3) & (train.hourofday < 4)) | (train.dayofweek < 3)
    
    tr = train.loc[cond2].reset_index(drop=True) ###
    val = train.loc[cond].reset_index(drop=True)
    y_val = val["is_attributed"]
    
    logger.info("Shape of train and test is {} and {}".format(train.shape, test.shape))
    logger.info("Shape of tr and val is {} and {}".format(tr.shape, val.shape))
    
    
    logger.info("Generate cumulative count features")
    feats2 = []
    for col in ['ip', 'ip_device_os', 'ip_device_os_app', 'ip_device_os_app_channel']:
        logger.info("Processing feature: {}".format(col))
        for SHIFT in [1,2]:
            col_name = "_".join([col]) + "_prev_click"
            logger.info("Gnerating feature: {} {} for tr/val set".format(col_name, SHIFT))
            get_prev_click(tr, val, [col], target='epoch_time', shift=SHIFT,
                            tr_filename=os.path.join(OUT_PATH, "tr_{}_{}.npy".format(col_name, SHIFT)),  
                            val_filename=os.path.join(OUT_PATH, "val_{}_{}.npy".format(col_name, SHIFT)), 
                            seed=786, rewrite=False)
            
            logger.info("Gnerating feature: {} for train/test set".format(col_name))
            get_prev_click(train, test, [col], target='epoch_time', shift=SHIFT,
                            tr_filename=os.path.join(OUT_PATH, "train_{}_{}.npy".format(col_name, SHIFT)),  
                            val_filename=os.path.join(OUT_PATH, "test_{}_{}.npy".format(col_name, SHIFT)), 
                            seed=786, rewrite=False)

            col_name = "_".join([col]) + "_next_click"
            logger.info("Gnerating feature: {} {} for tr/val set".format(col_name, SHIFT))
            
            get_next_click(tr, val, [col], target='epoch_time', shift=-SHIFT,
                            tr_filename=os.path.join(OUT_PATH, "tr_{}_{}.npy".format(col_name, SHIFT)),  
                            val_filename=os.path.join(OUT_PATH, "val_{}_{}.npy".format(col_name, SHIFT)), 
                            seed=786, rewrite=False)
            
            logger.info("Gnerating feature: {} {} for train/test set".format(col_name, SHIFT))
            get_next_click(train, test, [col], target='epoch_time', shift=-SHIFT,
                            tr_filename=os.path.join(OUT_PATH, "train_{}_{}.npy".format(col_name, SHIFT)),  
                            val_filename=os.path.join(OUT_PATH, "test_{}_{}.npy".format(col_name, SHIFT)), 
                            seed=786, rewrite=False)
    
    logger.info("Successfully Completed")
    
    

