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

from TargetEncoder import TargetEncoder
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('LGB_v8.log')
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
    LGB_PARAMS = {
                 "n_estimators" : 1500, ###
                 "max_depth"    : 3,
                 "subsample"    : 0.7,
                 "colsample_bytree": 0.7,
                 "min_child_samples": 200,
                 "reg_lambda"       : 0.1,
                 "scale_pos_weight" : 200, 
                 "num_leaves"       : 7,
                 "n_jobs"           :-1
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
    
    base_feats = ['ip', 'app', 'device', 'os', 'channel', 'hourofday', 'ip_device_os'] #+ COUNT_COLS + MEAN_COLS
    
    logger.info("Generate count features")
    
    tr, val, train, test, feats_count = load_count_features(tr, val, train, test, logger)
    tr, val, train, test, feats_expmean = load_expmean_features(tr, val, train, test, logger)
    tr, val, train, test, feats_unq = load_unq_features(tr, val, train, test, logger)
    
    feats = base_feats + feats_count + feats_unq #++ feats_expmean
    #feats.remove(['ip_expmean'])
    logger.info("Running LGB for develop set with feats {}".format(feats))
    model, val_preds = run_lgb(tr, val, LGB_PARAMS, logger, feats=feats, is_develop=True, save_preds=False)
    #score = roc_auc_score(y_val, val_preds)
        
    #logger.info("Running LGB for train/test set with feats {}".format(feats))
    #model, test_preds = run_lgb(train, test, LGB_PARAMS, logger, feats=feats, is_develop=False, save_preds=False)
    test_preds = model.predict_proba(test[feats])[:,1]
    prepare_submission(test_preds, save_path="../output/LGBv8_feats{}_test_preds.csv".format(len(feats)))