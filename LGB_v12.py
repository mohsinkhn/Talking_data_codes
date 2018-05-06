
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.preprocessing import QuantileTransformer

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
handler = logging.FileHandler('LGB_v11.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def normalize_cols(tr, val, train, test, cols):
    qnt = QuantileTransformer(output_distribution="normal")
    tr[cols] = qnt.fit_transform(tr[cols]).astype(np.float32)
    val[cols] = qnt.transform(val[cols]).astype(np.float32)
    
    train[cols] = qnt.fit_transform(train[cols]).astype(np.float32)
    test[cols] = qnt.transform(test[cols]).astype(np.float32)


# In[ ]:


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
    test["is_attributed"] = 0
    
    logger.info("Get time details")
    train = time_details(train)
    test = time_details(test)
    
    logger.info("Break train into tr and val")
    cond = (train.dayofweek == 3) & (train.hourofday.isin([4,5,9,10,13,14]))
    cond2 = ((train.dayofweek == 3) & (train.hourofday < 4)) | (train.dayofweek < 3)

    tr = train.loc[cond2] ###
    val = train.loc[cond]
    y_val = val["is_attributed"]
    
    logger.info("Shape of train and test is {} and {}".format(train.shape, test.shape))
    logger.info("Shape of tr and val is {} and {}".format(tr.shape, val.shape))
    
    base_feats = ['ip', 'app', 'device', 'os', 'channel', 'ip_device_os']
    
    logger.info("Generating expanding mean features")
    tr, val, train, test, feats_expmean = load_expmean_features(tr, val, train, test, logger)
    print(tr.shape, val.shape, train.shape, test.shape)
    
    logger.info("Generating time diff features")
    tr, val, train, test, feats_timediffs = load_timediff_features(tr, val, train, test, logger)
    print(tr.shape, val.shape, train.shape, test.shape)
    
    logger.info("Generating count features")
    tr, val, train, test, feats_count = load_count_features(tr, val, train, test, logger)
    print(tr.shape, val.shape, train.shape, test.shape)
    
    logger.info("Generating expanding count features")
    tr, val, train, test, feats_count = load_expcount_features(tr, val, train, test, logger)
    print(tr.shape, val.shape, train.shape, test.shape)
    
    logger.info("Generating nunique features")
    tr, val, train, test, feats_unq = load_unq_features(tr, val, train, test, logger)
    print(tr.shape, val.shape, train.shape, test.shape)
    
    LGB_PARAMS3 = {
            "learning_rate": 0.1, 
            "n_estimators" : 3000, ###
             "max_depth"    : 3,
             "subsample"    : 0.8,
             "colsample_bytree": 0.8,
             "min_child_samples": 1000,
             "reg_lambda"       : 0,
             "scale_pos_weight" : 200, 
             "num_leaves"       : 7,
             "n_jobs"           :-1
              }
    
    LGB_PARAMS4 = {
            "learning_rate": 0.1, 
            "n_estimators" : 2000, ###
             "max_depth"    : 3,
             "subsample"    : 0.7,
             "colsample_bytree": 0.9,
             "min_child_samples": 500,
             "reg_lambda"       : 10,
             "scale_pos_weight" : 100, 
             "num_leaves"       : 7,
             "n_jobs"           :-1
              }
    
    LGB_PARAMS5 = {
            "learning_rate": 0.1, 
            "n_estimators" : 1000, ###
             "max_depth"    : 4,
             "subsample"    : 0.8,
             "colsample_bytree": 0.6,
             "min_child_samples": 200,
             "reg_lambda"       : 1,
             "scale_pos_weight" : 200, 
             "num_leaves"       : 15,
             "n_jobs"           :-1
              }
    
    extra_feats = ['hourofday' ,'app_expmean', 'channel_expmean', 'os_expmean',
       'device_expmean', 'ip_expmean', 'ip_device_os_expmean',
       'ip_prev_click_1', 'ip_next_click_1',  'ip_device_os_prev_click_1',
       'ip_device_os_next_click_1', 'ip_device_os_app_prev_click_1', 'ip_device_os_app_prev_click_2',
       'ip_device_os_app_next_click_1', 'ip_device_os_app_next_click_2', 'app_count', 'channel_count',
       'os_count', 'ip_count', 'ip_device_os_count', 'ip_device_os_app_count',
       'ip_device_os_app_channel_count', 'app_expcount', 'channel_expcount',
       'ip_expcount', 'ip_unq_app']
    
    feats = base_feats + extra_feats
    
    tr, val, train, test = normalize_cols(tr, val, train, test, extra_feats)
    
    logger.info("Fitting model with param set 1")
    model, val_preds = run_lgb(tr, val, LGB_PARAMS1, logger, feats=feats, is_develop=True, save_preds=False)
    np.save("../output/val_preds_v11_set1.npy", val_preds)
    LGB_PARAMS1["n_estimators"] = int(model.best_iteration_ * 184/144)
    model, test_preds = run_lgb(train, test, LGB_PARAMS1, logger, feats=feats, is_develop=False, save_preds=False)
    np.save("../output/test_preds_v11_set1.npy", test_preds)
    prepare_submission(test_preds, "../output/LGBv11_set1.csv")
    
    logger.info("Fitting model with param set 2")
    model, val_preds = run_lgb(tr, val, LGB_PARAMS2, logger, feats=feats, is_develop=True, save_preds=False)
    np.save("../output/val_preds_v11_set2.npy", val_preds)
    LGB_PARAMS2["n_estimators"] = int(model.best_iteration_ * 184/144)
    model, test_preds = run_lgb(train, test, LGB_PARAMS2, logger, feats=feats, is_develop=False, save_preds=False)
    np.save("../output/test_preds_v11_set2.npy", test_preds)
    prepare_submission(test_preds, "../output/LGBv11_set2.csv")
    
    logger.info("Fitting model with param set 3")
    model, val_preds = run_lgb(tr, val, LGB_PARAMS3, logger, feats=feats, is_develop=True, save_preds=False)
    np.save("../output/val_preds_v11_set3.npy", val_preds)
    LGB_PARAMS3["n_estimators"] = int(model.best_iteration_ * 184/144)
    model, test_preds = run_lgb(train, test, LGB_PARAMS3, logger, feats=feats, is_develop=False, save_preds=False)
    np.save("../output/test_preds_v11_set3.npy", test_preds)
    prepare_submission(test_preds, "../output/LGBv11_set3.csv")

