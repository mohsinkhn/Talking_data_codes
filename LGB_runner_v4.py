import pandas as pd
#import ray.dataframe as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import *

import lightgbm as lgb
import os
import gc
import pickle


if __name__ == "__main__":
    
    VALID_HOURS = [14] #Make it 4,9,14 when training over whole data
    VALID_DAY = 3
    SKIPROWS = 150000000 #Make it 1 when training over whole data
    
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
            'ip_device_os'     : 'uint32',
            'ip_device_os_app'     : 'uint32',
            'ip_device_os_app_channel' : 'uint32'
            }

    #Read data
    print("Reading train data")
    train = pd.read_csv("../input/train_base.csv", 
                        usecols=['ip', 'app', 'device', 'os', 'channel', 'hourofday', 'dayofweek', 
                                 'ip_device_os', 'ip_device_os_app', 'ip_device_os_app_channel', 'is_attributed'],
                        dtype=dtypes, skiprows = list(range(1,SKIPROWS)))
    
    print("reading test data")
    test = pd.read_csv("../input/test_base.csv", 
                        usecols=['ip', 'app', 'device', 'os', 'channel', 'hourofday', 'dayofweek', 
                                 'ip_device_os', 'ip_device_os_app', 'ip_device_os_app_channel'],
                        dtype=dtypes)
    
    print(train.shape, test.shape)
    
    print("Get tr and validation sets")
    cond = train.hourofday.isin(VALID_HOURS) & (train['dayofweek'] == VALID_DAY)
    tr = train.loc[~cond]
    val = train.loc[cond]

    print(tr.shape, val.shape)
    
    #Model with just base features
    feats1 = ['ip', 'app', 'device', 'os', 'channel', 'hourofday', 'ip_device_os', 
             'ip_device_os_app', 'ip_device_os_app_channel']            #9 features
    
    #print("Fit model with base features")
    #model = lgb.LGBMClassifier(n_estimators=1000, max_depth=3, subsample=0.8,
    #                       num_leaves=8, min_child_samples=2000, n_jobs=-1)
    #model.fit( tr[feats], tr['is_attributed'], eval_set=[(val[feats], val['is_attributed'])], eval_metric='auc', 
    #      verbose=10, early_stopping_rounds=100,)
    
    #test_preds = model.predict_proba(test[feats])[:, 1]
    
    #Read sub file and prepare submission
    #sub = pd.read_csv("../input/sample_submission.csv")
    #sub['is_attributed'] = test_preds
    #print(sub.head())
    #sub.to_csv("../output/lgb_base9feats.csv", index=False)
    
    #del sub
    
    gc.collect()
    
    OUT_PATH = "../output"
    feats2 = []
    for cols in [['ip'], ['ip', 'app'], ['app'], ['app', 'channel'], ['device', 'os'],
               ['ip', 'app', 'hourofday'], ['device', 'os', 'app'], ['device', 'os', 'channel'],
               ['ip_device_os'], ['ip_device_os', 'hourofday'],
               ['ip_device_os_app'], ['ip_device_os_app_channel']]:
        
        col_name = '_'.join(cols) + "_cnt"
        file_name = col_name + ".pkl"
        with open(os.path.join(OUT_PATH, file_name), "rb") as f:
            cnts = pickle.load(f)

        print(cnts.head())
        cnts.name = col_name
        feats2.append(col_name)
        #del tr, val
        train = train.join(cnts, on=cols, how='left')
        test = test.join(cnts, on=cols, how='left')
    tr = train.loc[~cond]
    val = train.loc[cond]

    print(tr.shape, val.shape)
    
    print("Fit model with cnt features")
    feats = feats1 + feats2
    #print("Total features are ", len(feats))
    #model = lgb.LGBMClassifier(n_estimators=1500, max_depth=3, subsample=0.8,
    #                       num_leaves=8, min_child_samples=2000, n_jobs=-1)
    #model.fit( tr[feats], tr['is_attributed'], eval_set=[(val[feats], val['is_attributed'])], eval_metric='auc', 
    #      verbose=10, early_stopping_rounds=100,)
    
    
    ##Get time features
    feats3 = []
    for col in ['ip', 'ip_device_os', 'ip_device_os_app', 'ip_device_os_app_channel']:
        count_col  = col + "_timecount"
        diff_col = col + "_timediff"
        
        train_ft = pd.read_pickle(os.path.join(OUT_PATH, "train_time_"+col+".pkl"))
        test_ft = pd.read_pickle(os.path.join(OUT_PATH, "train_time_"+col+".pkl"))
        
        for f in [count_col, diff_col]:
            feats3.append(f)
            train[f] = train_ft[f].values[SKIPROWS-1:]
            test[f] = test_ft[f].values

    tr = train.loc[~cond]
    val = train.loc[cond]
    print(train.head())
    print("Fit model with time features")
    feats = feats1 + feats2 + feats3
    print("Total features are ", len(feats))
    model = lgb.LGBMClassifier(n_estimators=1500, max_depth=4, subsample=0.7, colsample_bytree=0.7, scale_pos_weight=9,
                           num_leaves=15, min_child_samples=1000, n_jobs=-1)
    model.fit( tr[feats], tr['is_attributed'], eval_set=[(val[feats], val['is_attributed'])], eval_metric='auc', 
          verbose=10, early_stopping_rounds=100,)
    test_preds = model.predict_proba(test[feats])[:, 1]
    
    #Read sub file and prepare submission
    sub = pd.read_csv("../input/sample_submission.csv")
    sub['is_attributed'] = test_preds
    print(sub.head())
    sub.to_csv("../output/lgb_base29feats.csv", index=False)
    
    
