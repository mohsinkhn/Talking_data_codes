
# coding: utf-8

# In[2]:


import pandas as pd
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
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('LGB_v6.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    A utlity class to help encode categorical variables using different methods.
    
    Inputs:
    cols: (List or str) Can be either a string or list of strings with column names
    targetcol: (str) Target column to encode column/group of columns with
    thresh: (int) Minimum count of grouping to encode (Acts as smoothing). Currently not implemented TODO
    func: (str or callable) Function to be applied on column/ group of columns to encode. 
          If str is provided, it should be a attribute of pandas Series
    cname: (str) Column name for new string
    func_kwargs: (dict) Additional arguments to be passed to function 
    add_to_orig: (bool) Whether to return dataframe with added feature or just the feature as series
    
    Output:
    pandas DataFrame/Series
    
    """
    def __init__(self, cols=None, targetcol=None, cname=None, thresh=0, func=np.mean,  add_to_orig=False,                              func_kwargs={}):
        self.cols = cols #Can be either a string or list of strings with column names
        self.targetcol = targetcol #Target column to encode column/group of columns with
        self.thresh = thresh  #Minimum count of grouping to encode (Acts as smoothing)
        self.func = func #Function to be applied on column/ group of columns to encode 
        self.add_to_orig = add_to_orig #Whether return a dataframe with added feature or just a series of feature
        self.cname = cname #Column to new feature generated
        self.func_kwargs = func_kwargs  #Additional key word arguments to be applied to func
    
    #@numba.jit        
    def fit(self, X, y=None):
            
        if isinstance(self.func, str):
            if hasattr(pd.Series, self.func):
                #print("here")
                vals = getattr(X.groupby(self.cols)[self.targetcol], self.func)
                self.dictmap = vals(**self.func_kwargs)
                
        else:
            self.dictmap = X.groupby(self.cols)[self.targetcol].apply(lambda x: self.func(x, **self.func_kwargs))
            
        if self.cname:
            self.dictmap.name = self.cname
        else:
            cname = ''
            cname = [cname + '_' +str(col) for col in self.cols]
            self.cname = '_'.join(cname) + "_" + str(self.func)
            self.dictmap.name = self.cname
            
        #print(self.cname)
        self.dictmap = self.dictmap
        return self
    
    #@numba.jit
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_transformed = X[self.cols]
            
            X_transformed = X_transformed.join(self.dictmap, on=self.cols, how='left')[self.cname]

            if self.add_to_orig:
                return pd.concat([X, X_transformed], axis=1, copy=False)
            else:
                return X_transformed.values

        else:
            raise TypeError("Input should be a pandas DataFrame")



def plot_feature(df, col):
    sns.distplot(df.loc[df.is_attributed == 0, col])
    sns.distplot(df.loc[df.is_attributed == 1, col])
    plt.show()

    
def time_details(df):
    df['epoch_time'] = ((pd.to_datetime(df['click_time']) - pd.to_datetime("2017-11-06 14:00:00"))).astype(np.int64)//10**9
    df['seconds'] = (df['epoch_time'] % 60).astype(np.uint8)
    df['epoch_minute'] = (df['epoch_time'] // 60).astype(np.uint32)
    df['minutes'] = (df['epoch_minute'] % 60).astype(np.uint8)
    
    #del df['click_time']
    return df


def load_if_saved(feature_generator):
    def wrapper(*args, **kwargs):
        tr_filename = kwargs.get('tr_filename', None)
        val_filename = kwargs.get('val_filename', None)
        rewrite = kwargs.get('rewrite', False) 
        if ((os.path.exists(tr_filename)) and (os.path.exists(val_filename)) and not(rewrite)):
            with open(tr_filename, "rb") as f:
                tr_data = pickle.load(f)
            with open(val_filename, "rb") as f:
                val_data = pickle.load(f)
        else:
            tr_data, val_data = feature_generator(*args, **kwargs)
            with open(tr_filename, "wb") as f:
                pickle.dump(tr_data, f)
            with open(val_filename, "wb") as f:
                pickle.dump(val_data, f)
        return tr_data, val_data
    return wrapper


@load_if_saved
def get_expanding_mean(tr, val, cols, target, tr_filename="../output/tr_tmp.pkl",  
                     val_filename="../output/val_tmp.pkl", seed=786, rewrite=False):
    col_name = "_".join(cols) + '_expmean'
    all_cols  = cols + [target]
    tr[col_name] = tr[all_cols].groupby(cols)[target].expanding(min_periods=1).mean().shift().fillna(-1).reset_index(level=0, drop=True)
    
    exp_mean = TargetEncoder(cols=cols,  targetcol=target, func='mean') 
    exp_mean.fit(tr[all_cols])
    val[col_name] = exp_mean.transform(val[all_cols])
    val[col_name] = val[col_name].fillna(-1)



def run_lgb(tr, val, params, feats=None, is_develop=True, 
           save_preds = False,
           save_path_preds="../output/preds_tmp.pkl"):
    
    model = lgb.LGBMClassifier(**params)
    X_tr = tr[feats]
    y_tr = tr['is_attributed']
    X_val = val[feats]
    
    logger.info("Starting LGB run \n")
    if is_develop:
        y_val = val['is_attributed']
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', 
                      verbose=10, early_stopping_rounds=100)
        val_preds = model.predict_proba(X_val)[:,1]
        logger.info("Score with feats {} on validation set is {}".format(feats, roc_auc_score(y_val,val_preds)))
        
    else:
        model.fit(X_tr, y_tr)
        val_preds = model.predict_proba(X_val)[:,1]
    
    if save_preds:
        logger.info("Saving validation/test predictions")
        with open(save_path_preds, "wb") as f:
            pickle.dump(val_preds, f)
        
    return model, val_preds
    
    
def prepare_submission(preds, save_path = "../output/test_preds.csv"):
    sub = pd.read_csv("../input/sample_submission.csv")
    sub["is_attributed"] = preds
    sub.to_csv(save_path, index=False)
    

if __name__ == "__main__":
    
    SKIPROWS = 100 * 10**6 ###
    NROWS    = 60 * 10**6 ###
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
                 "max_depth"    : 4,
                 "subsample"    : 0.7,
                 "colsample_bytree": 0.7,
                 "reg_lambda"       : 0.1,
                 "scale_pos_weight" : 9, 
                 "num_leaves"       : 8,
                 "n_jobs"           :-1
                  }
    
    logger.info("Reading train and test")
    train = pd.read_csv("../input/train_base.csv", dtype=DTYPES, 
                        #skiprows=range(1, SKIPROWS), nrows=NROWS ###
                       )
    test = pd.read_csv("../input/test_base.csv", dtype=DTYPES)
    test["is_attributed"] = np.nan
    
    logger.info("Get time details")
    train = time_details(train)
    test = time_details(test)
    
    logger.info("Load count features")
    COUNT_COLS = ["app_count", "ip_count", "channel_count",  "os_count", "ip_device_os_count",
                 "ip_device_os_app_count", "ip_device_os_app_channel_count"]
    DTYPES2 = {
               "app_count": "uint32",
                "ip_count": "uint32",
                "channel_count": "uint32",
                "os_count": "uint32",
                "ip_device_os_count": "uint32",
                "ip_device_os_app_count": "uint32",
                "ip_device_os_app_channel_count": "uint32"
                }
    train_f2 = pd.read_csv("../output/train_featsset2.csv", usecols=COUNT_COLS, dtype=DTYPES2)
    test_f2 = pd.read_csv("../output/test_featsset2.csv", usecols=COUNT_COLS, dtype=DTYPES2)
    
    logger.info("Merge count features with base dataframe")
    train = pd.concat([train, train_f2], axis=1)
    test  = pd.concat([test, test_f2], axis=1)
    
    logger.info("Break train into tr and val")
    cond = (train.dayofweek == 3) & (train.hourofday.isin([4,5,9,10,13,14]))
    cond2 = ((train.dayofweek == 3) & (train.hourofday < 4)) | (train.dayofweek < 3)
    
    tr = train.loc[cond2].reset_index(drop=True) ###
    #tr = train.loc[~cond].reset_index(drop=True) ###
    val = train.loc[cond].reset_index(drop=True)
    y_val = val["is_attributed"]
    
    logger.info("Shape of train and test is {} and {}".format(train.shape, test.shape))
    logger.info("Shape of tr and val is {} and {}".format(tr.shape, val.shape))
    
    base_feats = ['ip', 'app', 'device', 'os', 'channel', 'hourofday', 'minutes'] + COUNT_COLS

    logger.info("Running model with base feats")
    model, _ = run_lgb(tr, val, LGB_PARAMS, feats=base_feats, is_develop=True, save_preds=False)
    test_preds = model.predict_proba(test[base_feats])[:, 1]
    prepare_submission(test_preds, save_path="basefeats_withcounts1_test_preds.csv")
    
    logger.info("Generate train and test mean features")
    
    base_score = roc_auc_score(val["is_attributed"], model.predict_proba(val[base_feats])[:, 1])
    feats2 = []
    best_score = base_score
    
    for col in ['app', 'channel', 'os', 'device', 'ip', 'ip_device_os']:
        logger.info("Processing feature: {}".format(col))
        
        logger.info("Gnerating feature: {} for tr/val set".format(count_col))
        col_name = "_".join(col) + "_expmean"
        get_expanding_mean(tr, val, [col], "is_attributed", 
                           tr_filename=os.path.join(OUT_PATH, "tr_{}.pkl".format(col_name)),  
                           val_filename=os.path.join(OUT_PATH, "val_{}.pkl".format(col_name)), 
                           seed=786, rewrite=False)
        
        logger.info("Gnerating feature: {} for train/test set".format(count_col))
        get_expanding_mean(train, test, [col], "is_attributed", 
                           tr_filename=os.path.join(OUT_PATH, "train_{}.pkl".format(col_name)),  
                           val_filename=os.path.join(OUT_PATH, "test_{}.pkl".format(col_name)), 
                           seed=786, rewrite=False)

        
        feats2.append(col_name)
    
    feats = base_feats + feats2
        
    logger.info("Running LGB for develop set with feats {}".format(feats))
    model, val_preds = run_lgb(tr, val, LGB_PARAMS, feats=feats, is_develop=True, save_preds=False)
    score = roc_auc_score(y_val, val_preds)
        
    logger.info("Running LGB for train/test set with feats {}".format(feats))
    model, test_preds = run_lgb(train, test, LGB_PARAMS, feats=feats, is_develop=False, save_preds=False)
    prepare_submission(test_preds, save_path="../output/LGBv6_feats{}_test_preds.csv".format(len(feats)))

    logger.info("Saving train and test features")  
    train[feats2].to_csv("../output/train_featsset3.csv", index=False)
    
    test[feats2].to_csv("../output/test_featsset3.csv", index=False)
    
    

