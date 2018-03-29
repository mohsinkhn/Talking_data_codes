import pandas as pd
import dask.dataframe as dd
import dask.array as darray
import numpy as  np

from TargetEncoderv3 import TargetEncoder
from FeatureSelector import FeatureSelector

from sklearn.metrics import *
from sklearn.model_selection import *

import lightgbm as lgb
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('featureset1.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

def cross_val_predict_dask(model, X, y, cvlist, method='predict', verbose=1):
    y_new = pd.DataFrame(index=range(len(y)))
    for i, (tr_index, val_index) in enumerate(cvlist):
        if verbose:
            print("Working on fold {}".format(i))
        X_tr = X.loc[tr_index,:]
        model.fit(X_tr)
        vals = getattr(model, method)
        y_new.loc[val_index] = vals(X_tr)
        
    return dd.from_pandas(y_new, npartitions=3)
        
        
def cvFeatureGeneration(df, cvlist=None, cols=None, targetcol='is_attributed', func='mean', cname=None):
    #cvlist = StratifiedKFold(folds, random_state=1).split(df, df[targetcol])
    enc = TargetEncoder(cols=cols, targetcol=targetcol, func=func, cname=cname, add_to_orig=False)
    df[cname] = cross_val_predict_dask(enc, df, df[targetcol], cvlist, method='transform', verbose=1)
    return df

def testFeatureGeneration(train, test, cols=None, targetcol='is_attributed', func='mean', cname=None):
    enc = TargetEncoder(cols=cols, targetcol=targetcol, func=func, cname=cname, add_to_orig=False)
    test[cname] = enc.fit(train).transform(test)
    return test

def getTimeFeats(df):
    df["click_time"] = dd.to_datetime(df["click_time"])
    df["hourofday"] = df["click_time"].dt.hour.astype(np.uint16)
    df["dayofweek"] = df["click_time"].dt.weekday.astype(np.uint8)
    del df['click_time']
    return df

if __name__ == "__main__":
    dtypes = dtype = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }
    
    SKIPROWS = list(range(1,180000000))
    #Encoding strategy would be to generate features for train using cross-validation and generate for test using all train data
    
    #First read train and get all train features
    train = pd.read_csv("../input/train.csv", skiprows = SKIPROWS)
    train = dd.from_pandas(train, npartitions=3)
    logger.info("Reading train")
    #train = dd.read_csv("../input/train.csv") ##TO RUN ON FULL DATA
    #del train['attributed_time']
    #Get hour information
    logger.info("Generating time features for train")
    train = getTimeFeats(train)
    
    #ip hour channel day var
    #col_name = "ip_hour_channel_daynunq"
    #logger.info("processing feature {} for train".format(col_name))
    #enc = TargetEncoder(cols=['ip','hourofday','channel'], targetcol='dayofweek', func='nunique', cname=col_name, add_to_orig=False)
    #train[col_name] = enc.fit_transform(train)
    #logger.info("Stats for generated feature for train are {}".format(train[col_name].mean().compute()))
    
    #ip hour day count
    col_name = "ip_hour_day_count"
    logger.info("processing feature {} for train".format(col_name))
    enc = TargetEncoder(cols=['ip','hourofday','dayofweek'], targetcol='channel', func='count', cname=col_name, add_to_orig=False)
    train[col_name] = enc.fit_transform(train).compute()
    logger.info("Stats for generated feature for train are {}".format(train[col_name].mean().compute()))
    
    #ip hour day channel count
    #col_name = "ip_day_channel_hournunq"
    #logger.info("processing feature {} for train".format(col_name))
    #enc = TargetEncoder(cols=['ip','dayofweek', 'channel'], targetcol='hourofday', func='nunique', cname=col_name, add_to_orig=False)
    #train[col_name] = enc.fit_transform(train)
    #logger.info("Stats for generated feature for train are {}".format(train[col_name].mean().compute()))
    
    #Add priors and count for getting confidence on them
    CVFOLDS = list(StratifiedKFold(20).split(train, train.is_attributed)) ##Decrease to 10 folds
    
    for ftype in ['count', 'mean']:
        for col in ['ip', 'app', 'device', 'os', 'channel', 'hourofday', ['ip','app'],
                   ['app', 'device'], ['app', 'os'], ['app', 'channel'], ['app', 'hourofday'],
                   ['device', 'hourofday'], ['os', 'hourofday'], ['channel', 'hourofday'],
                   ['channel', 'os'], ['os', 'hourofday'], ['channel', 'hourofday']]:
            
            if isinstance(col, list):
                cols = col
                #col_name = ''
                col_name = '_'.join(cols)
                col_name = col_name + "_" + ftype
            elif isinstance(col, str):
                cols = [col]
                col_name = col + "_" + ftype
            logger.info("processing feature {} for train".format(col_name))
            train = cvFeatureGeneration(train, cvlist=CVFOLDS, cols=cols, func=ftype, cname=col_name).compute()
            logger.info("Stats for generated feature for train are {}".format(train[col_name].mean().compute()))
            
    logger.info("Writing out train file")
    train.to_csv("../input/train_featureset1.csv", index=False, compression='gzip')
    
    #del train
    
    #Read train again
    #train = dd.read_csv("../input/train.csv", skiprows = SKIPROWS)
    #train = dd.read_csv("../input/train.csv")###TO RUN ON FULL DATA
    #test = dd.read_csv("../input/test.csv", skiprows = list(range(1,10000000)))
    test = dd.read_csv("../input/test.csv")###TO RUN ON FULL DATA
    
    #train = getTimeFeats(train)
    test = getTimeFeats(test)
    
    #ip hour channel day var
    #col_name = "ip_hour_channel_daynunq"
    #logger.info("processing feature {} for train".format(col_name))
    #enc = TargetEncoder(cols=['ip','hourofday','channel'], targetcol='dayofweek', func='nunique', cname=col_name, add_to_orig=False)
    #test[col_name] = enc.fit_transform(test)
    #logger.info("Stats for generated feature for train are {}".format(test[col_name].mean().compute()))
    
    #ip hour day count
    col_name = "ip_hour_day_count"
    logger.info("processing feature {} for train".format(col_name))
    enc = TargetEncoder(cols=['ip','hourofday','dayofweek'], targetcol='channel', func='count', cname=col_name, add_to_orig=False)
    test[col_name] = enc.fit_transform(test).compute()
    logger.info("Stats for generated feature for train are {}".format(test[col_name].mean().compute()))
    
    #ip hour day channel count
    #col_name = "ip_day_channel_hournunq"
    #logger.info("processing feature {} for train".format(col_name))
    #enc = TargetEncoder(cols=['ip','dayofweek', 'channel'], targetcol='hourofday', func='nunique', cname=col_name, add_to_orig=False)
    #test[col_name] = enc.fit_transform(test)
    #logger.info("Stats for generated feature for train are {}".format(test[col_name].mean().compute()))
    
    
    for ftype in ['count', 'mean']:
        for col in ['ip', 'app', 'device', 'os', 'channel', 'hourofday', ['ip','app'],
                   ['app', 'device'], ['app', 'os'], ['app', 'channel'], ['app', 'hourofday'],
                   ['device', 'hourofday'], ['os', 'hourofday'], ['channel', 'hourofday'],
                   ['channel', 'os'], ['os', 'hourofday'], ['channel', 'hourofday']]:
            if isinstance(col, list):
                cols = col
                #col_name = ''
                col_name = '_'.join(cols)
                col_name = col_name + "_" + ftype
            elif isinstance(col, str):
                cols = [col]
                col_name = col + "_" + ftype
            logger.info("processing feature {} for test".format(col_name))
            test = testFeatureGeneration(train, test, cols=cols, func=ftype, cname=col_name).compute()
            logger.info("Stats for generated feature for test are {}".format(test[col_name].mean().compute()))
            
    logger.info("Writing out test file")
    test.to_csv("../input/test_featureset1.csv", index=False, compression='gzip')
    
    























































