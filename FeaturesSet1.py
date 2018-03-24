import pandas as pd
import numpy as  np

from TargetEncoderv2 import TargetEncoder
from FeatureSelector import FeatureSelector

from sklearn.metrics import *
from sklearn.model_selection import *

import lightgbm as lgb


def cvFeatureGeneration(df, folds=10, cols=None, targetcol='is_attributed', func='mean', cname=None):
    cvlist = StratifiedKFold(folds, random_state=1).split(df, df[targetcol])
    enc = TargetEncoder(cols=cols, targetcol=targetcol, func=func, cname=cname, add_to_orig=False)
    df[cname] = cross_val_predict(enc, df, df[targetcol], cv=cvlist, method='transform', verbose=1, pre_dispatch=None)
    return df

def testFeatureGeneration(train, test, cols=None, targetcol='is_attributed', func='mean', cname=None):
    enc = TargetEncoder(cols=cols, targetcol=targetcol, func=func, cname=cname, add_to_orig=False)
    test[cname] = enc.fit(train).transform(test)
    return test

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
    
    #Encoding strategy would be to generate features for train using cross-validation and generate for test using all train data
    
    #First read train and get all train features
    train = pd.read_csv("../input/train.csv", skiprows = list(range(1,180000000)))
    
    #Get hour information
    train["click_time"] = pd.to_datetime(train["click_time"])
    train["hourofday"] = train["click_time"].dt.hour
    train["minuteofhour"] = train["click_time"].dt.minute
    CVFOLDS = 20
    for ftype in ['count', 'mean']:
        for col in ['ip', 'app', 'device', 'os', 'channel', 'hour', ['app', 'device'], ['device', 'os'], 
                    ['channel', 'hour'], ['app', 'channel'], ['ip', 'hour'], ['app', 'hour'], ['device', 'channel'],
                    ['channel', '']]:
            if isinstance(col, list):
                cols = col
                col_name = ''
                col_name = [col_name + '_' + c for c in col ]
                col_name = col_name + "_" + ftype
            elif isinstance(col, str):
                cols = [col]
                col_name = col + "_" + ftype
            
            train = cvFeatureGeneration(train, folds=CVFOLDS, cols=cols, cname=col_name)
            
    train.to_csv("../input/train_featureset1.csv", index=False)
    
    del train
    
    #Read train again
    train = pd.read_csv("../input/train.csv", skiprows = list(range(1,180000000)))
    test = pd.read_csv("../input/test.csv", skiprows = list(range(1,17000000)))
    for ftype in ['count', 'mean']:
        for col in ['ip', 'app', 'device', 'os', 'channel']:
            col_name = col + "_" + ftype
            test = testFeatureGeneration(train, test, cols=[col], cname=col_name)    
            
    test.to_csv("../input/test_featureset1.csv", index=False)
    
    























































