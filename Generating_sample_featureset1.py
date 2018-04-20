
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as  np

from TargetEncoderv2 import TargetEncoder
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


# In[2]:


def cvFeatureGeneration(df, folds=10, cols=None, targetcol='is_attributed', func='mean', cname=None):
    cvlist = StratifiedKFold(folds, random_state=1).split(df, df[targetcol])
    enc = TargetEncoder(cols=cols, targetcol=targetcol, func=func, cname=cname, add_to_orig=False)
    df[cname] = cross_val_predict(enc, df, df[targetcol], cv=cvlist, method='transform', verbose=0, pre_dispatch=None)
    return df

def testFeatureGeneration(train, test, cols=None, targetcol='is_attributed', func='mean', cname=None):
    enc = TargetEncoder(cols=cols, targetcol=targetcol, func=func, cname=cname, add_to_orig=False)
    test[cname] = enc.fit(train).transform(test)
    return test

def getTimeFeats(df):
    df["click_time"] = pd.to_datetime(df["click_time"])
    df["hourofday"] = df["click_time"].dt.hour.astype(np.uint16)
    df["dayofweek"] = df["click_time"].dt.weekday.astype(np.uint8)
    del df['click_time']
    return df


# In[3]:


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
    


# In[4]:


train = pd.read_csv("../input/train_sample.csv")


# In[5]:


print(train.head())


# In[6]:


train = getTimeFeats(train)


# In[8]:


#ip hour channel day var
col_name = "ip_hour_channel_dayvar"
logger.info("processing feature {} for train".format(col_name))
enc = TargetEncoder(cols=['ip','hourofday','channel'], targetcol='dayofweek', func='var', cname=col_name, add_to_orig=False)
train[col_name] = enc.fit_transform(train)
logger.info("Stats for generated feature for train are {}".format(train[col_name].describe()))


# In[9]:


#ip hour day count
col_name = "ip_hour_day_count"
logger.info("processing feature {} for train".format(col_name))
enc = TargetEncoder(cols=['ip','hourofday','dayofweek'], targetcol='channel', func='count', cname=col_name, add_to_orig=False)
train[col_name] = enc.fit_transform(train)
logger.info("Stats for generated feature for train are {}".format(train[col_name].describe()))


# In[10]:


#ip hour day channel count
col_name = "ip_day_channel_hourvar"
logger.info("processing feature {} for train".format(col_name))
enc = TargetEncoder(cols=['ip','dayofweek', 'channel'], targetcol='hourofday', func='var', cname=col_name, add_to_orig=False)
train[col_name] = enc.fit_transform(train)
logger.info("Stats for generated feature for train are {}".format(train[col_name].describe()))


# In[11]:


#Add priors and count for getting confidence on them
CVFOLDS = 20 ##Decrease to 10 folds

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
        train = cvFeatureGeneration(train, folds=CVFOLDS, cols=cols, func=ftype, cname=col_name)
        logger.info("Stats for generated feature for train are {}".format(train[col_name].describe()))



# In[13]:


logger.info("Writing out train file")
train.to_csv("../input/train_featureset1_v2.csv", index=False, compression='gzip')


# In[14]:


test = pd.read_csv("../input/test.csv")###TO RUN ON FULL DATA
test = getTimeFeats(test)


# In[15]:


#ip hour channel day var
col_name = "ip_hour_channel_dayvar"
logger.info("processing feature {} for train".format(col_name))
enc = TargetEncoder(cols=['ip','hourofday','channel'], targetcol='dayofweek', func='var', cname=col_name, add_to_orig=False)
test[col_name] = enc.fit_transform(test)
logger.info("Stats for generated feature for train are {}".format(test[col_name].describe()))

#ip hour day count
col_name = "ip_hour_day_count"
logger.info("processing feature {} for train".format(col_name))
enc = TargetEncoder(cols=['ip','hourofday','dayofweek'], targetcol='channel', func='count', cname=col_name, add_to_orig=False)
test[col_name] = enc.fit_transform(test)
logger.info("Stats for generated feature for train are {}".format(test[col_name].describe()))

#ip hour day channel count
col_name = "ip_day_channel_hourvar"
logger.info("processing feature {} for train".format(col_name))
enc = TargetEncoder(cols=['ip','dayofweek', 'channel'], targetcol='hourofday', func='var', cname=col_name, add_to_orig=False)
test[col_name] = enc.fit_transform(test)
logger.info("Stats for generated feature for train are {}".format(test[col_name].describe()))


# In[16]:


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
        test = testFeatureGeneration(train, test, cols=cols, func=ftype, cname=col_name) 
        logger.info("Stats for generated feature for test are {}".format(test[col_name].describe()))


# In[17]:


logger.info("Writing out test file")
test.to_csv("../input/test_featureset1_v2.csv", index=False, compression='gzip')


# In[18]:


#import matplotlib.pyplot as plt
#import seaborn as sns


# In[19]:


print(train.columns)


# In[20]:


print(test.columns)


# In[21]:


feats = ['ip_hour_channel_dayvar',
       'ip_hour_day_count', 'ip_day_channel_hourvar', 'ip_count', 'app_count',
       'device_count', 'os_count', 'channel_count', 'hourofday_count',
       'ip_app_count', 'app_device_count', 'app_os_count', 'app_channel_count',
       'app_hourofday_count', 'device_hourofday_count', 'os_hourofday_count',
       'channel_hourofday_count', 'channel_os_count', 'ip_mean', 'app_mean',
       'device_mean', 'os_mean', 'channel_mean', 'hourofday_mean',
       'ip_app_mean', 'app_device_mean', 'app_os_mean', 'app_channel_mean',
       'app_hourofday_mean', 'device_hourofday_mean', 'os_hourofday_mean',
       'channel_hourofday_mean', 'channel_os_mean']

#for col in feats:
#    plt.subplot(1,2,1)
#    sns.distplot(train.loc[train.is_attributed == 0, col].fillna(train[col].mean()))
#    sns.distplot(train.loc[train.is_attributed == 1, col].fillna(train[col].mean()))
    
#    plt.subplot(1,2,2)
#    sns.distplot(train[col].fillna(train[col].mean()))
#    sns.distplot(test[col].fillna(train[col].mean()))
#    plt.show()


# In[22]:


#ip_app_count, device_count, ip_day_channel_hourvar, ip_hour_day_count, ip_hour_channel_dayvar
print(train.loc[train.is_attributed ==1, 'ip_hour_channel_dayvar'].describe())


# In[23]:


print(train.loc[train.is_attributed ==0, 'ip_hour_channel_dayvar'].describe())


# In[24]:


print(train.loc[train.is_attributed == 0, 'ip_day_channel_hourvar'].describe())

