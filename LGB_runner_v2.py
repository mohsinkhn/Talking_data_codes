
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as  np

from TargetEncoderv3 import TargetEncoder
from FeatureSelector import FeatureSelector

from sklearn.metrics import *
from sklearn.model_selection import *

import lightgbm as lgb
from utils import outoffold_crossvalidator, shuffle_crossvalidator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('LGB_featureset1.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
#import sys
#sys.stdout = logger



# In[2]:


dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
    }

logger.info("Reading train file")
train = pd.read_csv("../input/train_featureset1_v2.csv", dtype=dtypes, compression='gzip')

logger.info("Reading test file")
test = pd.read_csv("../input/test_featureset1_v2.csv", dtype=dtypes, compression='gzip')


# In[3]:


print(train.shape, test.shape)
print(train.describe())
print(train.describe())

#print(test.head())
#print(test.tail())


# In[4]:


print(train.dtypes)
print(test.dtypes)


# In[6]:


print(train.isnull().sum()/len(train))
print(test.isnull().sum()/len(test))


# In[7]:


logger.info("Generating train and validation sets")
val_idx = np.array(train.loc[(train.dayofweek == 3) & (train.hourofday.isin([4,5,14]))].index)
tr_idx  = np.array(train.loc[~((train.dayofweek == 3) & (train.hourofday.isin([4,5,14])))].index)
cvlist1 = [[tr_idx, val_idx]]

model = lgb.LGBMClassifier(num_leaves=7, max_depth=3, n_jobs=-1, n_estimators=1500, subsample=1.0, 
                           colsample_bytree=0.7, min_child_samples=1000, scale_pos_weigt=100,
                       verbose=10)


# In[8]:


features= ['ip', 'app','device','os','channel', 'ip_hour_day_count','ip_count',
           'app_count','device_count','os_count','channel_count','hourofday_count',
           'app_device_count','app_os_count','app_channel_count','app_hourofday_count',
           'device_hourofday_count','os_hourofday_count','channel_hourofday_count',
           'channel_os_count','app_mean','device_mean','os_mean','channel_mean',
           'app_device_mean','app_os_mean','app_channel_mean','app_hourofday_mean',
           'device_hourofday_mean','os_hourofday_mean','channel_hourofday_mean','channel_os_mean']

X = train[features]
y = train.is_attributed
print(X.loc[cvlist1[0][0]].shape, X.loc[cvlist1[0][1]].shape)
#print(len(cvlist1[0][0]))
logger.info("check model performance on validation set")
val_preds, y_val, _ = shuffle_crossvalidator(model, X, y, cvlist=cvlist1)
logger.info("Validation score is {}".format(roc_auc_score(y_val, val_preds)))


# In[9]:
model = lgb.LGBMClassifier(num_leaves=7, max_depth=3, n_jobs=-1, n_estimators=900, subsample=1.0, 
                           colsample_bytree=0.7, min_child_samples=1000, scale_pos_weigt=100,
                       verbose=10)


logger.info("fit model on all data and predict on test")
X_test = test[features]
test_preds = model.fit(train[features], train.is_attributed,
                      eval_set=[(train[features], train.is_attributed,)],
                       eval_metric='auc').predict_proba(X_test)[:,1]


# In[15]:

from scipy.stats import describe
print(describe(val_preds))
print(describe(test_preds))
#import matplotlib.pyplot as plt
#import seaborn as sns

#sns.distplot(val_preds)
#sns.distplot(test_preds)
#plt.savefig("./distributions.png")


# In[11]:


print(val_preds[:5])
print(test_preds[:5])


# In[12]:


logger.info("Write out submission")
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
sub['is_attributed'] = test_preds
logger.info(sub['is_attributed'].describe())
logger.info(sub.head())

sub.to_csv("../input/first_submission.csv", index=False)
    

