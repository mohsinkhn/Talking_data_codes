import pandas as pd
import numpy as np

import pickle
import os

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from TargetEncoder import TargetEncoder

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
def get_count_feature(tr, val, cols, target, tr_filename="../output/tr_tmp.pkl",  
                     val_filename="../output/val_tmp.pkl", seed=786, rewrite=False):
    all_cols = cols + [target]
    
    mean_enc = TargetEncoder(cols=cols,  targetcol=target, func='count')
    mean_enc.fit(pd.concat([tr[all_cols], val[all_cols]]))
    tr_data = mean_enc.transform(tr[all_cols])
    val_data = mean_enc.transform(val[all_cols])
    
    return tr_data, val_data


@load_if_saved
def get_mean_feature(tr, val, cols, target, tr_filename="../output/tr_tmp.pkl",  
                     val_filename="../output/val_tmp.pkl", cv=5, thresh=2, seed=786, rewrite=False):
    all_cols = cols + [target]
    
    mean_enc = TargetEncoder(cols=cols,  targetcol=target, func='mean')
    cvlist = KFold(n_splits=cv, shuffle=True, random_state=seed).split(tr[target])
    
    tr_data = cross_val_predict(mean_enc, tr[all_cols], tr[target], cv=cvlist, method='transform', verbose=1)
    val_data = mean_enc.fit(tr[all_cols]).transform(val[all_cols])
    
    return tr_data, val_data

@load_if_saved
def get_unq_feature(tr, val, cols, target, tr_filename="../output/tr_tmp.pkl",  
                     val_filename="../output/val_tmp.pkl", seed=786, rewrite=False):
    col_name = "_".join(cols) + '_unq_' + target
    all_cols  = cols + [target]
    unq_cnt = TargetEncoder(cols=cols,  targetcol=target, func='nunique') 
    
    tr[col_name] = unq_cnt.fit_transform(tr[all_cols])
    val[col_name] = unq_cnt.transform(val[all_cols])
    
    tr[col_name] = tr[col_name].fillna(0)
    val[col_name] = val[col_name].fillna(0)
    
    return tr[col_name], val[col_name]


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
    
    return tr[col_name], val[col_name]


def run_lgb(tr, val, params, logger, 
           feats=None, is_develop=True, 
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

