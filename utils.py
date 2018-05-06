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


def load_if_saved_numpy(feature_generator):
    def wrapper(*args, **kwargs):
        tr_filename = kwargs.get('tr_filename', None)
        val_filename = kwargs.get('val_filename', None)
        rewrite = kwargs.get('rewrite', False) 
        if ((os.path.exists(tr_filename)) and (os.path.exists(val_filename)) and not(rewrite)):
            tr_data = np.load(tr_filename)
            val_data = np.load(val_filename)

        else:
            tr_data, val_data = feature_generator(*args, **kwargs)
            np.save(tr_filename, tr_data )
            np.save(val_filename, val_data)
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
    unq_cnt.fit(pd.concat([tr[all_cols], val[all_cols]]))
    tr[col_name] = unq_cnt.transform(tr[all_cols])
    val[col_name] = unq_cnt.transform(val[all_cols])
    
    tr[col_name] = tr[col_name].fillna(0)
    val[col_name] = val[col_name].fillna(0)
    
    return tr[col_name].values.astype(np.int32), val[col_name].values.astype(np.int32)


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
    
    return tr[col_name].values.astype(np.float32), val[col_name].values.astype(np.float32)


@load_if_saved_numpy
def get_expanding_count(tr, val, cols, target, tr_filename="../output/tr_tmp.pkl",  
                     val_filename="../output/val_tmp.pkl", seed=786, rewrite=False):
    col_name = "_".join(cols) + '_expcount'
    all_cols  = cols + [target]
    tr[col_name] = tr[all_cols].groupby(cols)[target].expanding(min_periods=1).count().shift().fillna(-1).reset_index(level=0, drop=True)
    
    exp_mean = TargetEncoder(cols=cols,  targetcol=target, func='count') 
    exp_mean.fit(tr[all_cols])
    val[col_name] = exp_mean.transform(val[all_cols])
    val[col_name] = val[col_name].fillna(-1)
    
    return tr[col_name].values.astype(np.float32), val[col_name].values.astype(np.float32)


@load_if_saved_numpy
def get_std_feature(tr, val, cols, target, tr_filename="../output/tr_tmp.npy",  
                     val_filename="../output/val_tmp.npy", seed=786, rewrite=False):
    all_cols = cols + [target]
    
    var_enc = TargetEncoder(cols=cols,  targetcol=target, func='std')
    var_enc.fit(pd.concat([tr[all_cols], val[all_cols]]))
    tr_data = var_enc.transform(tr[all_cols])
    val_data = var_enc.transform(val[all_cols])
    
    return tr_data, val_data


@load_if_saved_numpy
def get_next_click(tr, val, cols, shift=-1, target='epoch_time', tr_filename="../output/tr_tmp.npy",  
                     val_filename="../output/val_tmp.npy", seed=786, rewrite=False):
    tr = tr.copy()
    val = val.copy()
    all_cols = cols + [target]
    tr['next_click'] = tr[all_cols].groupby(cols)[target].shift(shift) - tr['epoch_time']
    val['next_click'] = val[all_cols].groupby(cols)[target].shift(shift) - val['epoch_time']
    
    return tr['next_click'].fillna(-1).values, val['next_click'].fillna(-1).values


@load_if_saved_numpy
def get_prev_click(tr, val, cols, shift=1, target='epoch_time', tr_filename="../output/tr_tmp.npy",  
                     val_filename="../output/val_tmp.npy", seed=786, rewrite=False):
    all_cols = cols + [target]
    tr = tr.copy()
    val = val.copy()
    tr['prev_click'] = tr['epoch_time'] - tr[all_cols].groupby(cols)[target].shift(shift) 
    val['prev_click'] = val['epoch_time'] - val[all_cols].groupby(cols)[target].shift(shift) 
    
    return tr['prev_click'].fillna(-1).values, val['prev_click'].fillna(-1).values

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


def load_unq_features(tr, val, train, test, logger, out_path="../output/", rewrite=False):
    feats2 = []
    for cols, target in zip([['ip'], ['ip'], ['app'], ['app'], ['ip'], ['channel'], ['ip_device_os', 'dayofweek'], ['ip', 'os'], ['ip_device_os']],
                            ['app', 'channel', 'ip', 'os', 'ip_device_os', 'app', 'hourofday', 'app', 'dayofweek']):
        
        col_name = "_".join(cols) + "_unq_" + target
        logger.info("Gnerating feature: {} for tr/val set".format(col_name))
        
        tr[col_name], val[col_name] = get_unq_feature(tr, val, cols, target, 
                           tr_filename=os.path.join(out_path, "tr_{}.pkl".format(col_name)),  
                           val_filename=os.path.join(out_path, "val_{}.pkl".format(col_name)), 
                           seed=786, rewrite=rewrite)
        tr[col_name], val[col_name] = tr[col_name].astype(np.int32), val[col_name].astype(np.int32)

        logger.info("Gnerating feature: {} for train/test set".format(col_name))
        train[col_name], test[col_name] = get_unq_feature(train, test, cols, target, 
                           tr_filename=os.path.join(out_path, "train_{}.pkl".format(col_name)),  
                           val_filename=os.path.join(out_path, "test_{}.pkl".format(col_name)), 
                           seed=786, rewrite=rewrite)
        train[col_name], test[col_name] = train[col_name].astype(np.int32), test[col_name].astype(np.int32)
        
        feats2.append(col_name)
    return tr, val, train, test, feats2


def load_count_features(tr, val, train, test, logger, out_path="../output/", seed=786):
    feats2 = []
    for col in ['app', 'channel', 'os', 'device', 'ip', 'ip_device_os', 'ip_device_os_app', 'ip_device_os_app_channel']:
        logger.info("Processing feature: {}".format(col))
        
        col_name = "_".join([col]) + "_count"
        logger.info("Gnerating feature: {} for tr/val set".format(col_name))
        
        tr[col_name], val[col_name] = get_count_feature(tr, val, [col], "is_attributed", 
                           tr_filename=os.path.join(out_path, "tr_{}_{}.pkl".format(col_name, seed)),  
                           val_filename=os.path.join(out_path, "val_{}_{}.pkl".format(col_name, seed)), 
                           seed=786, rewrite=False)
        tr[col_name], val[col_name] = tr[col_name].astype(np.int32), val[col_name].astype(np.int32)

        logger.info("Gnerating feature: {} for train/test set".format(col_name))
        train[col_name], test[col_name] = get_count_feature(train, test, [col], "is_attributed", 
                           tr_filename=os.path.join(out_path, "train_{}_{}.pkl".format(col_name, seed)),  
                           val_filename=os.path.join(out_path, "test_{}_{}.pkl".format(col_name, seed)), 
                           seed=786, rewrite=False)
        feats2.append(col_name)
        train[col_name], test[col_name] = train[col_name].astype(np.int32), test[col_name].astype(np.int32)
    return tr, val, train, test, feats2


def load_expmean_features(tr, val, train, test, logger, out_path="../output/", rewrite=False):
    feats2 = []
    for col in ['app', 'channel', 'os', 'device', 'ip', 'ip_device_os']:
        logger.info("Processing feature: {}".format(col))
        
        col_name = "_".join([col]) + "_expmean"
        logger.info("Gnerating feature: {} for tr/val set".format(col_name))
        
        tr[col_name], val[col_name] = get_expanding_mean(tr, val, [col], "is_attributed", 
                           tr_filename=os.path.join(out_path, "tr_{}.pkl".format(col_name)),  
                           val_filename=os.path.join(out_path, "val_{}.pkl".format(col_name)), 
                           seed=786, rewrite=rewrite)
        tr[col_name], val[col_name] = tr[col_name].astype(np.float32), val[col_name].astype(np.float32)

        logger.info("Gnerating feature: {} for train/test set".format(col_name))
        train[col_name], test[col_name] = get_expanding_mean(train, test, [col], "is_attributed", 
                           tr_filename=os.path.join(out_path, "train_{}.pkl".format(col_name)),  
                           val_filename=os.path.join(out_path, "test_{}.pkl".format(col_name)), 
                           seed=786, rewrite=rewrite)
        train[col_name], test[col_name] = train[col_name].astype(np.float32), test[col_name].astype(np.float32)
        
        feats2.append(col_name)
    return tr, val, train, test, feats2
