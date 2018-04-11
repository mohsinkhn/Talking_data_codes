import numpy as np
import gc
from sklearn.metrics import roc_auc_score
import copy
from scipy.stats import gmean

#LightGBM compatible cross validator for shuffle validation splits
def shuffle_crossvalidator(model, X, y, cvlist, X_test=None, predict_test=False, 
                           scorer = roc_auc_score):
    y_trues = []
    y_preds = []
    scores = []
    y_test_preds = []
    X = np.asarray(X)
    y = np.asarray(y)
    for tr_index, val_index in cvlist:

        X_tr, y_tr = X[tr_index, :], y[tr_index]
        X_val, y_val = X[val_index, :], y[val_index]            

        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='auc', verbose=10, early_stopping_rounds=50)

        y_pred = model.predict_proba(X_val)[:,1]
        
        if predict_test:
            y_test_preds.append(model.predict_proba(X_test)[:,1])
        score = scorer(y_val, y_pred)
        scores.append(score)
        print("Score for this fold is ", score)
        y_trues.append(y_val)
        y_preds.append(y_pred)
        gc.collect()
        #break
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    if predict_test:
        y_test_preds = np.mean(y_test_preds, axis=0)
    score = scorer(y_trues, y_preds)
    print("Overall score on 10 fold CV is {}".format(score))
    
    return y_preds, y_trues, y_test_preds

#LightGBM compatible cross validator
def outoffold_crossvalidator(model, X, y, cvlist, X_test=None, predict_test=False, 
                           scorer = roc_auc_score):
    y_preds = np.zeros(y.shape)
    y_test_preds = []
    X = np.asarray(X)
    y = np.asarray(y)
    for tr_index, val_index in cvlist:
        
        X_tr, y_tr = X[tr_index, :], y[tr_index]
        X_val, y_val = X[val_index, :], y[val_index]   

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc', verbose=10, early_stopping_rounds=50)

        y_pred = model.predict_proba(X_val)[:,1]
        if predict_test:
            y_test_preds.append(model.predict_proba(X_test)[:,1])
        print("Score for this fold is ", scorer(y_val, y_pred))
        y_preds[val_index] = y_pred
        gc.collect()
        
    if predict_test:
        y_test_preds = np.mean(y_test_preds, axis=0)
    score = scorer(y, y_preds)
    print("Overall score on 10 fold CV is {}".format(score))
    
    return y_preds, y, y_test_preds

#basic time related features
def calcTimeFeats(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour
    df['minute'] = df['click_time'].dt.minute
    df['weekday'] = df['click_time'].dt.weekday
    df['hour_minute'] = df['hour'] + df['minute']/60.0
    df["hour_sine"] = np.sin(df["hour"]/24 * 2* np.pi)
    df["hour_cosine"] = np.cos(df["hour"]/24 * 2* np.pi)
    
    del df['hour_minute']
    del df['minute']
    del df['click_time']
    return df
