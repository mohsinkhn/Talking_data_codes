#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:18:40 2017

@author: Mohsin
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, thresh=0, func=np.mean, add_to_orig=False):
        self.cols = cols
        self.thresh = thresh
        self.func = func
        self.add_to_orig = add_to_orig
    
    #@numba.jit        
    def fit(self, X, y):
        self.prior = self.func(y)
        self._dict = {}
        for col in self.cols:
            if isinstance(col, (list, tuple)):
                print('here')
                tmp_df = X.loc[: ,col]
                col = tuple(col)
            else:
                tmp_df = X.loc[: ,[col]]
            tmp_df['y'] = y
            print(tmp_df.columns)
            #tmp_df = pd.DataFrame({'eval_col':X[col].values, 'y':y})
            if isinstance(col, (list, tuple)):
                print('here')
                col = tuple(col)
            self._dict[col] = tmp_df.groupby(col)['y'].apply(lambda x: 
                                self.func(x) if len(x) >= self.thresh  else self.prior).to_dict()
                                
            del tmp_df
        return self
    #@numba.jit
    def transform(self, X, y=None):
        X_transformed = []
        for col in self.cols:
            
            if isinstance(col, (list, tuple)):
                tmp_df = X.loc[:, col]
                enc = tmp_df[col].apply(lambda x: self._dict[tuple(col)][tuple(x)]
                                                                     if tuple(x) in self._dict[tuple(col)]
                                                                     else self.prior, axis=1).values
            else:
                tmp_df = X.loc[:, [col]]
                enc = tmp_df[col].apply(lambda x: self._dict[col][x]
                                                                     if x in self._dict[col]
                                                                     else self.prior).values
            del tmp_df
            X_transformed.append(enc)
        
        X_transformed = np.vstack(X_transformed).T
        
        if self.add_to_orig:
            return np.concatenate((X.values, X_transformed), axis=1)
            
        else:
            return X_transformed