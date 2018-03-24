#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:19:20 2017

@author: Mohsin
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

def get_obj_cols(df):
    """Return columns with object dtypes"""
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object':
            obj_cols.append(df.columns.values[idx])

    return obj_cols


def convert_input(X):
    """if input not a dataframe convert it to one"""
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, list):
            X = pd.DataFrame(np.array(X))
        elif isinstance(X, (np.generic, np.ndarray)):
            X = pd.DataFrame(X)
        elif isinstance(X, csr_matrix):
            X = pd.SparseDataFrame(X)
        else:
            raise ValueError('Unexpected input type: %s' % (str(type(X))))

        #X = X.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """ Class to do subset of features in sklearn pipeline"""
    def __init__(self, cols=None, return_df=True, verbose=0):
        self.cols = cols
        self.return_df = return_df
        self.verbose = verbose
        
    def fit(self, X, y=None):
        #Do nothing
        return self
    
    def transform(self, X, y=None):
        #if the input dataset isn't already a dataframe, convert it to one
        X = X.copy(deep=True)
        X = convert_input(X)
        X = X.loc[:, self.cols]
        
        if self.verbose:
            print("Selecting columns are {}".format(self.cols))
        if self.return_df:
            return X
        else:
            return X.values