######################################################################
# Script to generate count features for different column combinations using booth train and test data
# Author: Mohsin Hasan Khan
######################################################################

import pandas as pd
import numpy as np
import pickle
import os
from collections import Counter

import multiprocessing as mp

import itertools as IT

from functools import reduce
import gc
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('count_feature_generation.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def cntit(chunk, cols):
    """
    Given a chunk return a Counter object over tuples of given cols
    """
    return Counter(list(chunk[cols].itertuples(index=False, name=None)))
    
def gen_args(chunk, cols):
    """
    Helper function for Pool.starmap() to generate a iterable of arguments
    """
    for c in chunk:
        yield (c, cols)
        