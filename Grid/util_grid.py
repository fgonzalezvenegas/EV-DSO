# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 08:26:50 2020

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.control as ppc

def yearly_profile(df, step=30, day_ini=0, nweeks=None, ndays=None):
    """
    """
    if not (nweeks is None):
        step = 60 / (df.shape[0]/(7 * nweeks * 24))
    elif not (ndays is None):
        step = 60 / (df.shape[0]/(ndays * 24))
    mult = int(np.ceil(375/(df.shape[0]/(24*60/step))))
    
    idi = day_ini * 24 * int(60/step)
    ide = idi + 365 * 24 * int(60/step)
    cols = df.columns
    
    return pd.DataFrame(np.tile(df.values, (mult,1)), columns=cols).iloc[idi:ide,:]
  
