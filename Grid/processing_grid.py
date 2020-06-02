# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:30:46 2020

@author: felip_001
"""

from grid import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from time import time
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import coord_transform as ct
import util

#%% OPTIONAL: Load processed data

folder = r'C:\Users\felip_001\Downloads\Boriette\\'
subf = r'ProcessedData//'
hta = pd.read_csv(folder + subf + 'MVLines_full.csv', engine='python', index_col=0)
ps = pd.read_csv(folder + subf + 'SS_full.csv', engine='python', index_col=0)
bt = pd.read_csv(folder + subf + 'MVLV_full.csv', engine='python', index_col=0)
fnodes = pd.read_csv(folder + subf + 'Nodes_full.csv', engine='python', index_col=0)

hta.ShapeGPS = hta.ShapeGPS.apply(eval)


# TODO: Load IRIS polygons

#%% Compute independent feeders and open plot to define open/closed lines
if not ('Connected' in hta.columns):
    hta['Connected'] = True
# Main node
ps0 = 'Boriette'
n0 = ps.node[ps0]

on_off_lines(hta, n0, ps=ps, bt=bt)

#%% Save data!
# Reduced data w/o non connected
htared = hta[~hta.Feeder.isna()]
ns = unique_nodes(htared)
fnodesred = fnodes.loc[ns]
btred = bt[bt.node.isin(fnodesred.index)]
psred = ps[ps.node.isin(fnodesred.index)]

util.create_folder(folder + r'ProcessedData')
print('Saving reduced data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(htared), len(btred), len(ns)))
btred.to_csv(folder + r'ProcessedData\\' +  'MVLV.csv')
psred.to_csv(folder + r'ProcessedData\\' +  'SS.csv')
htared.to_csv(folder + r'ProcessedData\\' +  'MVLines.csv')
fnodesred.to_csv(folder + r'ProcessedData\\' +  'Nodes.csv')

print('Saving Full data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(hta), len(bt), len(unique_nodes(hta))))
bt.to_csv(folder + r'ProcessedData\\' +  'MVLV_full.csv')
ps.to_csv(folder + r'ProcessedData\\' +  'SS_full.csv')
hta.to_csv(folder + r'ProcessedData\\' +  'MVLines_full.csv')
fnodes.to_csv(folder + r'ProcessedData\\' +  'Nodes_full.csv')