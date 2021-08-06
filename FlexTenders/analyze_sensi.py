# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:18:11 2021
Analyze sensi - iterations on Energy Policy article
@author: U546416
"""
  
import numpy as np
from matplotlib import pyplot as plt
#import EVmodel
#import scipy.stats as stats
#import time
#import util
#import flex_payment_funcs
import pandas as pd
import os

#%% Load data
sets = ['Company', 'Commuter_HP', 'Commuter_LP']
sstr = [s.replace('_',' ') for s in sets]
folder = r'C:\Users\u546416\AnacondaProjects\EV-DSO\FlexTenders\EnergyPolicy\paretto\\'
av_w  = ['0_24', '17_20']
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
pay = ['Payments_Avg', 'Payments_perc_l', 'Payments_perc_h']
und = ['UnderDel_Avg', 'UnderDel_perc_l', 'UnderDel_perc_h']

# read all files and and put them in one big DF
fs = [f for f in os.listdir(folder) if f.endswith('.csv')]

colsidx = ['VxG','Fleet', 'AvWindow', 'nevs', 'nactivation', 'service_duration', 'confidence',
               'penalty_threshold', 'penalties']
if 'full_data.csv' in fs:
    res = pd.read_csv(folder + 'full_data.csv', engine='python')
    if 'sensi_param' in res.columns:
        colsidx += ['sensi_param', 'sensi_val']
    res.set_index(colsidx, inplace=True)
else:
    res = pd.DataFrame()
    for f in fs:
        data = pd.read_csv(folder + f, engine='python')
        avw = '0_24' if '0_24' in f else '17_20'
        fleet = 'Company' if 'Company' in f else '' + 'Commuter_HP' if 'HP' in f else 'Commuter_LP' if 'LP' in f else 'Commuter_EN'
        data['AvWindow'] = avw
        data['Fleet'] = fleet
        res = pd.concat([res, data], ignore_index=True)
    if 'sensi_param' in res.columns:
        colsidx += ['sensi_param', 'sensi_val']
    res.set_index(colsidx, inplace=True, drop=True)
    res.to_csv(folder + 'full_data.csv')

#%%
# idx = 'VxG', 'Fleet', 'AvWindow', 'nevs', 'nactivation', 'service_duration', 'confidence', 'penalty_threshold', 'penalties', 'sensi_param', 'sensi_val']

v='v2g'
f=['Company', 'Commuter_LP', 'Commuter_HP']
aw = ['17_20','0_24']
nevs=31
nac = 10
sd = 30
conf = 0.9
pth = 0.6
pp = 0

base_eve = res.Payments_Avg[v,:,aw[0],nevs,nac,sd,conf,pth,pp,'batt_size',50]
base_fdw = res.Payments_Avg[v,:,aw[1],nevs,nac,sd,conf,pth,pp,'batt_size',50]

# Computing deltas
# wrt battery size
dbatt_eve = (res.Payments_Avg[v,:,aw[0],nevs,nac,sd,conf,pth,pp,'batt_size',:]-base_eve)
dbatt_fdw = (res.Payments_Avg[v,:,aw[1],nevs,nac,sd,conf,pth,pp,'batt_size',:]-base_fdw) 

# wrt charging power
dbatt_eve = (res.Payments_Avg[v,:,aw[0],nevs,nac,sd,conf,pth,pp,'charging_power',:]-base_eve)
dbatt_fdw = (res.Payments_Avg[v,:,aw[1],nevs,nac,sd,conf,pth,pp,'charging_power',:]-base_fdw)


#%%
b0 = res.xs('batt_size', level='sensi_param').xs(50,level='sensi_val')

dxs =  {'batt_size': [25,75],
        'charging_power': [11,3.5],
        'target_soc' : [1],
        'tou_ini': [23]}

delta = []
delta_pc = []
nitems = []
items = []
for s, vs in dxs.items():
    for v in vs:
#        print(s, v, res.xs(s, level='sensi_param').xs(v,level='sensi_val').shape)
        delta.append(res.xs(s, level='sensi_param').xs(v,level='sensi_val')-b0)
        delta_pc.append((res.xs(s, level='sensi_param').xs(v,level='sensi_val')-b0)/b0)
        nitems.append(delta[-1].shape[0])
        items += [(s, v)]
delta = pd.concat(delta)  
delta['sensi_param'] = [items[i][0] for i, nit in enumerate(nitems) for j in range(nit)]
delta['sensi_val'] =   [items[i][1] for i, nit in enumerate(nitems) for j in range(nit)]
delta_pc = pd.concat(delta_pc)  
delta_pc['sensi_param'] = [items[i][0] for i, nit in enumerate(nitems) for j in range(nit)]
delta_pc['sensi_val'] =   [items[i][1] for i, nit in enumerate(nitems) for j in range(nit)]

red = delta.xs(('v2g',31,10,30,0.9,0.6,17.5), level=('VxG', 'nevs', 'nactivation', 'service_duration', 'confidence', 'penalty_threshold', 'penalties'))
red = red.reset_index().set_index(['Fleet', 'AvWindow', 'sensi_param', 'sensi_val']).sort_index(level=('Fleet', 'AvWindow'))

red_pc = delta_pc.xs(('v2g',31,10,30,0.9,0.6,17.5), level=('VxG', 'nevs', 'nactivation', 'service_duration', 'confidence', 'penalty_threshold', 'penalties'))
red_pc = red_pc.reset_index().set_index(['Fleet', 'AvWindow', 'sensi_param', 'sensi_val']).sort_index(level=('Fleet', 'AvWindow'))
