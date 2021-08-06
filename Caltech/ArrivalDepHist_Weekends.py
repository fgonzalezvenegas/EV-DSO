# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 00:03:22 2020

@author: U546416
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util

def random_from_cdf(cdf, bins):
    """Returns a random bin value given a cdf.
    cdf has n values, and bins n+1, delimiting initial and final boundaries for each bin
    """
    if cdf.max() > 1.0001 or cdf.min() < 0:
        raise ValueError('CDF is not a valid cumulative distribution function')
    r = np.random.rand(1)
    x = int(np.digitize(r, cdf))
    return bins[x] + np.random.rand(1) * (bins[x+1] - bins[x])

# Histogram of travel time. All trips (incl. not work) except walking 
dttime = np.array([0,5,10,15,30,45,60,90,120])
perc = np.array([21.2,22,19.3,23.9,7.2,3.6,2.2,0.7])
dh = np.array([dttime[i]-dttime[i-1] for i in range(1,len(dttime))])
x = np.array([(dttime[i]+dttime[i-1])/2 for i in range(1,len(dttime))])
val = perc/(dh/5)
plt.bar(x=x, height=val, width=dh, alpha=0.5)
cdf_time = (perc/perc.sum()).cumsum()
## Histogram of travel time. Work-trips, but all means (incl. walking)
#dttimedt = np.array([0,5,15,30,45,60,120])
#percdt = np.array([14.5,35.7,30.8,10.7,5.4,3.1])
#dhdt = np.array([dttimedt[i]-dttimedt[i-1] for i in range(1,len(dttimedt))])
#xdt = np.array([(dttimedt[i]+dttimedt[i-1])/2 for i in range(1,len(dttimedt))])
#valdt = percdt/(dhdt/5)
#plt.bar(x=xdt, height=valdt, width=dhdt, alpha=0.5)


folder_arrdep = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité'
res_arr_dep_wd = pd.read_csv(folder_arrdep + r'\EN_arrdep_wd_modifFR.csv', 
                             engine='python', index_col=0)
res_arr_dep_we = pd.read_csv(folder_arrdep + r'\EN_arrdep_we_modifFR.csv', 
                             engine='python', index_col=0)
nsessions = 10000
sessions = []
cdfdepart = res_arr_dep_we.sum(axis=0).cumsum() # departures from home
cdfarrival = res_arr_dep_we.sum(axis=1).cumsum() # departures from home
bins= np.arange(0,24.5,0.5)
for i in range(nsessions):
    while True:
        start_at_home = random_from_cdf(cdfdepart, bins)
        end_at_home = random_from_cdf(cdfarrival, bins)
        if (start_at_home<end_at_home) or (end_at_home<4):    
            traveltime = random_from_cdf(cdf_time, bins=dttime)/60
            traveltime = min(traveltime, ((end_at_home-start_at_home)%24)/3)
            arr = start_at_home + traveltime
            dep = end_at_home - traveltime
            sessions.append([arr, dep])
            break
sessions=np.array(sessions).squeeze()
h, _, _ = np.histogram2d(sessions[:,0], sessions[:,1], bins=bins)
util.plot_arr_dep_hist(h)
util.plot_arr_dep_hist(res_arr_dep_we.values)

xb = (bins[1:] + bins[:-1])/2
plt.figure()
plt.bar(xb-0.125, res_arr_dep_we.sum(axis=0), width=0.25, alpha=0.5, label='Home departure')
plt.bar(xb+0.125, h.sum(axis=1)/h.sum(), width=0.25, alpha=0.5, label='Loc arrival')
plt.bar(xb-0.125, -res_arr_dep_we.sum(axis=1), width=0.25, alpha=0.5, label='Home arrival')
plt.bar(xb+0.125, -h.sum(axis=0)/h.sum(), width=0.25, alpha=0.5, label='Loc departure')
plt.legend()
plt.figure()
plt.plot(xb, res_arr_dep_we.sum(axis=0), label='Home departure')
plt.plot(xb, h.sum(axis=1)/h.sum(),label='Loc arrival')
plt.plot(xb, -res_arr_dep_we.sum(axis=1), label='Home arrival')
plt.plot(xb, -h.sum(axis=0)/h.sum(), label='Loc departure')
plt.legend()


#%%