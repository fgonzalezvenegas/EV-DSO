# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:07:16 2019

@author: U546416
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

edt = pd.read_excel(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité\EmploiDuTemps.xlsx')
#%%
cumsuma = edt.Percentage.cumsum()

cumsuma = cumsuma / cumsuma.iloc[-1]

#tirage of N
N = 1e7
pos = np.digitize(np.random.rand(int(N)), cumsuma)

inW = (np.array(edt.AvgIn.iloc[pos] + np.random.randn(int(N)) * edt.StdDevIn.iloc[pos]))%24
outW = (np.array(edt.AvgOut.iloc[pos] + np.random.randn(int(N)) * edt.StdDevOut.iloc[pos]))%24

nmin = 5
bins = [i*nmin/60 for i in range(int(24*60/nmin+1))]
hin, b = np.histogram(inW, bins=bins)
hout, b = np.histogram(outW, bins=bins)

working = (hin.cumsum() - hout.cumsum())/N
#plt.plot(b[:-1], working)
#plt.xlim(0,24)
#plt.ylim(0,1)


#%% Supposing a uniform random time from home to work, with the same time to go home and come back
thw = np.array([8.2,13.8,15.1,15.1,12.1,9.5,6.8,4.2,3.5,2.8,2.1,2.6,3.7]).cumsum()
thw = thw/thw[-1]   
binsthw = pd.Series([i*5/60 for i in range(13)])


timehw = np.array(binsthw.iloc[np.digitize(np.random.rand(int(N)), thw)] + 2.5/60)
#%%
dep = (inW - timehw)%24
arr = (outW + timehw)%24

deph, b = np.histogram(dep, bins=bins)
arrh, b = np.histogram(arr, bins=bins)
#%%
f, ax = plt.subplots()
ax.plot(b[:-1], hin, label='ArrAtWork')
ax.plot(b[:-1], hout, label='DepFrWork')
ax.plot(b[:-1], deph, label='DepFrHome')
ax.plot(b[:-1], arrh, label='ArrAtHome')
plt.legend()
ax.set_xlim(0,24)
ax.set_title('Arrival and departure to/from work')

athome = 1-(deph.cumsum() - arrh.cumsum())/N
f, ax = plt.subplots()
ax.plot(b[:-1], working, label='At Work')
ax.plot(b[:-1], athome, label='At Home')
ax.plot(b[:-1], working + athome, label='At Work+Home')
ax.set_xlim(0,24)
ax.set_ylim(0,1)
plt.legend()
ax.set_title('Vehicles available')

#%% Load RTE data
edtRTE = pd.read_excel(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité\DepartureArrivals.xlsx', index_col=0)
#%% create random agents according edtRTE
cumsumRTE = edtRTE.HomeArrivalRTE.cumsum() / 100

arrhRTE = (np.digitize(np.random.rand(int(N)), cumsumRTE) + 0.5 + np.random.rand(int(N)) * 2 - 1)%24
depwRTE = (arrhRTE - timehw)%24

nmin = 60
bins = [i*nmin/60 for i in range(int(24*60/nmin+1))]
hin, b = np.histogram(inW, bins=bins)
hout, b = np.histogram(outW, bins=bins)
deph, b = np.histogram(dep, bins=bins)
arrh, b = np.histogram(arr, bins=bins)
houtRTE, b = np.histogram(depwRTE, bins=bins)
#%%
f, ax = plt.subplots()
ax.plot(b[:-1], hin/N, label='ArrAtWork')
ax.plot(b[:-1], hout/N, label='DepFrWork')
ax.plot(b[:-1], deph/N, label='DepFrHome')
ax.plot(b[:-1], arrh/N, label='ArrAtHome')

ax.plot(b[:-1], edtRTE.HomeArrivalRTE/100, label='ArrAtHomeRTE')
ax.plot(b[:-1], houtRTE/N, label='DepFrWorkRTE')
plt.legend()

#%% Save
tosave = pd.DataFrame([hin/N, hout/N, deph/N, arrh/N, edtRTE.HomeArrivalRTE/100, houtRTE/N], 
                     columns=edtRTE.index, 
                     index=['ArrWork', 'DepWork', 'DepHome', 
                           'ArrHome', 'ArrHomeRTE', 'DepWorkRTE']).transpose()
tosave.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité\DepartureArrivals.xlsx')