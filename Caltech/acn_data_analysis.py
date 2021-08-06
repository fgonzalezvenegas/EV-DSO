# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:50:28 2020

@author: U546416
"""

import pandas as pd
import json
import pytz
import numpy as np
import matplotlib.pyplot as plt
import util

folder = r'c:\user\U546416\Documents\PhD\Data\Caltech_ACN\\'
# Sites = Caltech; JPL; Office1
site = ['Caltech', 'JPL', 'Office1']

data = {}
for s in site:
    fname = 'acndata_sessions_' + s + '_20200213.json'

    with open(folder + fname) as data_file:    
        d = json.load(data_file)  

    data[s] = pd.DataFrame(d["_items"])
    data[s]['site'] = s

ch_data = pd.concat(data.values(), ignore_index=True)
#%% Read data and parse dates
## open data
#with open(folder + fname) as data_file:    
#    d = json.load(data_file)  
#
#ch_data = pd.DataFrame(d["_items"])

# parse dates
Nat = pd.to_datetime(pd.DataFrame({'t' : [None]}).t)[0]
zone =  pytz.timezone(ch_data.timezone[0])
utc = pytz.timezone('GMT')

ch_data.connectionTime = pd.to_datetime(ch_data.connectionTime).apply(lambda x: utc.localize(x).astimezone(zone) if x==x else x)
ch_data.disconnectTime = pd.to_datetime(ch_data.disconnectTime).apply(lambda x: utc.localize(x).astimezone(zone) if x==x else x)
ch_data.doneChargingTime = pd.to_datetime(ch_data.doneChargingTime).apply(lambda x: utc.localize(x).astimezone(zone) if x==x else x)

# Get session length and initial time in floats
ch_data['sessionTime'] = (ch_data.disconnectTime - ch_data.connectionTime).dt.days * 24 + (ch_data.disconnectTime - ch_data.connectionTime).dt.seconds/60/60
ch_data['chargingTime'] = (ch_data.doneChargingTime - ch_data.connectionTime).dt.seconds/60/60
idx = ch_data[ch_data.chargingTime<=0.08].index
ch_data.chargingTime[idx] = ch_data.sessionTime[idx]
ch_data['initTime'] = ch_data.connectionTime.dt.hour + ch_data.connectionTime.dt.minute/60 
ch_data['endTime'] = ch_data.disconnectTime.dt.hour + ch_data.disconnectTime.dt.minute/60 
ch_data['avgChargePower'] = ch_data.kWhDelivered/ch_data.chargingTime
ch_data['date'] = ch_data.connectionTime.dt.date
ch_data['dayofweek'] = ch_data.connectionTime.dt.dayofweek
ch_data['week'] = ch_data.connectionTime.dt.week
ch_data['year'] = ch_data.connectionTime.dt.year

#%% Do 2D -histogram of Start vs session length
for s in site:
    binsh = np.arange(0,24.5,0.5)
    hist, binx, biny = np.histogram2d(ch_data[(ch_data.site==s) & (ch_data.dayofweek<=5)].sessionTime, 
                                      ch_data[(ch_data.site==s) & (ch_data.dayofweek<=5)].initTime, 
                                      bins=[binsh,binsh])
    
    f, ax = plt.subplots()
    i = ax.imshow(hist/hist.sum().sum(), origin='lower', extent=(0,24,0,24))
    ax.set_title('Distribution of sessions for site '+ s)
    ax.set_xlabel('Start of charging sessions [h]')
    ax.set_ylabel('Duration of charging sessions [h]')
    plt.xticks(np.arange(0,25,6))
    plt.yticks(np.arange(0,25,6))
    plt.colorbar(i)

#%% Do 2D -histogram of Start vs End(weekday)
for s in site:
    binsh = np.arange(0,24.5,0.5)
    hist, binx, biny = np.histogram2d(ch_data[(ch_data.site==s) & (ch_data.dayofweek<5)].initTime, 
                                      ch_data[(ch_data.site==s) & (ch_data.dayofweek<5)].endTime, 
                                      bins=[binsh,binsh])
    
    f, (ax, ax2) = plt.subplots(1,2)
    i = ax.imshow(hist.T/hist.sum().sum(), origin='lower', extent=(0,24,0,24))
    ax.set_title('Distribution of sessions')
    ax.set_xlabel('Start of charging sessions')
    ax.set_ylabel('End of charging sessions')
    ax.set_xticks(np.arange(0,25,2))
    ax.set_yticks(np.arange(0,25,2))
    ax.set_xticklabels(np.arange(0,25,2))
    ax.set_yticklabels(np.arange(0,25,2))
    plt.colorbar(i, ax=ax)
    
    ax2.bar((binsh[:-1]+binsh[1:])/2, hist.sum(axis=1)/hist.sum().sum(), width=0.5, label='Arrivals')
    ax2.bar((binsh[:-1]+binsh[1:])/2, -hist.sum(axis=0)/hist.sum().sum(), width=0.5, label='Departures')
    ax2.set_xlim(0,24)
    ax2.set_xticks(np.arange(0,25,2))
    ax2.set_xticklabels(np.arange(0,25,2))
    ax2.set_title('Arrival and departure distribution')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Distribution')
    ax2.legend()
    ax2.grid()
    f.suptitle('Weekdays, ' + s)

#%% Save histograms
of = r'Outputs\\'
for s in ['JPL', 'Caltech']:
    binsh = np.arange(0,24.5,0.5)
    hist, binx, biny = np.histogram2d(ch_data[(ch_data.site==s) & (ch_data.dayofweek<5)].initTime, 
                                      ch_data[(ch_data.site==s) & (ch_data.dayofweek<5)].endTime, 
                                      bins=[binsh,binsh])
    
    pd.DataFrame(hist/hist.sum().sum(), columns=binx[:-1], index=biny[:-1]).to_csv(folder + of + s + '_wd.csv')
    

#%% Do 2D -histogram of Start vs End (weekend)
for s in site:
    binsh = np.arange(0,24.5,0.5)
    hist, binx, biny = np.histogram2d(ch_data[(ch_data.site==s) & (ch_data.dayofweek >= 5)].initTime, 
                                      ch_data[(ch_data.site==s) & (ch_data.dayofweek >= 5)].endTime, 
                                      bins=[binsh,binsh])
    if hist.sum().sum() == 0:
        continue
    f, (ax, ax2) = plt.subplots(1,2)
    i = ax.imshow(hist.T/hist.sum().sum(), origin='lower', extent=(0,24,0,24))
    ax.set_title('Distribution of sessions')
    ax.set_xlabel('Start of charging sessions')
    ax.set_ylabel('End of charging sessions')
    ax.set_xticks(np.arange(0,25,2))
    ax.set_yticks(np.arange(0,25,2))
    ax.set_xticklabels(np.arange(0,25,2))
    ax.set_yticklabels(np.arange(0,25,2))
    plt.colorbar(i, ax=ax)
    
    ax2.bar((binsh[:-1]+binsh[1:])/2, hist.sum(axis=1)/hist.sum().sum(), width=0.5, label='Arrivals')
    ax2.bar((binsh[:-1]+binsh[1:])/2, -hist.sum(axis=0)/hist.sum().sum(), width=0.5, label='Departures')
    ax2.set_xlim(0,24)
    ax2.set_xticks(np.arange(0,25,2))
    ax2.set_xticklabels(np.arange(0,25,2))
    ax2.set_title('Arrival and departure distribution')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Distribution')
    ax2.legend()
    ax2.grid()
    f.suptitle('Weekends, ' + s)

##%% tweek some data to use later as input for EVmodel
## Remove overnight charging
#for i in range(24):
#    for e in range(i):
#        hist[i,e]=0
#f, ax = plt.subplots()
#i = ax.imshow(hist.T, origin='lower', extent=(0,24,0,24))
##ax.set_title('Number of sessions for site '+ site)
#ax.set_xlabel('Start of charging sessions')
#ax.set_ylabel('End of charging sessions')
#plt.colorbar(i)
##
##%%
## Increase a bit the short sessions (<3h)
#for i in range(24):
#    hist[i,i:i+3]=hist[i,i:i+3]*2
#for i in range(15):
#    hist[i,i:i+3]=hist[i,i:i+3]*1.5
#f, ax = plt.subplots()
#i = ax.imshow(hist.T, origin='lower', extent=(0,24,0,24))
##ax.set_title('Number of sessions for site '+ site)
#ax.set_xlabel('Start of charging sessions')
#ax.set_ylabel('End of charging sessions')
#plt.colorbar(i)
##%%
## Modify it in one hour to have peak arrival at 8 and departure at 5
#h = np.zeros((25,25))
#h[1:,1:] = hist
#h[0,0] = hist[-1,-1]
#h[0,1:] = hist[-1,:]
#h[1:,0] = hist[:,-1]
#h = h[0:24,0:24]
##%%
#plt.subplots()
#plt.plot(hist.sum(axis=1), label='Arrival time')
#plt.plot(hist.sum(axis=0), label='Departure time')
#plt.plot(h.sum(axis=1), label='Arrival time - Mod')
#plt.plot(h.sum(axis=0), label='Departure time- Mod')
#

#%% kWh and session length
binsh = np.arange(0,15,0.5)
binskwh = np.arange(0,60,2)
hist, binx, biny = np.histogram2d(ch_data.sessionTime, ch_data.kWhDelivered, bins=[binsh,binskwh])

f, ax = plt.subplots()
i = ax.imshow(hist.T, origin='lower', extent=(0,max(binsh),0,max(binskwh)), aspect=0.25)
#ax.set_title('Number of sessions for site '+ site)
ax.set_ylabel('Delivered kWh')
ax.set_xlabel('Length of charging sessions')
plt.colorbar(i)

#%% Distinct users

nusers = ch_data.userID.nunique()
nsessions = ch_data.groupby('userID')['_id'].nunique()
kwhs = ch_data.groupby('userID')['kWhDelivered'].sum()

nullsessions = ch_data.userID.isnull().sum()
kwhsnulls = ch_data[ch_data.userID.isnull()].kWhDelivered.sum()
cum_ns  = np.concatenate([ch_data.userID.isnull().sum()], nsessions.sort_values().cumsum().values)


#%% Daily charges

dailych = ch_data.groupby('date')['kWhDelivered'].sum()
dailysessions = ch_data.groupby(['site','date'])['_id'].nunique()
weekch = ch_data.groupby(['year', 'week'])['kWhDelivered'].sum()
dowch = ch_data.groupby('dayofweek')['kWhDelivered'].sum()


#%% connected evs
dt = 0.25
bins = np.arange(0,24,dt)
conn_evs = {}

for s, d in dailysessions.index:
    if d.day==1:
        print(s, d)
    nevs = np.zeros(len(bins))
    ch_d = ch_data[(ch_data.date == d) & (ch_data.site == s)]
    for n, b in enumerate(bins):
        nevs[n] = ch_d[(ch_d.sessionTime > (b-ch_d.initTime)%24)].shape[0]
    conn_evs[s,d] = nevs
conn_evs = pd.DataFrame(conn_evs).T
conn_evs.columns = bins
#bins = np.arange(0,24,0.25)
#conn_evs = pd.DataFrame(columns=bins)
#
#for d in dailysessions['Caltech'].index:
#    if d.day==1:
#        print(d)
#    nevs = np.zeros(len(bins))
#    ch_d = ch_data[ch_data.date == d]
#    for i, e in ch_d.iterrows():
#        
#    conn_evs[d] = nevs
    
#%%
conn_2019wd = conn_evs.reset_index()
conn_2019wd = conn_2019wd[conn_2019wd['level_1'].apply(lambda x: ((x.weekday()<5) & (x.year == 2019)))]    
conn_2019wd.set_index(['level_0', 'level_1'], inplace=True)

# dropping hollidays
hollidays = conn_2019wd.loc['Caltech'][conn_2019wd.loc['Caltech'].max(axis=1)<10].index
conn_2019wd.drop(hollidays, inplace=True, level='level_1')
nevses = {s: ch_data[ch_data.site==s].spaceID.nunique() for s in site}
for s in site:
    conn_2019wd.loc[s].T.plot()
    plt.title('Number of connected EVs, {}'.format(s))
    plt.gca().get_legend().remove()
    plt.subplots()
    plt.plot((conn_2019wd.loc[s].max(axis=1).clip_upper(nevses[s])))
    plt.title('Max simultaneous connections, {}'.format(s))
    plt.axhline(nevses[s], color='r', linestyle='--', label='# EVSE')
                
#%% Plot availability as %
cm = plt.get_cmap('Paired')
NUM_COLORS = 5
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
ncolor = 3

for s in site:
    d = conn_2019wd.loc[s].values
    d.sort(axis=0)
    l, _ = d.shape

    f, ax = plt.subplots()
    ax.plot(bins, d[int(l * 0.5)], color=colors[ncolor], label='Average')
    ax.fill_between(bins, y1=d[int(l*0.25)], y2=d[int(l*0.75)],
                    alpha=0.4, color=colors[ncolor], label='75%')
    ax.fill_between(bins, y1=d[int(l*0.05)], y2=d[int(l*0.95)],
                    alpha=0.2, color=colors[ncolor], label='95%')
    
    ax.set_xlabel('Time [hh:mm]')
    ax.set_ylabel('Connected EVs')
    ax.set_xlim(0,24)
    ax.set_title('Connected EVs, {}'.format(s))
    ax.grid(linestyle='--')
    plt.legend()
                
#%% Get Charging power

plt.subplots()
ch_data.avgChargePower.hist(bins=np.arange(0,20,1))
# Data shows that max is 7 kW
powerEVSEs = ch_data[ch_data.avgChargePower<=7].groupby('stationID')['avgChargePower'].max()
# just to check if all EVSEs have the same Power
#%% Compute Flex pot
powerEVSE = 7
ch_data['flexFactor'] = 1-ch_data.kWhDelivered / (ch_data.sessionTime * powerEVSE)

binsh = np.arange(0,24,0.5)
binsflex = np.arange(0,1.05,0.05)
hist, binx, biny = np.histogram2d(ch_data.flexFactor, ch_data.initTime, bins=[binsflex, binsh])

f, ax = plt.subplots()
i = ax.imshow(hist, origin='lower', extent=(0,max(binsh),0,max(binsflex)), aspect=24)
ax.set_title('Number of sessions for site '+ site)
ax.set_xlabel('Start of charging session')
ax.set_ylabel('Flexibility factor')
plt.colorbar(i)


