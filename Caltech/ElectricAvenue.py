# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:45:41 2020
Electric Nation data analysis

@author: U546416
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import datetime as dt
import util

folder_data = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\\'
print('Reading charging sessions')
try: 
    ch_data = pd.read_csv(folder_data + 'All_Transactions.csv', index_col=0, engine='python')
except:
#    charge_dataGF = pd.read_excel(folder_data + 'GreenFlux Transactions.xlsx')
    charge_dataGF = pd.read_csv(folder_data + 'GreenFlux Transactions.csv',
                                engine='python', sep=';', decimal=',')
    charge_dataCC = pd.read_csv(folder_data + 'CrowdCharge Transactions.csv',
                                engine='python', sep=';', decimal=',')
    charge_dataGF.columns = ['TransactionID', 'ChargerID', 'ParticipantID', 'ParticipantCarkW',
                             'ParticipantCarkWh', 'StartTime', 'StopTime', 'GroupID', 'Trial',
                             'AdjustedStartTime', 'AdjustedStopTime', 'PluggedInTime', 'ConsumedkWh',
                             'PartOfManagedGroup', 'WeekdayOrWeekend', 'ActiveChargingStart',
                             'MaxAmpsDrawnForT', 'EndCharge', 'ChargingDuration',
                             'tInactiveStart', 'tInactiveEnd', 'UsedATimer', 
                             'BeganInWeekdayEveningPeak', 'HotUnplug', 'Managed', 'PercentageTimeInTransactionManaged']
    charge_dataGF.drop('PercentageTimeInTransactionManaged', axis=1, inplace=True)
    charge_dataCC.columns = ['TransactionID', 'ChargerID', 'ParticipantID', 'ParticipantCarkW', 
                             'ParticipantCarkWh', 'StartTime', 'StopTime', 'GroupID', 'Trial', 
                             'AdjustedStartTime', 'AdjustedStopTime', 'PluggedInTime', 'ConsumedkWh',
                             'PartOfManagedGroup', 'WeekdayOrWeekend', 'ActiveChargingStart',
                             'MaxAmpsDrawnForT', 'EndCharge', 'ChargingDuration',
                             'tInactiveStart', 'tInactiveEnd', 'UsedATimer',
                             'BeganInWeekdayEveningPeak', 'HotUnplug', 'Managed',
                             'T2_Managed', 'Restricton T1', 'Restriction T2']
    charge_dataCC.drop(['T2_Managed', 'Restricton T1', 'Restriction T2'], axis=1, inplace=True)
    charge_dataGF['Aggregator'] = 'GreenFlux'
    charge_dataCC['Aggregator'] = 'CrowdCharge'
    
    ch_data = pd.concat([charge_dataGF, charge_dataCC], ignore_index=True)
    ch_data.to_csv(folder_data + 'All_Transactions.csv')
print('formating')
#ch_data = ch_data[ch_data.PluggedInTime > 20]
ch_data.StartTime = pd.to_datetime(ch_data.StartTime, format='%d/%m/%Y %H:%M')
ch_data.StopTime = pd.to_datetime(ch_data.StopTime, format='%d/%m/%Y %H:%M')
ch_data.AdjustedStartTime = pd.to_datetime(ch_data.AdjustedStartTime, format='%d/%m/%Y %H:%M')
ch_data.AdjustedStopTime = pd.to_datetime(ch_data.AdjustedStopTime, format='%d/%m/%Y %H:%M')
ch_data = ch_data[~ch_data.ParticipantID.isnull()]
ch_data.sort_values(['ChargerID', 'StartTime'], inplace=True) #, ignore_index=True)
ch_data.reset_index(inplace=True, drop=True)
print('reading charger types/EVs')
chargers = pd.read_excel(folder_data + 'ChargerInstall.xlsx', index_col=1)
chargers.index = chargers.index.astype(str)
chargers.PIVType = chargers.PIVType.apply(lambda x: x[-6:].replace('(','').replace(' ','').replace(')',''))
bevs = list(chargers[chargers.PIVType=='BEV'].index)
user_charger = pd.Series(data=chargers.index, index=chargers.ParticipantID)
print('Reading comm data')
try:
    comm = pd.read_csv(folder_data + 'Comm_All.csv', index_col=0, engine='python')
except:
    commGF = pd.read_csv(folder_data + 'CommGF.csv', index_col=0,
                         engine='python', sep=';', decimal=',')
    commCC = pd.read_csv(folder_data + 'CommCC.csv', index_col=0,
                         engine='python', sep=';', decimal=',')
    colsCC = ['{}'.format('2017' if '2017' in c else '2018') + 'W{:02d}'.format(int(c[3:5])) for c in commCC.columns]
    commCC.columns = colsCC
    commCC.index = user_charger[commCC.index]
    colsGF = ['{}'.format('2017' if '2017' in c else '2018') + 'W{:02d}'.format(int(c[2:4])) for c in commGF.columns]
    commGF.columns = colsGF
    commGF['2018W08'] = 0 # this column is wrong
    commGF = commGF/commGF.max().max() # un de deux est en base 100 et l'autre en pu
    commCC = commCC/commCC.max().max()
    comm = pd.concat([commGF, commCC])
    comm.index.name='ChargerID'
    comm.to_csv(folder_data + 'Comm_All.csv')


#%% Cleaning data
print('Cleaning data: initial charging sessions {}; unique users {}'.format(ch_data.shape[0], ch_data.ParticipantID.nunique()))

# Identifying chargign sessions where end and start of next one are really close to each other
dtmax = 10 # minutes
# time between starts of ch sessions
dss = -ch_data.StartTime.diff(-1)
# time of charging session
dcs = ch_data.StopTime - ch_data.StartTime

# time between end of session and next one
dbtwn = dss-dcs
# identifying sessions where the time between is too small
idxs = dbtwn[(dbtwn.dt.seconds/60 <= dtmax) &     # delta between sessions in minutes (always positive)
             (dbtwn.dt.days == 0) &               # detla between sessions in days (can be negative)
             (ch_data.ParticipantID == ch_data.ParticipantID.shift(-1))].index

# For those sessions, we will merge them in only one
for j,i in enumerate(idxs):
#    if j%100==0:
#        print('\t', j)
    ch_data.loc[i+1, 'StarTime'] =  ch_data.loc[i,'StartTime']
    ch_data.loc[i+1, 'AdjustedStartTime'] = ch_data.loc[i,'AdjustedStartTime']
#    ch_data.PluggedInTime[i+1] = (ch_data.PluggedInTime[i] + ch_data.PluggedInTime[i] +
#                                     int(dbtwn[i].seconds/60))
ch_data.PluggedInTime = (ch_data.AdjustedStopTime - ch_data.AdjustedStartTime).dt.seconds/60
ch_data.drop(idxs, inplace=True)
ch_data = ch_data[ch_data.PluggedInTime > 20]
print('Joining close sessions, remaining sessions {}; unique users {}'.format(ch_data.shape[0], ch_data.ParticipantID.nunique()))

#% Dropping values really wrong:
ch_data = ch_data[ch_data.StopTime.dt.year.isin([2017,2018])]
ch_data = ch_data[ch_data.StartTime.dt.year.isin([2017,2018])]
print('Removing invalid dates, remaining sessions {}; unique users {}'.format(ch_data.shape[0], ch_data.ParticipantID.nunique()))

nminsessions = 20
value_counts = ch_data.ChargerID.value_counts()
ch_data = ch_data[ch_data.ChargerID.isin(value_counts[value_counts>=nminsessions].index)]
print('Removing users with too few ch sessions, remaining sessions {}; unique users {}'.format(ch_data.shape[0], ch_data.ParticipantID.nunique()))

# Remove PHEVs & REX
bev_data = ch_data[ch_data.ChargerID.isin(bevs)]
bev_data.reset_index(inplace=True, drop=True)
print('Removed PHEVs & REX, remaining sessions {}; unique users {}'.format(bev_data.shape[0], bev_data.ParticipantID.nunique()))

# Useful KPI
# Number of sessions
value_counts = bev_data.ChargerID.value_counts()
# First session
tini = bev_data.groupby('ChargerID').StartTime.min()
# last session
tend = bev_data.groupby('ChargerID').StopTime.max()
# Days between first and last session
pilottime = (tend-tini).dt.days

ndaysmin = 90
bev_data = bev_data[bev_data.ChargerID.isin(pilottime[pilottime>=ndaysmin].index)]
print('Removing users with time at trial less than {} days, remaining sessions {}; unique users {}'.format(ndaysmin, 
      bev_data.shape[0], bev_data.ParticipantID.nunique()))

# Useful KPI
# Number of sessions
value_counts = bev_data.ChargerID.value_counts()
# First session
tini = bev_data.groupby('ChargerID').StartTime.min()
# last session
tend = bev_data.groupby('ChargerID').StopTime.max()
# Days between first and last session
pilottime = (tend-tini).dt.days

# Daily sessions
ch_sessions = (value_counts/pilottime).sort_values()
# Correcting weird charging sessions where consumed kWh>car batt size 
bev_data['correctedConskWh'] = bev_data[['ConsumedkWh', 'ParticipantCarkWh']].min(axis=1)
# Total charged energy
ch_enc = bev_data.groupby('ChargerID').correctedConskWh.sum()
# Daily charged energy
daily_enc = ch_enc/pilottime
# Charged energy per session
enc_session = ch_enc/value_counts
# SOC charged per session
soc_session = enc_session/chargers.CarkWh[bevs]


# Computing KPIs of communication availability
wi = tini.dt.week
we = tend.dt.week
yi = tini.dt.year
ye = tend.dt.year
comm_av = {}
comm_sup = {}
comm_sup_cols = {}
for i in wi.index:
    w, y, wf, yf = wi[i], yi[i], we[i], ye[i]
    dm = (wf-w) + 52*(yf-y)
    cols = ['{}W{:02d}'.format((y + (w+d)//53), ((w+d)%53 + (w+d)//53)) for d in range(dm+1)]
    k = comm.loc[i, cols]
    comm_av[i] = k.mean()
    comm_sup_cols[i] = k[k>0.7]
    comm_sup[i] = k[k>0.7].mean()
comm_av_sup = pd.DataFrame([comm_sup, comm_av], index=['Average', 'SupAvg']).T

#%% Plot distribution of batt sizes
bevs = ch_sessions.index

dx = 10 #kWh
bins=np.arange(10,101,dx)
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(chargers.CarkWh[small], bins=bins)
h2,_ = np.histogram(chargers.CarkWh[big], bins=bins)
x=(bins[:-1] + bins[1:])/2

f = plt.figure()
plt.grid(axis='y', linestyle='--', zorder=2)
plt.rcParams['hatch.linewidth'] = 0.5
plt.bar(x, h1, width=dx, edgecolor=['k' for i in range(len(h1))], linewidth=0.7, zorder=10, label='Small BEVs')
plt.bar(x, h2, width=dx, edgecolor=['k' for i in range(len(h2))], linewidth=0.7, zorder=10, hatch='x', label='Large BEVs')
plt.xticks(bins)
#plt.title('Distribution of battery size')
plt.xlabel('Battery size [kWh]')
plt.ylabel('Count')  
plt.legend()
#f.set_size_inches(5.6,4)

f.set_size_inches(3.5,2.8)
plt.tight_layout()

folder_img = r'c:\user\U546416\Pictures\ElectricNation\PlugIn\\'
f.savefig(folder_img + 'Batt_size_vsinglecol.png')
f.savefig(folder_img + 'Batt_size_vsinglecol.pdf')

#%% Plot histogram of charging sessions // daily
db=0.125
bins= np.arange(0,2.5,db)
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(ch_sessions[big], bins=bins)
h2,_ = np.histogram(ch_sessions[small], bins=bins)
xb = (bins[:-1]+bins[1:])/2
ks = ['k' for k in range(len(xb))]
plt.figure()
plt.rcParams['hatch.linewidth'] = 0.5
plt.bar(xb, h2, label='Small BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=10)
plt.bar(xb, h1, bottom=h2, label='Large BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=9, hatch='x') 
plt.grid(axis='y', linestyle='--', zorder=2)
#plt.xticks(bins)
plt.title('Average daily charging sessions')
plt.xlabel('Daily charging sessions')
plt.ylabel('Count') 
plt.axvline(x=ch_sessions[big].median(), linestyle='--', linewidth=1.5, color='r', zorder=20, label='_')
plt.axvline(x=ch_sessions[small].median(), linestyle='-.', linewidth=1.5, color='darkblue', zorder=21, label='_')
x0 = ch_sessions[big].median()
x1 = ch_sessions[small].median()
y0 = max(h1+h2)*0.8
y1 = max(h1+h2)*0.75
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.8, s='Large BEV - Median {:.2f}'.format(ch_sessions[big].median()))
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.75, s='Small BEV - Median {:.2f}'.format(ch_sessions[big].median()))
plt.annotate('Large BEV - Median {:.2f}'.format(ch_sessions[big].median()), 
             xy=(x0, y0), xytext=(x1+2*db,y0), arrowprops=dict(arrowstyle='<-', color='r'), 
             horizontalalignment='left', verticalalignment='center')
plt.annotate('Small BEV - Median {:.2f}'.format(ch_sessions[small].median()), 
             xy=(x1, y1), xytext=(x1+2*db,y1), arrowprops=dict(arrowstyle='<-', color='darkblue'),
             horizontalalignment='left', verticalalignment='center')
plt.legend() 

#%% Plot histogram of charging sessions // weekly
db=0.5
bins= np.arange(0,7*2,db)
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(ch_sessions[big]*7, bins=bins)
h2,_ = np.histogram(ch_sessions[small]*7, bins=bins)
xb = (bins[:-1]+bins[1:])/2
ks = ['k' for k in range(len(xb))]
plt.figure()
plt.bar(xb, h2, label='Small BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=10)
plt.bar(xb, h1, bottom=h2, label='Large BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=10, hatch='x') 
plt.grid(axis='y', linestyle='--', zorder=2)
#plt.xticks(bins)
#plt.title('Average weekly charging sessions')
plt.xlabel('Weekly charging sessions')
plt.ylabel('Count')
plt.axvline(x=ch_sessions[big].median()*7, linestyle='--', color='r', zorder=20, label='_')
plt.axvline(x=ch_sessions[small].median()*7, linestyle='--', color='darkblue', zorder=20, label='_')
x0 = ch_sessions[big].median()*7
x1 = ch_sessions[small].median()*7
y0 = max(h1+h2)*0.82
y1 = max(h1+h2)*0.77
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.8, s='Large BEV - Median {:.2f}'.format(ch_sessions[big].median()))
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.75, s='Small BEV - Median {:.2f}'.format(ch_sessions[big].median()))
plt.annotate('Large BEV - Median {:.2f}'.format(ch_sessions[big].median()*7), 
             xy=(x0, y0), xytext=(x1+2*db,y0), arrowprops=dict(arrowstyle='<-', color='r'), 
             horizontalalignment='left', verticalalignment='center')
plt.annotate('Small BEV - Median {:.2f}'.format(ch_sessions[small].median()*7), 
             xy=(x1, y1), xytext=(x1+2*db,y1), arrowprops=dict(arrowstyle='<-', color='darkblue'),
             horizontalalignment='left', verticalalignment='center')
plt.legend() 
plt.xlim(0,14)

#%% Plot histogram of charged energy // daily
db=1
bins= np.arange(0,30,db)
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(daily_enc[big], bins=bins)
h2,_ = np.histogram(daily_enc[small], bins=bins)
xb = (bins[:-1]+bins[1:])/2
ks = ['k' for k in range(len(xb))]
plt.figure()
plt.rcParams['hatch.linewidth'] = 0.5
plt.bar(xb, h2, label='Small BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=10)
plt.bar(xb, h1, bottom=h2, label='Large BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=9, hatch='x') 
plt.grid(axis='y', linestyle='--', zorder=2)
#plt.xticks(bins)
plt.title('Average daily charged energy [kWh]')
plt.xlabel('Daily charged energy [kWh]')
plt.ylabel('Count') 
plt.axvline(x=daily_enc[big].median(), linestyle='--', linewidth=1.5, color='r', zorder=20, label='_')
plt.axvline(x=daily_enc[small].median(), linestyle='-.', linewidth=1.5, color='darkblue', zorder=21, label='_')
x0 = daily_enc[big].median()
x1 = daily_enc[small].median()
y0 = max(h1+h2)*0.78
y1 = max(h1+h2)*0.73
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.8, s='Large BEV - Median {:.2f}'.format(ch_sessions[big].median()))
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.75, s='Small BEV - Median {:.2f}'.format(ch_sessions[big].median()))
plt.annotate('Large BEV - Median {:.1f} kWh'.format(daily_enc[big].median()), 
             xy=(x0, y0), xytext=(12,y0), arrowprops=dict(arrowstyle='<-', color='r'), 
             horizontalalignment='left', verticalalignment='center', zorder=20)
plt.annotate('Small BEV - Median {:.1f} kWh'.format(daily_enc[small].median()), 
             xy=(x1, y1), xytext=(12,y1), arrowprops=dict(arrowstyle='<-', color='darkblue'),
             horizontalalignment='left', verticalalignment='center', zorder=20)
plt.legend() 

#%% Plot histogram of charged energy per session
db=2
bins= np.arange(0,60,db)
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(enc_session[big], bins=bins)
h2,_ = np.histogram(enc_session[small], bins=bins)
xb = (bins[:-1]+bins[1:])/2
ks = ['k' for k in range(len(xb))]
plt.figure()
plt.rcParams['hatch.linewidth'] = 0.5
plt.bar(xb, h2, label='Small BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=10)
plt.bar(xb, h1, bottom=h2, label='Large BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=9, hatch='x') 
plt.grid(axis='y', linestyle='--', zorder=2)
#plt.xticks(bins)
plt.title('Charged energy per session [kWh]')
plt.xlabel('Energy [kWh]')
plt.ylabel('Count') 
plt.axvline(x=enc_session[big].median(), linestyle='--', linewidth=1.5, color='r', zorder=20, label='_')
plt.axvline(x=enc_session[small].median(), linestyle='-.', linewidth=1.5, color='darkblue', zorder=21, label='_')
x0 = enc_session[big].median()
x1 = enc_session[small].median()
y0 = max(h1+h2)*0.78
y1 = max(h1+h2)*0.73
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.8, s='Large BEV - Median {:.2f}'.format(ch_sessions[big].median()))
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.75, s='Small BEV - Median {:.2f}'.format(ch_sessions[big].median()))
plt.annotate('Large BEV - Median {:.1f} kWh'.format(enc_session[big].median()), 
             xy=(x0, y0), xytext=(33,y0), arrowprops=dict(arrowstyle='<-', color='r'), 
             horizontalalignment='left', verticalalignment='center', zorder=20)
plt.annotate('Small BEV - Median {:.1f} kWh'.format(enc_session[small].median()), 
             xy=(x1, y1), xytext=(33,y1), arrowprops=dict(arrowstyle='<-', color='darkblue'),
             horizontalalignment='left', verticalalignment='center', zorder=20)
plt.legend() 

#%% Plot histogram of charged SOC per session
db=0.05
bins= np.arange(0,1.01,db)
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(soc_session[big], bins=bins)
h2,_ = np.histogram(soc_session[small], bins=bins)
xb = (bins[:-1]+bins[1:])/2
ks = ['k' for k in range(len(xb))]
plt.figure()
plt.rcParams['hatch.linewidth'] = 0.5
plt.bar(xb, h2, label='Small BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=10)
plt.bar(xb, h1, bottom=h2, label='Large BEV', edgecolor=ks, linewidth=0.7, width=db, zorder=9, hatch='x') 
plt.grid(axis='y', linestyle='--', zorder=2)
#plt.xticks(bins)
plt.title('Charged SOC per session')
plt.xlabel('SOC')
plt.ylabel('Count') 
plt.axvline(x=soc_session[big].median(), linestyle='--', linewidth=1.5, color='r', zorder=20, label='_')
plt.axvline(x=soc_session[small].median(), linestyle='-.', linewidth=1.5, color='darkblue', zorder=21, label='_')
x0 = soc_session[big].median()
x1 = soc_session[small].median()
y0 = max(h1+h2)*0.78
y1 = max(h1+h2)*0.73
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.8, s='Large BEV - Median {:.2f}'.format(ch_sessions[big].median()))
#plt.text(x=ch_sessions[small].median()+db/2, y=max(h1+h2)*0.75, s='Small BEV - Median {:.2f}'.format(ch_sessions[big].median()))
plt.annotate('Large BEV - Median {:.2f}'.format(soc_session[big].median()), 
             xy=(x0, y0), xytext=(0.65,y0), arrowprops=dict(arrowstyle='<-', color='r'), 
             horizontalalignment='left', verticalalignment='center', zorder=20)
plt.annotate('Small BEV - Median {:.2f}'.format(soc_session[small].median()), 
             xy=(x1, y1), xytext=(0.65,y1), arrowprops=dict(arrowstyle='<-', color='darkblue'),
             horizontalalignment='left', verticalalignment='center', zorder=20)
plt.legend()

#%% Plot histogram of equivalent distance // daily
db=10
bins= np.arange(0,150,db)
# driving efficiency according to Weiss et al, 2020,
#Energy efficiency trade-offs in small to large electric vehicles
# E[kWh/100km]= (0.09+-0.01) batt_size[kWh] + 14+-1
# => dreff(25kWh) = 16.25 +- 1.25
# => dreff(50kWh) = 18.5 +- 1. 
# => dreff(75kWh) = 20.75
dreff= (chargers.CarkWh[bevs] * 0.09 + 14)/100
ch_eff = 0.95
daily_dist = (daily_enc[bevs]*ch_eff)/dreff[bevs]
#eff_small= 0.16
#eff_big = 0.2
big = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh>50) & chargers.index.isin(ch_sessions.index)].index)
small = list(chargers[(chargers.PIVType=='BEV') & (chargers.CarkWh<50) & chargers.index.isin(ch_sessions.index)].index)
h1,_ = np.histogram(daily_dist[big], bins=bins)
h2,_ = np.histogram(daily_dist[small], bins=bins)
xb = (bins[:-1]+bins[1:])/2
ks = ['k' for k in range(len(xb))]
f,ax=plt.subplots(1,2)
plt.rcParams['hatch.linewidth'] = 0.5

plt.sca(ax[0])
c=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
plt.bar(xb, h2/h2.sum(), label='Small BEV', color=c, 
        edgecolor=ks, linewidth=0.7, width=db, zorder=10)
plt.grid(axis='y', linestyle='--', zorder=2)
plt.title('Small BEV')
plt.xlabel('Distance [km]')
plt.ylabel('Distribution') 
plt.axvline(x=daily_dist[small].median(), linestyle='--', 
            linewidth=1.5, color='darkblue', zorder=20, label='_')
plt.xticks(np.linspace(0,150,7))
plt.ylim(0,0.26)
plt.yticks(np.linspace(0,0.25,6))
x0 = daily_dist[small].median()
y0=0.21
x1=75
plt.annotate('Median {:.1f} km'.format(x0),
             xy=(x0, y0), xytext=(x1,y0), arrowprops=dict(arrowstyle='<-', color='b'), 
             horizontalalignment='left', verticalalignment='center', zorder=20)
#plt.text(x=100, y=0.21, 
#         s='Median {:.1f} km'.format(daily_enc[small].median()/eff_small),
#         horizontalalignment='center', verticalalignment='center')

plt.sca(ax[1])
c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
plt.bar(xb, h1/h1.sum(), label='Large BEV',  color=c, 
        edgecolor=ks, linewidth=0.7, width=db, zorder=9, hatch='x') 
plt.grid(axis='y', linestyle='--', zorder=2)
plt.title('Large BEV')
plt.xlabel('Distance [km]')
plt.ylabel('Distribution') 
plt.axvline(x=daily_dist[big].median(), linestyle='-.', 
            linewidth=1.5, color='r', zorder=21, label='_')
plt.xticks(np.linspace(0,150,7))
plt.ylim(0,0.26)
plt.yticks(np.linspace(0,0.25,6))
x0 = daily_dist[big].median()
y0=0.21
x1=75
plt.annotate('Median {:.1f} km'.format(x0),
             xy=(x0, y0), xytext=(x1,y0), arrowprops=dict(arrowstyle='<-', color='r'), 
             horizontalalignment='left', verticalalignment='center', zorder=20)
f.set_size_inches(10.9,4.76)
#
#plt.text(x=100, y=0.21, 
#         s='Median {:.1f} km'.format(daily_enc[big].median()/eff_big),
#         horizontalalignment='center', verticalalignment='center')





#%% Do 3d scatter
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
cmap = plt.get_cmap('viridis')
cs = cmap(chargers.CarkWh[bevs]/chargers.CarkWh[bevs].max())

ax.scatter(ch_sessions[bevs]*7, soc_session[bevs], daily_dist[bevs], color=cs)

plt.xlabel('Weekly sessions')
plt.ylabel('Charged SOC per session')
ax.set_zlabel('Daily distance [km]')

#%% Plot various scatter plots. => Scatter Weekly sessions vs. Energy per session And 
### Scatter Weekly sessions vs. Daily distance

#f, ax = plt.subplots()
#ax.scatter(ch_sessions[small]*7, daily_dist[small])
#ax.scatter(ch_sessions[big]*7, daily_dist[big])
#plt.xlabel('Weekly sessions')
#plt.ylabel('Daily distance')

#f, axs = plt.subplots(1,2)


dreff= (chargers.CarkWh[bevs] * 0.09 + 14)/100
ch_eff = 0.95
daily_dist = (daily_enc[bevs]*ch_eff)/dreff[bevs]

f = plt.figure()
cmap = plt.get_cmap('viridis')
cs = cmap(chargers.CarkWh[bevs]/chargers.CarkWh[bevs].max())

dy = 0.08
yt = -0.25
#plt.sca(axs[0])
f.add_axes([0.07,0.1+dy,0.37,0.85-dy])
plt.scatter(ch_sessions[bevs]*7, enc_session[bevs], color=cs, alpha=0.7)
plt.xlabel('Weekly sessions')
plt.ylabel('Charged energy per session [kWh]')
plt.grid('--', alpha=0.5)
plt.xlim(0,12)
plt.ylim(0,70)
plt.gca().set_title('(a)', y=yt)


#plt.sca(axs[1])
ax=f.add_axes([0.51,0.1+dy,0.37,0.85-dy])
plt.scatter(ch_sessions[bevs]*7, daily_dist[bevs], color=cs, alpha=0.7)
plt.xlabel('Weekly sessions')
plt.ylabel('Daily distance [km]')
plt.grid('--', alpha=0.5)
plt.xlim(0,12)
plt.ylim(0,160)
plt.gca().set_title('(b)', y=yt)
#f.set_tight_layout(True)
#f.subplots_adjust(right=0.825)

from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap)
sm.set_array(np.arange(0,101,10))
cax = f.add_axes([0.9, 0.1+dy, 0.025, 0.85-dy])
cbar = plt.colorbar(mappable=sm, cax=cax)
cbar.set_label('Battery size [kWh]')
f.set_size_inches(9,4)#(11,4.76)


# FITTING
import statsmodels.api as sm
## adding Linear regression for small and large EVs
lrbig = sm.OLS(daily_dist[big], sm.add_constant(ch_sessions[big]*7)).fit()
lrsmall = sm.OLS(daily_dist[small], sm.add_constant(ch_sessions[small]*7)).fit()
x = np.arange(0,15,1)
X = sm.add_constant(x)

#lrbig = sm.OLS(daily_dist[big], ch_sessions[big]*7).fit()
#lrsmall = sm.OLS(daily_dist[small], ch_sessions[small]*7).fit()
#X = np.arange(0,15,1)

predbig = lrbig.get_prediction(X)
predsmall = lrsmall.get_prediction(X)
ax.plot(x, predbig.predicted_mean, color='orange', alpha=0.7, zorder=0)
ax.plot(x, predsmall.predicted_mean, color='darkblue', alpha=0.7, zorder=0)
ax.fill_between(x, predbig.conf_int()[:,0],predbig.conf_int()[:,1], color='orange', alpha=0.2)
ax.fill_between(x, predsmall.conf_int()[:,0],predsmall.conf_int()[:,1], color='darkblue', alpha=0.2)
#
rsqbig=lrbig.rsquared
rsqsmall=lrsmall.rsquared

# adding regression values string
dxb, dyb = -0.2, +0.5
dxs, dys = -0.2, -40.5
xb = 9
xs = 12

eq = 'y={:.1f}x+{:.1f}\n'.format(lrbig.params.iloc[1], lrbig.params.iloc[0])
r2 = r'$r^2$={:.2f}'.format(rsqbig)
ax.text(xb+dxb,predbig.predicted_mean[xb]+dyb,
         eq+r2, 
         horizontalalignment='right')
#eq = 'y={:.1f}x+{:.1f}\n'.format(lrsmall.params.iloc[1], lrsmall.params.iloc[0])
#r2 = r'$r^2$={:.2f}'.format(rsqsmall)
#ax.text(xs+dx,predsmall.predicted_mean[xs]+dy,
#         eq+r2, 
#         horizontalalignment='right')

eq = 'y={:.1f}x+{:.1f}\n'.format(lrsmall.params.iloc[1], lrsmall.params.iloc[0])
r2 = r'$r^2$={:.2f}'.format(rsqsmall)
ax.text(xs+dxs,predsmall.predicted_mean[xs]+dys,
         eq+r2, 
         horizontalalignment='right')

#f.savefig(folder_img + 'sed_vsmall.png')
#f.savefig(folder_img + 'sed_vsmall.pdf')

# plot fits found with ev model
#x = np.arange(0,15,1)
#ybig = 25.9*x + -24.9
#ysmall = 14*x-12.9
#ax.plot(x, ybig, color='orange', alpha=0.7, zorder=0)
#ax.plot(x, ysmall, color='darkblue', alpha=0.7, zorder=0)


#f, ax = plt.subplots()
#ax.scatter(ch_sessions[small]*7, enc_session[small])
#ax.scatter(ch_sessions[big]*7, enc_session[big])
#plt.xlabel('Weekly sessions')
#plt.ylabel('Charged energy per session [kWh]')

#f, ax = plt.subplots()
#cmap = plt.get_cmap('viridis')
#cs = cmap(chargers.CarkWh[bevs]/chargers.CarkWh[bevs].max())
#ax.scatter(ch_sessions[bevs]*7, enc_session[bevs], color=cs, alpha=0.7)
#plt.xlabel('Weekly sessions')
#plt.ylabel('Charged energy per session [kWh]')
#plt.grid('--', alpha=0.5)
#plt.xlim(0,10)
#plt.ylim(0,70)
#
#from matplotlib.cm import ScalarMappable
#sm = ScalarMappable(cmap=cmap)
#sm.set_array(np.arange(0,101,10))
#plt.colorbar(mappable=sm, ax=ax)

#f, ax = plt.subplots()
#ax.scatter(ch_sessions[small]*7, soc_session[small])
#ax.scatter(ch_sessions[big]*7, soc_session[big])
#plt.xlabel('Weekly sessions')
#plt.ylabel('Charged SOC per session')

#f, ax = plt.subplots()
#ax.scatter(daily_dist[small], soc_session[small])
#ax.scatter(daily_dist[big],soc_session[big])
#plt.xlabel('Daily dist')
#plt.ylabel('Charged SOC per session')
#%% Do Arrival departure histogram
save_folder = r'c:\user\U546416\Pictures\ElectricNation\ArrDeps\\'
# Weekday histogram
binsh = np.arange(0,24.5,0.5)
histwd, binx, biny = np.histogram2d(ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.minute/60, 
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.minute/60,
                                  bins=[binsh,binsh])

# Weekend histogram
histwe, binx, biny = np.histogram2d(ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.minute/60, 
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.minute/60,
                                  bins=[binsh,binsh])
util.plot_arr_dep_hist(histwd, binsh, ftitle = '')
#plt.savefig(save_folder + 'ArrDep_wd.pdf')
#plt.savefig(save_folder + 'ArrDep_wd.png')
plt.suptitle('Electric Nation, Weekdays')
util.plot_arr_dep_hist(histwe, binsh, ftitle = '')
#plt.savefig(save_folder + 'ArrDep_we.pdf')
#plt.savefig(save_folder + 'ArrDep_we.png')
plt.suptitle('Electric Nation, Weekends')

#pd.DataFrame(histwd/histwd.sum().sum(), columns=binsh[:-1], index=binsh[:-1]).to_csv(folder_data + 'Outputs//EN_arrdep_wd.csv')
#pd.DataFrame(histwe/histwe.sum().sum(), columns=binsh[:-1], index=binsh[:-1]).to_csv(folder_data + 'Outputs//EN_arrdep_we.csv')

# Do histograms only for BEV data
# Weekday histogram
binsh = np.arange(0,24.5,0.5)
histwd, binx, biny = np.histogram2d(bev_data[(bev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.hour +
                                  bev_data[(bev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.minute/60, 
                                  bev_data[(bev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.hour +
                                  bev_data[(bev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.minute/60,
                                  bins=[binsh,binsh])
# Weekend histogram
histwe, binx, biny = np.histogram2d(bev_data[(bev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.hour +
                                  bev_data[(bev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.minute/60, 
                                  bev_data[(bev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.hour +
                                  bev_data[(bev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.minute/60,
                                  bins=[binsh,binsh])

util.plot_arr_dep_hist(histwd, binsh, ftitle = '')
#plt.savefig(save_folder + 'ArrDep_wd_BEVs.pdf')
#plt.savefig(save_folder + 'ArrDep_wd_BEVs.png')
plt.suptitle('Electric Nation, Weekdays. Only BEVs')
#util.plot_arr_dep_hist(histwe, binsh, ftitle = '')
#plt.savefig(save_folder + 'ArrDep_we_BEVs.pdf')
#plt.savefig(save_folder + 'ArrDep_we_BEVs.png')
plt.suptitle('Electric Nation project, Weekends. Only BEVs')
#pd.DataFrame(histwd/histwd.sum().sum(), columns=binsh[:-1], index=binsh[:-1]).to_csv(folder_data + 'Outputs//EN_arrdep_wd_BEVs.csv')
#pd.DataFrame(histwe/histwe.sum().sum(), columns=binsh[:-1], index=binsh[:-1]).to_csv(folder_data + 'Outputs//EN_arrdep_we_BEVs.csv')


# Do histograms only for PHEV data
# Weekday histogram
binsh = np.arange(0,24.5,0.5)
phev_data = ch_data[~ch_data.ChargerID.isin(bevs)]
histwd, binx, biny = np.histogram2d(phev_data[(phev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.hour +
                                  phev_data[(phev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.minute/60, 
                                  phev_data[(phev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.hour +
                                  phev_data[(phev_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.minute/60,
                                  bins=[binsh,binsh])

# Weekend histogram
histwe, binx, biny = np.histogram2d(phev_data[(phev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.hour +
                                  phev_data[(phev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.minute/60, 
                                  phev_data[(phev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.hour +
                                  phev_data[(phev_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.minute/60,
                                  bins=[binsh,binsh])

util.plot_arr_dep_hist(histwd, binsh, ftitle = '')
#plt.savefig(save_folder + 'ArrDep_wd_PHEV.pdf')
#plt.savefig(save_folder + 'ArrDep_wd_PHEV.png')
plt.suptitle('Electric Nation project, Weekdays. PHEVs & REX')
util.plot_arr_dep_hist(histwe, binsh, ftitle = '')
#plt.savefig(save_folder + 'ArrDep_we_PHEV.pdf')
#plt.savefig(save_folder + 'ArrDep_we_PHEV.png')
plt.suptitle('Electric Nation project, Weekends. PHEVs & REX')

#%% MODIFIED DATA TO USE IN FRANCE, ADDING 30min to schedules to make people get a bit later
# Weekday histogram
binsh = np.arange(0,24.5,0.5)
de = 0.5 # hours
histwd, binx, biny = np.histogram2d((ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStartTime.dt.minute/60 + de)%24, 
                                  (ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday<5)].AdjustedStopTime.dt.minute/60 + de)%24,
                                  bins=[binsh,binsh])

histwe, binx, biny = np.histogram2d((ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStartTime.dt.minute/60 + de)%24, 
                                  (ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.hour +
                                  ch_data[(ch_data.AdjustedStartTime.dt.weekday>=5)].AdjustedStopTime.dt.minute/60 + de)%24,
                                  bins=[binsh,binsh])

util.plot_arr_dep_hist(histwd, binsh, ftitle = 'Electric Nation project, Weekdays')
util.plot_arr_dep_hist(histwe, binsh, ftitle = 'Electric Nation project, Weekends')

#pd.DataFrame(histwd/histwd.sum().sum(), columns=binsh[:-1], index=binsh[:-1]).to_csv(folder_data + 'Outputs//EN_arrdep_wd_modifFR.csv')
#pd.DataFrame(histwe/histwe.sum().sum(), columns=binsh[:-1], index=binsh[:-1]).to_csv(folder_data + 'Outputs//EN_arrdep_we_modifFR.csv')

#%% Charge energy
bins_kwh = np.arange(0,70,1)
bins_soc = np.arange(0,1.05,0.05)

hist_kwh, _ = np.histogram(ch_data.ConsumedkWh, bins=bins_kwh)
hist_soc, _ = np.histogram(ch_data.ConsumedkWh/ch_data.ParticipantCarkWh, 
                        bins=bins_soc)

f, axs = plt.subplots(2,1)
axs[0].bar((bins_kwh[1:]+bins_kwh[:-1])/2, hist_kwh/sum(hist_kwh), width=(bins_kwh[1]-bins_kwh[0]))
axs[0].set_xlabel('Charged Energy [kWh]')
axs[1].bar((bins_soc[1:]+bins_soc[:-1])/2, hist_soc/sum(hist_soc), width=(bins_soc[1]-bins_soc[0]))
axs[1].set_xlabel('Charged SOC [p.u.]')

# do it but separate small/medium/big
cars = {'EV < 15kWh': (0,15),
       '15kWh < EV < 30kWh': (15,30),
       '30kWh < EV < 50kWh': (30,50),
       '50 kWh < EV ': (50,200)}
for c, batts in cars.items():
    data = ch_data[(ch_data.ParticipantCarkWh > batts[0]) & 
                   (ch_data.ParticipantCarkWh<= batts[1])]
    
    hist_kwh, _ = np.histogram(data.ConsumedkWh, bins=bins_kwh)
    hist_soc, _ = np.histogram(data.ConsumedkWh/data.ParticipantCarkWh, 
                            bins=bins_soc)
    
    f, axs = plt.subplots(2,1)
    axs[0].bar((bins_kwh[1:]+bins_kwh[:-1])/2, hist_kwh/sum(hist_kwh), width=(bins_kwh[1]-bins_kwh[0]))
    axs[0].set_xlabel('Charged Energy [kWh]')
    k = 0.55
    axs[0].set_position([0.125, k, 0.9-0.125, 0.88-k])
    axs[1].bar((bins_soc[1:]+bins_soc[:-1])/2, hist_soc/sum(hist_soc), width=(bins_soc[1]-bins_soc[0]))
    axs[1].set_xlabel('Charged SOC [p.u.]')
    axs[1].set_position([0.125, 0.11, 0.9-0.125, 0.88-k])
    f.suptitle('Charge per session, {}'.format(c))
    
#%% Analyse number of charging sessions per day
bev_data['StartDate'] = bev_data.AdjustedStartTime.dt.date
nsessionsperday = bev_data.groupby(['ParticipantID', 'StartDate']).TransactionID.count()
nsessionsperday.hist(bins=np.arange(0,10,1))
avgsessperday = nsessionsperday.groupby('ParticipantID').mean()
n90 = int(avgsessperday.size * .9)
normalusers = avgsessperday.sort_values().index[:n90]
print('Share of one-charge days: ', round((nsessionsperday==1).mean()*100, 1), '%')
print('Share of one-charge days for 90% normal users: ', round((nsessionsperday[list(normalusers)]==1).mean()*100,1), '%')
