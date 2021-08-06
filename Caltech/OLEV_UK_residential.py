# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:45:46 2021

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util

# reading data
data = pd.read_csv(r'C:\Users\u546416\Downloads\electric-chargepoint-analysis-2017-raw-domestics-data.csv', engine='python')
# parsing dates
data.StartDate  = pd.to_datetime(data.StartDate)
data.EndDate  = pd.to_datetime(data.EndDate)
# extracting usefull info
data['dow'] = data.StartDate.dt.dayofweek
data['doy'] = data.StartDate.dt.dayofyear
data['Tini'] = data.StartTime.apply(lambda x: int(x[0:2]) + int(x[3:5])/60)
data['Tend'] = data.EndTime.apply(lambda x: int(x[0:2]) + int(x[3:5])/60)

# Sorting to User/Date/Time
data = data.sort_values(['CPID','StartDate','StartTime'])
# Adding datarows on day/night sessions
# id daysessions
data['DaySession'] = data.StartDate.eq(data.EndDate)
# id nighttime sessions (at least end date >= startdate + 1)
data['NightSession'] = data.StartDate.ne(data.EndDate)
# id sessions where next session starts same day than previous
data['SameDayNextSession']= data.EndDate.eq(data.StartDate.shift(-1))

# Some ratios
print('Day sessions: {:.1f}%'.format(data.DaySession.mean()*100))
print('Night sessions: {:.1f}%'.format(data.NightSession.mean()*100))
print('Day sessions that are followed by an ovn session {:.1f}%'.format((data.DaySession & data.SameDayNextSession).mean()/data.DaySession.mean()*100))

folder_output = r'c:\user\U546416\Pictures\UK_RES_OLEV\\'
bins = np.arange(0,24.1,0.25)

#Setting plt settings to small
font = {'size':8}
plt.rc('font', **font)


#%% Analyzing arrival departure

dataday = data[data.dow<4]
h,_,_ = np.histogram2d(dataday.Tini, dataday.Tend, bins=bins)

util.plot_arr_dep_hist(h, bins, ftitle='Monday to Thursday')
f = plt.gcf()
f.set_size_inches(7.47,3.3)
f.tight_layout()
plt.savefig(folder_output + 'ArrDep_monthu.pdf')
plt.savefig(folder_output + 'ArrDep_monthu.png')

for d in range(4,7):
    dataday = data[data.dow==d]
    h,_,_ = np.histogram2d(dataday.Tini, dataday.Tend, bins=bins)
    util.plot_arr_dep_hist(h, bins, ftitle=util.daysnames[d])
    f = plt.gcf()
    f.set_size_inches(7.47,3.3)
    f.tight_layout()
    plt.savefig(folder_output + 'ArrDep_{}.pdf'.format(util.daysnames[d]))
    plt.savefig(folder_output + 'ArrDep_{}.png'.format(util.daysnames[d]))



#%% Estimating charging frequency
    
startday = data.groupby('CPID').doy.min()
endday = data.groupby('CPID').doy.max()
numbersessions = data.CPID.value_counts()
batt = data.groupby('CPID').Energy.max()
driv_eff = 0.14 + 0.0009 * batt
en_session = data.groupby('CPID').Energy.mean() 
ch_eff = 0.9
soc_session = en_session * ch_eff/batt

minns = 10
minbatt = 10 #kWh

idxs = numbersessions[(numbersessions>minns) & (batt>minbatt)].index

freq = numbersessions / (endday+1-startday) * 7
daily_dist = en_session * freq / 7 / driv_eff * ch_eff

# Histogram of ch sessions
f=plt.figure()
freq[idxs].hist(bins=np.arange(0,12,0.5))
f.set_size_inches(3.5,3.3)

plt.ylabel('Users (Count)')
plt.xlabel('Weekly charging sessions')
f.tight_layout()

print('Mean charging sessions: {:.2f} per week'.format(freq.mean()))
print('Median charging sessions: {:.2f} per week'.format(freq.median()))
f.savefig(folder_output + 'weeklychsss.pdf')
f.savefig(folder_output + 'weeklychsss.png')

#%% Plot share night/day per week day

plt.figure()

dows = data.groupby('dow')
# Share of day sessions
dayshare = dows.DaySession.mean()
# Share of daily week sessions over a week
weekshare = dows.DaySession.count()/data.shape[0]

plt.bar(dayshare.index, (1-dayshare)*weekshare, bottom=dayshare*weekshare, label='Overnight charging')
plt.bar(dayshare.index, dayshare*weekshare, label='Daytime charging')
plt.xticks(dayshare.index, util.daysnames, rotation=45)
plt.legend(loc=4)
plt.gcf().set_size_inches(3.5,3.3)
plt.tight_layout()
plt.savefig(folder_output + 'sharedaynight.pdf')
plt.savefig(folder_output + 'sharedaynight.png')


#%% Plot share night/day per batt size

f,axs= plt.subplots(3,1)

sizes = [0,15,35,1000]
n = len(sizes)-1

nmsszs = ['B < {} kWh'.format(sizes[1])] + ['{} kWh < B < {} kWh'.format(sizes[i], sizes[i+1]) for i in range(1,n-1)] + ['{} kWh > B'.format(sizes[-2])]
for s in range(n):
    plt.sca(axs[s])
    bs = batt[(batt>sizes[s]) & (batt<=sizes[s+1])].index
    dows = data[data.CPID.isin()].groupby('dow')
    # Share of day sessions
    dayshare = dows.DaySession.mean()
    # Share of daily week sessions over a week
    weekshare = dows.DaySession.count()/dows.DaySession.count().sum()
    lo = 'Overnight charging' if s == 0 else '_'
    ld = 'Daytime charging' if s == 0 else '_'
    plt.bar(dayshare.index, (1-dayshare)*weekshare, bottom=dayshare*weekshare, label=lo)
    plt.bar(dayshare.index, dayshare*weekshare, label=ld)
    plt.xticks(dayshare.index, util.dsnms)
    plt.title(nmsszs[s])
    plt.grid(alpha=0.5)
f.legend(loc=9, ncol=2)
f.set_size_inches(3.5,3.3*(n-1))
plt.tight_layout()
for i,ax in enumerate(axs):
    dy = 0.01
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0-(dy*(len(axs)-i-1)), pos.width, pos.height-dy])
plt.savefig(folder_output + 'sharedaynight_bs.pdf')
plt.savefig(folder_output + 'sharedaynight_bs.png')

#%%Computing avg ch sessions and share of day/night per batt size
xds = []
xfr = []
for s in range(n):
    bs = batt[(batt>sizes[s]) & (batt<=sizes[s+1])].index
    dows = data[data.CPID.isin(bs)]
    # avg freq
    xfr.append(freq[bs].mean())
    # share of day sessions
    xds.append(dows.DaySession.mean())
#%% Plotting
plt.figure()
xfr = np.array(xfr)
xds = np.array(xds)
x=range(len(xds))

lo = 'Overnight charging'
ld = 'Daytime charging'

plt.barh(x, (1-xds)*xfr, left=xds*xfr, label=lo)
plt.barh(x, xds*xfr, label=ld)

plt.yticks(x, nmsszs)
f.legend(loc=9, ncol=2)
f.set_size_inches(3.5,3.3*(n-1))
plt.tight_layout()
##%%
#for i in dayshare.index:
#    d = data[data.dow==i]
#    print('Night sessions: {:.1f}%'.format(d.NightSession.mean()*100))
#    print('Day sessions that are followed by an ovn session {:.1f}%'.format((d.DaySession & d.SameDayNextSession).mean()/d.DaySession.mean()*100))


#%% Correcting for sessions way too long
# number of days
dmax = 20
maxplugin = data.groupby('CPID').PluginDuration.max()/24>dmax
datar = data[~data.CPID.isin(maxplugin[maxplugin>dmax].index)]

#%% Estimating charging frequency
    
startday = datar.groupby('CPID').doy.min()
endday = datar.groupby('CPID').doy.max()
numbersessions = datar.CPID.value_counts()
batt = datar.groupby('CPID').Energy.max()
driv_eff = 0.14 + 0.0009 * batt
en_session = datar.groupby('CPID').Energy.mean() 
ch_eff = 0.9
soc_session = en_session * ch_eff/batt

minns = 10
minbatt = 10 #kWh

idxs = numbersessions[(numbersessions>minns) & (batt>minbatt)].index

freq = numbersessions / (endday+1-startday) * 7
daily_dist = en_session * freq / 7 / driv_eff * ch_eff

# Histogram of ch sessions
f=plt.figure()
freq[idxs].hist(bins=np.arange(0,12,0.5))
f.set_size_inches(3.5,3.3)

plt.ylabel('Users (Count)')
plt.xlabel('Weekly charging sessions')
f.tight_layout()

print('Mean charging sessions: {:.2f} per week'.format(freq.mean()))
print('Median charging sessions: {:.2f} per week'.format(freq.median()))
f.savefig(folder_output + 'weeklychsss_corrected.pdf')
f.savefig(folder_output + 'weeklychsss_corrected.png')


#%% Do Charging indicators
### Scatter Weekly sessions vs. Energy per session And 
### Scatter Weekly sessions vs. Daily distance
f = plt.figure()
cmap = plt.get_cmap('viridis')
bmax = 100
cs = cmap(batt[idxs]/bmax)
mrkrsize = 5
alpha = 0.4

# creating axes for first scatter
dy = 0.08
yt = -0.25
f.add_axes([0.07,0.1+dy,0.37,0.85-dy])

plt.scatter(freq[idxs], en_session[idxs], color=cs, alpha=alpha, s=mrkrsize)
plt.xlabel('Weekly sessions')
plt.ylabel('Charged energy per session [kWh]')
plt.grid('--', alpha=0.5)
plt.xlim(0,12)
plt.ylim(0,70)
plt.gca().set_title('(a)', y=yt)

# Creating axes for second scatter
ax=f.add_axes([0.51,0.1+dy,0.37,0.85-dy])
plt.scatter(freq[idxs], daily_dist[idxs], color=cs, alpha=alpha, s=mrkrsize)
plt.xlabel('Weekly sessions')
plt.ylabel('Daily distance [km]')
plt.grid('--', alpha=0.5)
plt.xlim(0,12)
plt.ylim(0,160)
plt.gca().set_title('(b)', y=yt)

# Creating axes for colorbar
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap)
sm.set_array(np.arange(0,101,10))
cax = f.add_axes([0.9, 0.1+dy, 0.025, 0.85-dy])
cbar = plt.colorbar(mappable=sm, cax=cax)
cbar.set_label('Battery size [kWh]')
f.set_size_inches(7.47,3.3)#(11,4.76)

f.savefig(folder_output + 'charging_idicators_corr.pdf')
f.savefig(folder_output + 'charging_idicators_corr.png')

#%% Do 3d scatter
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
cmap = plt.get_cmap('viridis')
cs =  cmap(batt[idxs]/bmax)

ax.scatter(freq[idxs], soc_session[idxs], daily_dist[idxs], color=cs)

ax.set_xlim(0,12)

plt.xlabel('Weekly sessions')
plt.ylabel('Charged SOC per session')
ax.set_zlabel('Daily distance [km]')

#%% Plot share night/day per week day

plt.figure()

dows = data.groupby('dow')
# Share of day sessions
dayshare = dows.DaySession.mean()
# Share of daily week sessions over a week
weekshare = dows.DaySession.count()/data.shape[0]

plt.bar(dayshare.index, (1-dayshare)*weekshare, bottom=dayshare*weekshare, label='Overnight charging')
plt.bar(dayshare.index, dayshare*weekshare, label='Daytime charging')
plt.xticks(dayshare.index, util.daysnames, rotation=45)
plt.legend(loc=4)
plt.gcf().set_size_inches(3.5,3.3)
plt.tight_layout()
plt.savefig(folder_output + 'sharedaynight_corrbat.pdf')
plt.savefig(folder_output + 'sharedaynight_corrbat.png')


#%% Plot share night/day per batt size

f,axs= plt.subplots(3,1)

sizes = [0,15,35,1000]
n = len(sizes)-1

nmsszs = ['B < {} kWh'.format(sizes[1])] + ['{} kWh < B < {} kWh'.format(sizes[i], sizes[i+1]) for i in range(1,n-1)] + ['{} kWh > B'.format(sizes[-2])]
for s in range(n):
    plt.sca(axs[s])    
    bs = batt[(batt>sizes[s]) & (batt<=sizes[s+1])].index
    dows = datar[datar.CPID.isin(bs)].groupby('dow')
    # Share of day sessions
    dayshare = dows.DaySession.mean()
    # Share of daily week sessions over a week
    weekshare = dows.DaySession.count()/dows.DaySession.count().sum()
    lo = 'Overnight charging' if s == 0 else '_'
    ld = 'Daytime charging' if s == 0 else '_'
    plt.bar(dayshare.index, (1-dayshare)*weekshare, bottom=dayshare*weekshare, label=lo)
    plt.bar(dayshare.index, dayshare*weekshare, label=ld)
    plt.xticks(dayshare.index, util.dsnms)
    plt.title(nmsszs[s])
    plt.grid(alpha=0.5)
f.legend(loc=8, ncol=2)
f.set_size_inches(3.5,3.3*(n-1))
plt.tight_layout()
for i,ax in enumerate(axs):
    dy = 0.01
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0+(dy*i),pos.width, pos.height-dy])  #(len(axs)-i-1)), 
plt.savefig(folder_output + 'sharedaynight_bs_corrbat.pdf')
plt.savefig(folder_output + 'sharedaynight_bs_corrbat.png')
#%% Computing avg ch sessions and share of day/night per batt size
xds = []
xfr = []
for s in range(n):
    bs = batt[(batt>sizes[s]) & (batt<=sizes[s+1])].index
    dows = data[data.CPID.isin(bs)]
    # avg freq
    xfr.append(freq[bs].mean())
    # share of day sessions
    xds.append(dows.DaySession.mean())
#%% Plotting
plt.figure()
xfr = np.array(xfr)
xds = np.array(xds)
x=range(len(xds))

lo = 'Overnight charging'
ld = 'Daytime charging'

plt.barh(x, (1-xds)*xfr, left=xds*xfr, label=lo)
plt.barh(x, xds*xfr, label=ld)

plt.yticks(x, nmsszs)
f.legend(loc=9, ncol=2)
f.set_size_inches(3.5,3.3*(n-1))
plt.tight_layout()
##%%