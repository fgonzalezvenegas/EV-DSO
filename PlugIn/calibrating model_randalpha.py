# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 05:26:24 2020

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EVmodel
    

## Simulation parameters
nweeks = 10
ndays = nweeks*7
step = 60

nevs = 1000

batt = dict(allevs=50,
            allevs40=46,
            small=28, 
            big=79)

# KPIs from Electric Nation
chsmean = dict(allevs=0.4469,
               allevs40=0.4469,
               small=0.4731,
               big=0.3986)
chsmedian = dict(allevs=0.3891,
                 allevs40=0.3891,
                 small=0.4280,
                 big=0.3270)
#distmean = dict(allevs=41.22,
#                allevs40=41.22,
#                small=36.67, 
#                big=52.08)
#distmedian = dict(allevs=33.66,
#                  allevs40=33.66,
#                small=36.20, 
#                big=47.86)
distmean = dict(allevs=38.09,
                allevs40=38.09,
                small=33.28, 
                big=46.95)
distmedian = dict(allevs=32.6,
                  allevs40=32.6,
                small=29.7, 
                big=43.6)


driving_eff = {k : (14+0.09*b)/100 for k, b in batt.items()}

#%% Create grid and Simulate
# creating grid and evs
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=0, batt_size=50)
ds = [float(ev.dist_wd) for ev in evs]
dmean = np.mean(ds)*2
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

ch_sessions = {}

# Simulate for different combinations
for idx in batt.keys():
    print('Calibrating for {}'.format(idx))
    # update ev values of batt sizes and driving effs
    grid.set_evs_param(param='batt_size', value=batt[idx])
    grid.set_evs_param(param='driving_eff', value=driving_eff[idx])
    chsmean_test = {}
    chsmedian_test = {}
    for i, ev in enumerate(evs):
        ev.dist_wd = ds[i] * distmean[idx]/dmean
        ev.dist_we = ds[i] * distmean[idx]/dmean
    di = [ev.dist_wd for ev in evs]
    print('\tEV distances, mean {:.1f}, median {:.1f}'.format(np.mean(di)*2, np.median(di)*2))
    
    # array of alphas to test, between 1/100 (almost never) to 100 (always)
    ns = np.logspace(np.log10(1),np.log10(100),11)
    ns = np.sort(np.array([1/n for n in ns] + list(ns)))
    k=True
    for n in ns:
        if k:
            print('\t\talpha {:.2f}'.format(n))
        k=not k
        grid.reset()
        alphs = np.random.lognormal(mean=np.log(n), sigma=1, size=nevs)
        for i, ev in enumerate(evs):
            ev.n_if_needed = alphs[i]
        grid.do_days()
        chsmean_test[n] = np.array([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                                    for ev in evs]).mean()/ndays
        chsmedian_test[n] = np.median(np.array([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                                    for ev in evs])/ndays)
    chsmean_test = pd.Series(chsmean_test)
    chsmedian_test=  pd.Series(chsmedian_test)
    ch_sessions[idx, 'mean'] = chsmean_test
    ch_sessions[idx, 'median'] = chsmedian_test

#%% save data:
output_folder = r'c:\user\U546416\Documents\PhD\Data\Plug-in Model\Calibration\\'
df = pd.DataFrame(ch_sessions)
df.index.name = 'alpha'
df.to_csv(output_folder + 'ch_sessions_randalpha.csv')


#%% Plot based on mean charging sessions
alphas = {}
for idx in batt.keys():
    plt.figure()
    data = ch_sessions[idx, 'mean']
    plt.plot(data.index, data, '-x', label='Mean - EV model')
    plt.gca().set_xscale('log')
#    plt.plot(chsmean_test.index, chsmedian_test, label='median')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Daily charging sessions')
    plt.ylim(0,1.1)
    plt.xlim(0.01,100)
    plt.title(idx)
    x1 = data[data>chsmean[idx]].index[0]
    x0 = data[data<=chsmean[idx]].index[-1]
    dx = np.log10(x1)-np.log10(x0)
#    dx = x1-x0
    y0 = data[x0]
    y1 = data[x1]
    yk = chsmean[idx]
#    xx = x0 + dx * (yk-y0)/(y1-y0)
    xx = np.exp((np.log10(x0) + dx * (yk-y0)/(y1-y0)) / np.log10(np.e))
    plt.axhline(y=yk, linestyle='--', color='darkblue', label='Mean - Electric Nation')
    plt.axvline(x=xx, linestyle='--', color='r',label='Selected ' + r'$\alpha$' + ': {:.1f}'.format(xx))
    plt.legend()
    alphas['mean', idx] = xx
    
#%%  Plot based on median charging sessions
for idx in batt.keys():
    plt.figure()
    data = ch_sessions[idx, 'median']
    plt.plot(data.index, data, '-x', label='Median - EV model')
    plt.gca().set_xscale('log')
#    plt.plot(chsmean_test.index, chsmedian_test, label='median')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Daily charging sessions')
    plt.ylim(0,1.1)
    plt.xlim(0.01,100)
    plt.title(idx)
    x1 = data[data>chsmedian[idx]].index[0]
    x0 = data[data<=chsmedian[idx]].index[-1]
    dx = np.log10(x1)-np.log10(x0)
#    dx = x1-x0
    y0 = data[x0]
    y1 = data[x1]
    yk = chsmedian[idx]
#    xx = x0 + dx * (yk-y0)/(y1-y0)
    xx = np.exp((np.log10(x0) + dx * (yk-y0)/(y1-y0)) / np.log10(np.e))
    plt.axhline(y=yk, linestyle='--', color='darkblue', label='Median - Electric Nation')
    plt.axvline(x=xx, linestyle='--', color='r',label='Selected ' + r'$\alpha$' + ': {:.1f}'.format(xx))
    plt.legend()
    alphas['median', idx] = xx
#%% One graph, mean   
plt.figure()
idxs = ['small', 'allevs40', 'big']
strs = {'small':'Small', 'allevs40':'Average', 'big':'Large'}
colors = ['darkblue', 'darkgreen', 'orange']
for i, idx in enumerate(idxs):    
    data = df[idx, 'mean']
    plt.plot(data.index, data*7, '-', label='EV Model - {} BEVs'.format(strs[idx]), alpha=0.8, color=colors[i])
    plt.gca().set_xscale('log')
#    plt.plot(chsmean_test.index, chsmedian_test, label='median')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Weekly sessions')
    plt.ylim(0,8)
    plt.xlim(0.01,100)
    x1 = data[data>chsmean[idx]].index[0]
    x0 = data[data<=chsmean[idx]].index[-1]
    dx = np.log10(x1)-np.log10(x0)
#    dx = x1-x0
    y0 = data[x0]*7
    y1 = data[x1]*7
    yk = chsmean[idx]*7
#    xx = x0 + dx * (yk-y0)/(y1-y0)
    xx = np.exp((np.log10(x0) + dx * (yk-y0)/(y1-y0)) / np.log10(np.e))
    plt.plot(xx, yk, '*', color=colors[i], label='{} BEV '.format(strs[idx])+ r'$\alpha$' +  ': {:.2f}'.format(xx) )
    plt.axhline(y=yk, linestyle='--', color='grey', label='_', alpha=0.2)
    plt.axvline(x=xx, linestyle='--', color='grey',label='_', alpha=0.2)
    plt.legend()
#%%
alphas = pd.DataFrame(alphas, index=['alpha']).T
alphas.index.names = ['ind', 'case']
alphas.to_csv(output_folder + 'selected_alphas_randalpha.csv', )
#alphas = pd.read_csv(output_folder + 'selected_alphas_equalwewds.csv', engine='python', index_col=[0,1])
#%% Do simulation for selected alpha and plot users indicators:
ind = 'mean'
grid = EVmodel.Grid(ndays=int(ndays), step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=0, batt_size=50)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

ch_sessions = {}
en_session = {}
daily_dist = {}
for (i, idx), alphadata in alphas.iterrows():
    if i != ind:
        continue
    if idx == 'allevs':
        continue
    alpha = alphadata['alpha']
    print('Doing simulation for case {}, alpha={:.2f}'.format(idx, alpha))
    # update ev values of batt sizes and driving effs
    grid.set_evs_param(param='batt_size', value=batt[idx])
    grid.set_evs_param(param='driving_eff', value=driving_eff[idx])
    for i, ev in enumerate(evs):
        ev.dist_wd = ds[i] * distmean[idx]/distmean['allevs']
#        ev.dist_we = ds[i] * distmean[idx]/distmean['allevs']
    di = [float((ev.dist_wd*5 + ev.dist_we*2)/7) for ev in evs]
    daily_dist[idx] = np.array(di)*2
    print('\tEV distances, mean {:.1f}, median {:.1f}'.format(np.mean(di)*2, np.median(di)*2))
    grid.reset()
    grid.set_evs_param(param='n_if_needed', value=alpha)
    grid.do_days()
    ch_sessions[idx] = np.array([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                                    for ev in evs])/grid.ndays
    en_session[idx] = np.array([ev.charged_energy.sum()/ev.n_plugs for ev in evs])

ch_sessions = pd.DataFrame(ch_sessions)
en_session = pd.DataFrame(en_session)
daily_dist = pd.DataFrame(daily_dist)
#%% Plot scatters to show results from simulation
cmap = plt.get_cmap('viridis')

#idx = 'allevs40'
#f,ax = plt.subplots(1,2)
#color = cmap(batt[idx]/100)
#plt.sca(ax[0])
#plt.scatter(ch_sessions[idx]*7, en_session[idx], color=color)
#plt.sca(ax[1])
#plt.scatter(ch_sessions[idx]*7, daily_dist[idx], color=color)
#plt.sca(ax[1])

colorsLR = {'big': 'orange', 'allevs40': 'darkgreen', 'small': 'darkblue'}
f,ax = plt.subplots(1,2)
f.set_size_inches(11,4.76)

n = 1000
evpl = daily_dist[daily_dist.allevs40<120].iloc[0:n].index

for j, idx in enumerate(['small', 'allevs40', 'big' ]):
    color = cmap(batt[idx]/100)
    plt.sca(ax[0])
    plt.scatter(ch_sessions[idx][evpl]*7, en_session[idx][evpl], color=color, label='{} BEVs'.format(strs[idx]), alpha=0.3)
    plt.xlabel('Weekly sessions')
    plt.ylabel('Charged energy per session [kWh]')
    plt.grid('--', alpha=0.5)
    plt.xlim(0,12)
    plt.ylim(0,70)
    plt.legend()
    plt.sca(ax[1])
    plt.scatter(ch_sessions[idx][evpl]*7, daily_dist[idx][evpl], color=color, label='{} BEVs'.format(strs[idx]), alpha=0.3)
    plt.xlabel('Weekly sessions')
    plt.ylabel('Daily distance [km]')
    plt.grid('--', alpha=0.5)
    plt.xlim(0,12)
    plt.ylim(0,160)
    import statsmodels.api as sm
    ## adding Linear regression for small and large EVs
    lr = sm.OLS(pd.Series(daily_dist[idx]), sm.add_constant(pd.Series(ch_sessions[idx]*7))).fit()
    x = np.arange(0,15,1)
    X = sm.add_constant(x)
    pred = lr.get_prediction(X)
    plt.plot(x, pred.predicted_mean, color=colorsLR[idx], alpha=0.7, zorder=0)
    plt.fill_between(x, pred.conf_int()[:,0], pred.conf_int()[:,1], color=colorsLR[idx], alpha=0.2)
    #
    rsq=lr.rsquared
    
    # adding regression values string
    dx, dy = -0.2, +0.5
    yb = 100+20*j
    xs = 12 - 3*j
    eq = 'y={:.1f}x+{:.1f}\n'.format(lr.params.iloc[1], lr.params.iloc[0])
    r2 = r'$r^2$={:.2f}'.format(rsq)
    plt.text(xs+dx,yb+dy,
             eq + r2, 
             horizontalalignment='right')
    print(idx, '\t', eq[:-1], r2)
    plt.legend(loc=4)
    f.tight_layout()
#%% save results
ch_sessions.to_csv(output_folder + 'ch_sessions_simu_fixedalpha.csv')
en_session.to_csv(output_folder + 'en_session_simu_fixedalpha.csv')
daily_dist.to_csv(output_folder + 'daily_dist_simu_fixedalpha.csv')

#%% Try grid with random alpha
    
    #### THIS IS WEIRDLY GOOOOOD ####
      
ind = 'mean'
grid = EVmodel.Grid(ndays=int(ndays), step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=0, batt_size=50)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

ch_sessions = {}
en_session = {}
daily_dist = {}
for (i, idx), alphadata in alphas.iterrows():
    if i != ind:
        continue
    if idx == 'allevs':
        continue
    alpha = alphadata['alpha']
    alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
    print('Doing simulation for case {}, alpha={:.2f}'.format(idx, alpha))
    # update ev values of batt sizes and driving effs
    grid.set_evs_param(param='batt_size', value=batt[idx])
    grid.set_evs_param(param='driving_eff', value=driving_eff[idx])
    for i, ev in enumerate(evs):
        ev.dist_wd = ds[i] * distmean[idx]/distmean['allevs']
        ev.n_if_needed = alphs[i]
#        ev.dist_we = ds[i] * distmean[idx]/distmean['allevs']
    di = [float((ev.dist_wd*5 + ev.dist_we*2)/7) for ev in evs]
    daily_dist[idx] = np.array(di)*2
    print('\tEV distances, mean {:.1f}, median {:.1f}'.format(np.mean(di)*2, np.median(di)*2))
    grid.reset()
#    grid.set_evs_param(param='n_if_needed', value=alpha)
    grid.do_days()
    ch_sessions[idx] = np.array([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                                    for ev in evs])/grid.ndays
    en_session[idx] = np.array([ev.charged_energy.sum()/ev.n_plugs for ev in evs])

ch_sessions = pd.DataFrame(ch_sessions)
en_session = pd.DataFrame(en_session)
daily_dist = pd.DataFrame(daily_dist)
soc_session = en_session / np.array([batt[i] for i in en_session])

#%% save results
#ch_sessions.to_csv(output_folder + 'ch_sessions_simu_randalpha.csv')
#en_session.to_csv(output_folder + 'en_session_simu_randalpha.csv')
#daily_dist.to_csv(output_folder + 'daily_dist_simu_randalpha.csv')
#
ch_sessions = pd.read_csv(output_folder + 'ch_sessions_simu_randalpha.csv',
                          engine='python', index_col=0)
en_session = pd.read_csv(output_folder + 'en_session_simu_randalpha.csv',
                          engine='python', index_col=0)
daily_dist = pd.read_csv(output_folder + 'daily_dist_simu_randalpha.csv',
                          engine='python', index_col=0)
#%% Plot scatters to show results from simulation
cmap = plt.get_cmap('viridis')

#idx = 'allevs40'
#f,ax = plt.subplots(1,2)
#color = cmap(batt[idx]/100)
#plt.sca(ax[0])
#plt.scatter(ch_sessions[idx]*7, en_session[idx], color=color)
#plt.sca(ax[1])
#plt.scatter(ch_sessions[idx]*7, daily_dist[idx], color=color)
#plt.sca(ax[1])

colorsLR = {'big': 'orange', 'allevs40': 'darkgreen', 'small': 'darkblue'}
f,ax = plt.subplots(1,2)
f.set_size_inches(11,4.76)

n = 1000
a = 0.3
dmax = 140
evpl = daily_dist[daily_dist.allevs40<dmax].iloc[0:n].index
xs = [12,11,9]
ys = [70,110,140]

for j, idx in enumerate(['small', 'allevs40', 'big' ]):
    color = cmap(batt[idx]/100)
    plt.sca(ax[0])
    plt.scatter(ch_sessions[idx][evpl]*7, en_session[idx][evpl], color=color, label='{} BEVs'.format(strs[idx]), alpha=a)
    plt.xlabel('Weekly sessions')
    plt.ylabel('Charged energy per session [kWh]')
    plt.grid('--', alpha=0.5)
    plt.xlim(0,12)
    plt.ylim(0,70)
    plt.legend()
    plt.sca(ax[1])
    plt.scatter(ch_sessions[idx][evpl]*7, daily_dist[idx][evpl], color=color, label='{} BEVs'.format(strs[idx]), alpha=a)
    plt.xlabel('Weekly sessions')
    plt.ylabel('Daily distance [km]')
    plt.grid('--', alpha=0.5)
    plt.xlim(0,12)
    plt.ylim(0,160)
    import statsmodels.api as sm
    ## adding Linear regression for small and large EVs
    lr = sm.OLS(pd.Series(daily_dist[idx][evpl]), sm.add_constant(pd.Series(ch_sessions[idx][evpl]*7))).fit()
    x = np.arange(0,15,1)
    X = sm.add_constant(x)
    pred = lr.get_prediction(X)
    plt.plot(x, pred.predicted_mean, color=colorsLR[idx], alpha=0.7, zorder=0)
    plt.fill_between(x, pred.conf_int()[:,0], pred.conf_int()[:,1], color=colorsLR[idx], alpha=0.2)
    #
    rsq=lr.rsquared
    
    # adding regression values string
    eq = 'y={:.1f}x+{:.1f}\n'.format(lr.params.iloc[1], lr.params.iloc[0])
    r2 = r'$r^2$={:.2f}'.format(rsq)
    plt.text(xs[j], ys[j],
             eq + r2, 
             horizontalalignment='right')
    print(idx, '\t', eq[:-1], r2)
    plt.legend(loc=4)
    f.tight_layout()