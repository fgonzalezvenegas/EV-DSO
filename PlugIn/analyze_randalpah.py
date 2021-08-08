# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 20:57:15 2020
Run EV model to analyze charging sessions, peak load and flex factor
@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import EVmodel
import time

def charging_sessions(grid, key=None, stats=True):
    # compute hist
    nsessions = np.asarray([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                            for ev in grid.get_evs(key)])/grid.ndays * 7
    h_bins = np.append(np.arange(0,8,1), 100)
    hs = np.histogram(nsessions, h_bins)
    if stats:
        return hs[0]/sum(hs[0]), np.mean(nsessions), np.median(nsessions)   
    return hs[0]/sum(hs[0])

def charging_sessions_perc(grid, key=None, perc=[0,10,90,100]):
    # compute hist
    nsessions = np.asarray([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                            for ev in grid.get_evs(key)])/grid.ndays * 7
    return np.percentile(nsessions, perc)

def peak_to_series_and_save(peak, namesave='peakload.csv', 
                            name='Peak_load', 
                            indexnames=['Batt_size', 'Charger_power', 'alpha', 'fleet_size']):
    peak = pd.Series(peak)
    peak.index.names = indexnames
    peak.name = name
    peak.to_csv(folder_outputs + namesave, header=True)
    print('Saved data {}'.format(namesave))
    return peak

folder_outputs = r'c:\user\U546416\Documents\PhD\Data\Plug-in Model\Results\RandAlpha\\'
folder_images = r'c:\user\U546416\Pictures\PlugInModel\RandAlpha\\'
#%% Simulation to get ch_sessions per week
## Simulation parameters
nweeks = 5
ndays = nweeks*7
step = 60

nevs = 1000

# battery sizes
batts = np.arange(15,100.1,2.5)
#batts = np.arange(15,100,40)
# driving efficiency
driving_eff = ((batts * 0.09)+14)/100
# n_if_needed
n_if_needed = [0.5, 1, 0.66, 1.31, 2.62, 3.4]

# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

mean = {}
median= {}
hists = {}
#dec = {}

for alpha in n_if_needed:
    print('\tAlpha {}'.format(alpha))
    alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
    for i, ev in enumerate(evs):
        ev.n_if_needed = alphs[i]
#            grid.set_evs_param(param='n_if_needed', value=n)
    for i, b in enumerate(batts):
        print('\t\tBatt {} kWh'.format(b))
        grid.set_evs_param(param='batt_size', value=batts[i])
        grid.set_evs_param(param='driving_eff', value=driving_eff[i])
        grid.reset()
        grid.do_days()
        hists[alpha, b], mean[alpha, b], median[alpha, b] = charging_sessions(grid, stats=True)
#        dec[alpha, b] 
        
hists = pd.DataFrame(hists).T
hists.index.names=['alpha', 'Batt_size']
stats = pd.DataFrame(mean, index=['Mean']).T
stats['Median'] = median.values()
stats.index.names=['alpha', 'Batt_size']

#folder_save = r'c:\user\U546416\Pictures\ElectricNation\PlugIn\Tests\\'
#n = 'var' if type(n_if_needed)==list else n_if_needed
hists.to_csv(folder_outputs + 'hists_randalpha.csv')
stats.to_csv(folder_outputs + 'stats_randalpha.csv')

#%% Do Histogram of charging sessions
hists = pd.read_csv(folder_outputs + 'hists_randalpha.csv', index_col=[0,1])
labels=['{} to {} per week'.format(i,i+1) for i in range(7)]
labels.append('More than 7')
for a in hists.index.levels[0]:
    #
    f, ax = plt.subplots()
    ax.stackplot(batts, hists.loc[a,:].T*100, labels=labels, alpha=0.8)    
    ax.set_xlim([15,97])
    ax.set_ylim([0,100])
    ax.set_xlabel('Battery size [kWh]')
    ax.set_ylabel('Share of EVs')
    #ax.axvline(50, color='k', linestyle='--')
    #ax.text(x=51, y=30, s='Peugeot e208')
    plt.legend(loc=4)
    plt.savefig(folder_images + 'charging_sessions_alpha{}.pdf'.format(a))
    plt.savefig(folder_images + 'charging_sessions_alpha{}.png'.format(a))
    ax.set_title(r'Charging sessions per week (cumulated), $\alpha$ {}'.format(a))

#%% Do hisogram of ch sessions, one fig in 3 plots
hists = pd.read_csv(folder_outputs + 'hists_randalpha.csv', index_col=[0,1])
stats = pd.read_csv(folder_outputs + 'stats_randalpha.csv', index_col=[0,1])


alphas = [0.5,1.31,3.4]
titles = ['Low plug-in', 'Average plug-in', 'High plug-in']
f, axs = plt.subplots(1,3)
labels=['{} to {} per week'.format(i,i+1) for i in range(7)]
labels.append('More than 7')

for i, ax in enumerate(axs):
    if i== 0:
        ax.stackplot(batts, hists.loc[alphas[i],:].T*100, labels=labels, alpha=0.8)    
    else:
        ax.stackplot(batts, hists.loc[alphas[i],:].T*100, alpha=0.8)        
    ax.set_xlim([15,97])
    ax.set_ylim([0,100])
    ax.set_xlabel('Battery size [kWh]')
    ax.set_ylabel('Share of EVs')
    ax.set_title(titles[i])
    
#    secax = ax.secondary_yaxis('right') 
#    secax.plot(batts, stats['Mean'].loc[alphas[i],:], label='Mean', color='k', linestyle='--')
f.legend(loc=8, ncol=4) 
f.set_size_inches(11,4.76)   
f.tight_layout()
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.1
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy])
plt.savefig(folder_images + 'charging_sesions_vcor.pdf')
plt.savefig(folder_images + 'charging_sesions_vcor.png')

#%% Charging sessions for Urban/Rural cases
## Simulation parameters
nweeks = 5
ndays = nweeks*7
step = 60

nevs = 1000

# battery sizes
batts = np.arange(15,100.1,2.5)
#batts = np.arange(15,100,40)
# driving efficiency
driving_eff = ((batts * 0.09)+14)/100
# n_if_needed
n_if_needed = [1.31]

# histograms of driving 
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=[0,1,2,3,4])
#havg = (hhome * np.arange(1,101,2)).sum(axis=1)/hhome.sum(axis=1)
#havg.dropna(inplace=True)
idx = pd.IndexSlice
# Rural communes of Seine et Marne (Paris peri urban areas)
hrural = hhome.loc[idx[:,:,'R',:,77],:].sum(axis=0)/hhome.loc[idx[:,:,'R',:,77],:].sum().sum()
# Urban communes of Paris
hurban = hhome.loc[idx[:,:,'C',:,75],:].sum(axis=0)/hhome.loc[idx[:,:,'C',:,75],:].sum().sum()
#if 'ZE' in hhome.columns:
#    hhome = hhome.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
percentiles = [i for i in range(0,105,5)]



# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('base', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50)
evsurban = grid.add_evs('urban', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50,
                 dist_wd=dict(cdf=hurban.cumsum().values))
evsrural = grid.add_evs('rural', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50,
                 dist_wd=dict(cdf=hrural.cumsum().values))
ds = [float(ev.dist_wd) for ev in evs]
dsu = [float(ev.dist_wd) for ev in evsurban]
dsr = [float(ev.dist_wd) for ev in evsrural]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))
print('Urban EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(dsu)*2, np.median(ds)*2))
print('Rural EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(dsr)*2, np.median(ds)*2))

mean = {}
median= {}
hists = {}
percs = {}
#dec = {}
cases = ['base', 'urban', 'rural']
# urban
for alpha in n_if_needed:
    print('\tAlpha {}'.format(alpha))
    alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
    for i, ev in enumerate(evs):
        ev.n_if_needed = alphs[i]
    for i, ev in enumerate(evsurban):
        ev.n_if_needed = alphs[i]
    for i, ev in enumerate(evsrural):
        ev.n_if_needed = alphs[i]
#            grid.set_evs_param(param='n_if_needed', value=n)
    for i, b in enumerate(batts):
        print('\t\tBatt {} kWh'.format(b))
        grid.set_evs_param(param='batt_size', value=batts[i])
        grid.set_evs_param(param='driving_eff', value=driving_eff[i])
        grid.reset()
        grid.do_days()
        for k in cases:
            hists[k, alpha, b], mean[k, alpha, b], median[k, alpha, b] = charging_sessions(grid, key=k, stats=True)
            percs[k, alpha, b] = charging_sessions_perc(grid, key=k, perc=percentiles)
#        dec[alpha, b] 
    
#hists = peak_to_series_and_save(hists, namesave='hists_randalpha_urbanrural.csv',name=range(8),
                               
hists = pd.DataFrame(hists).T
hists.index.names=['case', 'alpha', 'Batt_size']
stats = pd.DataFrame(mean, index=['Mean']).T
stats['Median'] = median.values()
stats.index.names=['case', 'alpha', 'Batt_size']
percs = pd.DataFrame(percs).T
percs.columns = percentiles
percs.index.names=['case', 'alpha', 'Batt_size']

#folder_save = r'c:\user\U546416\Pictures\ElectricNation\PlugIn\Tests\\'
#n = 'var' if type(n_if_needed)==list else n_if_needed
hists.to_csv(folder_outputs + 'hists_randalpha_urbanrural.csv')
stats.to_csv(folder_outputs + 'stats_randalpha_urbanrural.csv')
percs.to_csv(folder_outputs + 'percs_randalpha_urbanrural.csv')

#%% Do one fig in 3 plots for urban, & rural
hists = pd.read_csv(folder_outputs + 'hists_randalpha_urbanrural.csv', index_col=[0,1,2])
stats = pd.read_csv(folder_outputs + 'stats_randalpha_urbanrural.csv', index_col=[0,1,2])


alpha = 1.31
cases = ['urban', 'base', 'rural']
titles = ['Urban', 'Average', 'Rural']
f, axs = plt.subplots(1,3)
labels=['{} to {} per week'.format(i,i+1) for i in range(7)]
labels.append('More than 7')
batts = np.arange(15,100.1,2.5)


for i, ax in enumerate(axs):
    if i== 0:
        ax.stackplot(batts, hists.loc[cases[i],alpha,:].T*100, labels=labels, alpha=0.8)    
    else:
        ax.stackplot(batts, hists.loc[cases[i],alpha,:].T*100, alpha=0.8)        
    ax.set_xlim([15,97])
    ax.set_ylim([0,100])
    ax.set_xlabel('Battery size [kWh]')
    ax.set_ylabel('Share of EVs')
    ax.set_title(titles[i])
    
#    secax = ax.secondary_yaxis('right') 
#    secax.plot(batts, stats['Mean'].loc[alphas[i],:], label='Mean', color='k', linestyle='--')
handles, labels = axs[0].get_legend_handles_labels()
# reordering legends
order = [0,4,1,5,2,6,3,7]
hs = [handles[o] for o in order]
ls = [labels[o] for o in order]
f.legend(hs, ls, loc=8, ncol=4) 
f.set_size_inches(11,4.76)   
f.tight_layout()
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.1
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy])
plt.savefig(folder_images + 'charging_sesions_urbanrural.pdf')
plt.savefig(folder_images + 'charging_sesions_urbanrural.png')

#%% Do precentiles in one fig in 3 plots for urban, & rural
percs = pd.read_csv(folder_outputs + 'percs_randalpha_urbanrural.csv', index_col=[0,1,2])
stats = pd.read_csv(folder_outputs + 'stats_randalpha_urbanrural.csv', index_col=[0,1,2])

alpha = 1.31
cases = ['urban', 'base', 'rural']
titles = ['Urban', 'Average', 'Rural']
abc = ['(a)','(b)','(c)']
f, axs = plt.subplots(1,3, sharey=True)
batts = np.arange(15,100.1,2.5)

percentiles = [5,15,25,50]#35,45]

labels=['{:2d}%-{:2d}%'.format(p,100-p) for p in percentiles]

alpha_plot=0.15
color = 'b'
colormean = 'k'
colormedian = 'b'

for i, ax in enumerate(axs):
    #plot mean value
    if i==0:
        ax.plot(batts, stats.Mean.loc[cases[i], alpha,:], color=colormean, label='Mean', linestyle='--')
        ax.plot(batts, percs['50'].loc[cases[i], alpha,:], color=colormedian, label='Median')
    else:
        ax.plot(batts, stats.Mean.loc[cases[i], alpha,:], color=colormean, label='_', linestyle='--')
        ax.plot(batts, percs['50'].loc[cases[i], alpha,:], color=colormedian, label='Median')
    # plot percentiles
    for j, p in enumerate(percentiles[:-1]):
        if i== 0:
            ax.fill_between(batts, percs.loc[cases[i],alpha,:][str(p)], percs.loc[cases[i],alpha,:][str(percentiles[j+1])], 
                            facecolor=color, alpha=alpha_plot*(j+1), label=labels[j])
            ax.fill_between(batts, percs.loc[cases[i],alpha,:][str(100-percentiles[j])], percs.loc[cases[i],alpha,:][str(100-percentiles[j+1])], 
                            facecolor=color, alpha=alpha_plot*(j+1), label='_')##, alpha=alpha_plot)
        else:
            ax.fill_between(batts, percs.loc[cases[i],alpha,:][str(p)], percs.loc[cases[i],alpha,:][str(percentiles[j+1])], 
                            label='_', facecolor=color, alpha=alpha_plot*(j+1))
            ax.fill_between(batts, percs.loc[cases[i],alpha,:][str(100-percentiles[j])], percs.loc[cases[i],alpha,:][str(100-percentiles[j+1])], 
                            label='_', facecolor=color, alpha=alpha_plot*(j+1))##, alpha=alpha_plot)
    ax.set_xlim([15,100])
    ax.set_ylim([0,9])
    ax.set_xlabel('Battery size [kWh]')
    if i==0:
        ax.set_ylabel('Weekly sessions')
    
    ax.grid(linestyle='--')
#    ax.text(x=(batts[0]+batts[-1])/2, y=8.3, 
#            s=titles[i],horizontalalignment='center',fontweight='bold')
    ax.set_title('{} {}'.format(abc[i], titles[i]), y=-0.25)
#    secax = ax.secondary_yaxis('right') 
#    secax.plot(batts, stats['Mean'].loc[alphas[i],:], label='Mean', color='k', linestyle='--')
handles, labels = axs[0].get_legend_handles_labels()
# reordering legends
#order = [0,4,1,5,2,6,3,7]
#hs = [handles[o] for o in order]
#ls = [labels[o] for o in order]
#f.legend(hs, ls, loc=8, ncol=4)

f.legend(loc=9,ncol=5)
f.set_size_inches(9,4) #11,3.8  
f.tight_layout()
## resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0
    dyu = 0.06
    dx = 0 #0.033
    ax.set_position([pos.x0-dx*i, pos.y0+dy, pos.width-dx, pos.height-dy-dyu])

f.tight_layout()
## resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0
    dyu = 0.06
    dx = 0 #0.033
    ax.set_position([pos.x0-dx*i, pos.y0+dy, pos.width-dx, pos.height-dy-dyu])
plt.savefig(folder_images + 'charging_sesions_urbanruralv3.pdf')
plt.savefig(folder_images + 'charging_sesions_urbanruralv3.png')

    
#%% Identifying peak load
t = [time.time()]
## Simulation parameters
nweeks = 12
ndays = nweeks*7
step = 30

nevs = 10000

# EV PARAMS:
# Battery
batts = np.array((25,50,75))
# driving eff
driving_eff = ((batts * 0.09)+14)/100
# charging power [kw]
pcharger = (3.6,7.2,10.7)
# fleet sizes to study, between 1 to 1000
nevs_fleets = list(set([int(n) for n in np.logspace(np.log10(1), np.log10(10000), 67)]))
nevs_fleets.sort()
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,0.66,1,1.31,2,3.4,10000)
#n_if_needed = (0,0.5,1,1.6,100)
# number of fleets to analyze for pmax
nfleets = 500
# arrival and departure data from electric nation
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)
bins=np.arange(0,24.5,0.5)
adwd = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe = dict(pdf_a_d=arr_dep_we.values, bins=bins)     

# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50,
                 arrival_departure_data_wd=adwd, 
                 arrival_departure_data_we=adwe)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

peak99 = {}
peak95 = {}
peak90 = {}
peak50 = {}
peakmean = {}
peaktime = {}
nsessions = {}

# indexes to reduce data. We'll only look for pmax between these hours
times = [16,24] # hours in which we'll check for pmax
dt = times[1]-times[0] 
dsteps = int(dt*60/step)
idxs = [True if ((t%24 >= times[0]) & (t%24 < times[1])) else False for (_,t,_) in grid.times ]

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, alpha in enumerate(n_if_needed):
            alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
            for i, ev in enumerate(evs):
                ev.n_if_needed = alphs[i]
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(alpha))
            _, m, _ = charging_sessions(grid)
            nsessions[b,pch,alpha] = m
            ta = [time.time()]
            # get charging profiles
            ch_profs = np.array([ev.charging for ev in evs])
            # reducing data to make it faster
            ch_profs = ch_profs[:, idxs]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                p = []
                pt = []
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    profs = ch_profs[idx, :].sum(axis=0)
                    p.append(profs.max())
                    pt.append(times[0] + (profs.argmax()%dsteps) * (step/60))
                p.sort()
                peak99[b,pch,alpha,nev] = p[int(nfleets * 0.99)-1]/nev
                peak95[b,pch,alpha,nev] = p[int(nfleets * 0.95)-1]/nev
                peak90[b,pch,alpha,nev] = p[int(nfleets * 0.90)-1]/nev
                peak50[b,pch,alpha,nev] = p[int(nfleets * 0.50)-1]/nev
                peakmean[b,pch,alpha,nev] = np.mean(p)/nev
                peaktime[b,pch,alpha,nev] = np.mean(pt)
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(alpha, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
t.append(time.time())
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))


peak99 = peak_to_series_and_save(peak99, namesave='peak99_adEN.csv')
peak95 = peak_to_series_and_save(peak95, namesave='peak95_adEN.csv')
peak90 = peak_to_series_and_save(peak90, namesave='peak90_adEN.csv')
peak50 = peak_to_series_and_save(peak50, namesave='peak50_adEN.csv')
peakmean = peak_to_series_and_save(peakmean, namesave='peakmean_adEN.csv')
peaktime = peak_to_series_and_save(peaktime, namesave='peaktime_adEN.csv', name='Peaktime')
nsessions = peak_to_series_and_save(nsessions, namesave='nsessions_adEN.csv', name='nsessions', indexnames=['Batt_size', 'Charger_power', 'alpha'])


#%% Identifying peak load version2 - ToU charging
# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 batt_size=50,
                 arrival_departure_data_wd=adwd, 
                 arrival_departure_data_we=adwe,
                 tou_ini=22, tou_end=8, tou=True)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

peak99 = {}
peak95 = {}
peak90 = {}
peak50 = {}
peakmean = {}

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, alpha in enumerate(n_if_needed):
            alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
            for i, ev in enumerate(evs):
                ev.n_if_needed = alphs[i]
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(alpha))
            _, m, _ = charging_sessions(grid)
            nsessions[b,pch,alpha] = m
            ta = [time.time()]
            # get charging profiles
            ch_profs = np.array([ev.charging for ev in evs])
            # reducing data to make it faster
            times = [22,24] # hours in which we'll check for pmax
            idxs = [True if ((t%24 >= times[0]) & (t%24 < times[1])) else False for (_,t,_) in grid.times ]
            ch_profs = ch_profs[:, idxs]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                p = []
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    p.append(ch_profs[idx, :].sum(axis=0).max())
                p.sort()
                peak99[b,pch,alpha,nev] = p[int(nfleets * 0.99)-1]/nev
                peak95[b,pch,alpha,nev] = p[int(nfleets * 0.95)-1]/nev
                peak90[b,pch,alpha,nev] = p[int(nfleets * 0.90)-1]/nev
                peak50[b,pch,alpha,nev] = p[int(nfleets * 0.50)-1]/nev
                peakmean[b,pch,alpha,nev] = np.mean(p)/nev
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(alpha, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
t.append(time.time())
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))

peak99 = peak_to_series_and_save(peak99, namesave='peak99_ToU_adEN.csv')
peak95 = peak_to_series_and_save(peak95, namesave='peak95_ToU_adEN.csv')
peak90 = peak_to_series_and_save(peak90, namesave='peak90_ToU_adEN.csv')
peak50 = peak_to_series_and_save(peak50, namesave='peak50_ToU_adEN.csv')
peakmean = peak_to_series_and_save(peakmean, namesave='peakmean_ToU_adEN.csv')

#%% Identifying peak load - Rural case
t = [time.time()]
## Simulation parameters
nweeks = 12
ndays = nweeks*7
step = 30

nevs = 10000

# EV PARAMS:
# Battery
batts = np.array((25,50,75))
# driving eff
driving_eff = ((batts * 0.09)+14)/100
# charging power [kw]
pcharger = (3.6,7.2,10.7)
# fleet sizes to study, between 1 to 1000
nevs_fleets = list(set([int(n) for n in np.logspace(np.log10(1), np.log10(10000), 67)]))
nevs_fleets.sort()
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,1.31,3.4,10000)
#n_if_needed = (0,0.5,1,1.6,100)
# number of fleets to analyze for pmax
nfleets = 500
# arrival and departure data from electric nation
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)
bins=np.arange(0,24.5,0.5)
adwd = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe = dict(pdf_a_d=arr_dep_we.values, bins=bins)     


# histograms of driving 
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=[0,1,2,3,4])
# 
idx = pd.IndexSlice

hrural = hhome.loc[idx[:,:,'R',:,77],:].sum(axis=0)/hhome.loc[idx[:,:,'R',:,77],:].sum().sum()
hurban = hhome.loc[idx[:,:,'C',:,75],:].sum(axis=0)/hhome.loc[idx[:,:,'C',:,75],:].sum().sum()

# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50,
#                 arrival_departure_data_wd=adwd, 
#                 arrival_departure_data_we=adwe,
                 dist_wd=dict(cdf=hrural.cumsum().values))
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

peak99 = {}
peak95 = {}
peak90 = {}
peak50 = {}
peakmean = {}
peaktime = {}
nsessions = {}

# indexes to reduce data. We'll only look for pmax between these hours
times = [16,24] # hours in which we'll check for pmax
dt = times[1]-times[0] 
dsteps = int(dt*60/step)
idxs = [True if ((t%24 >= times[0]) & (t%24 < times[1])) else False for (_,t,_) in grid.times ]

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, alpha in enumerate(n_if_needed):
            alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
            for i, ev in enumerate(evs):
                ev.n_if_needed = alphs[i]
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(alpha))
            _, m, _ = charging_sessions(grid)
            nsessions[b,pch,alpha] = m
            ta = [time.time()]
            # get charging profiles
            ch_profs = np.array([ev.charging for ev in evs])
            # reducing data to make it faster
            ch_profs = ch_profs[:, idxs]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                p = []
                pt = []
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    profs = ch_profs[idx, :].sum(axis=0)
                    p.append(profs.max())
                    pt.append(times[0] + (profs.argmax()%dsteps) * (step/60))
                p.sort()
                peak99[b,pch,alpha,nev] = p[int(nfleets * 0.99)-1]/nev
                peak95[b,pch,alpha,nev] = p[int(nfleets * 0.95)-1]/nev
                peak90[b,pch,alpha,nev] = p[int(nfleets * 0.90)-1]/nev
                peak50[b,pch,alpha,nev] = p[int(nfleets * 0.50)-1]/nev
                peakmean[b,pch,alpha,nev] = np.mean(p)/nev
                peaktime[b,pch,alpha,nev] = np.mean(pt)
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(alpha, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
t.append(time.time())
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))


peak99 = peak_to_series_and_save(peak99, namesave='peak99rural.csv')
peak95 = peak_to_series_and_save(peak95, namesave='peak95rural.csv')
peak90 = peak_to_series_and_save(peak90, namesave='peak90rural.csv')
peak50 = peak_to_series_and_save(peak50, namesave='peak50rural.csv')
peakmean = peak_to_series_and_save(peakmean, namesave='peakmeanrural.csv')
peaktime = peak_to_series_and_save(peaktime, namesave='peaktimerural.csv', name='Peaktime')
nsessions = peak_to_series_and_save(nsessions, namesave='nsessionsrural.csv', name='nsessions', indexnames=['Batt_size', 'Charger_power', 'alpha'])

#%% Identifying peak load - Urban case
t = [time.time()]
## Simulation parameters
nweeks = 12
ndays = nweeks*7
step = 30

nevs = 10000

# EV PARAMS:
# Battery
batts = np.array((25,50,75))
# driving eff
driving_eff = ((batts * 0.09)+14)/100
# charging power [kw]
pcharger = (3.6,7.2,10.7)
# fleet sizes to study, between 1 to 1000
nevs_fleets = list(set([int(n) for n in np.logspace(np.log10(1), np.log10(10000), 67)]))
nevs_fleets.sort()
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,1.31,3.4,10000)
#n_if_needed = (0,0.5,1,1.6,100)
# number of fleets to analyze for pmax
nfleets = 500
# arrival and departure data from electric nation
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)
bins=np.arange(0,24.5,0.5)
adwd = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe = dict(pdf_a_d=arr_dep_we.values, bins=bins)     


# histograms of driving 
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=[0,1,2,3,4])
#
idx = pd.IndexSlice
 
hrural = hhome.loc[idx[:,:,'R',:,77],:].sum(axis=0)/hhome.loc[idx[:,:,'R',:,77],:].sum().sum()
hurban = hhome.loc[idx[:,:,'C',:,75],:].sum(axis=0)/hhome.loc[idx[:,:,'C',:,75],:].sum().sum()

# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50,
#                 arrival_departure_data_wd=adwd, 
#                 arrival_departure_data_we=adwe,
                 dist_wd=dict(cdf=hurban.cumsum().values))
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

peak99 = {}
peak95 = {}
peak90 = {}
peak50 = {}
peakmean = {}
peaktime = {}
nsessions = {}

# indexes to reduce data. We'll only look for pmax between these hours
times = [16,24] # hours in which we'll check for pmax
dt = times[1]-times[0] 
dsteps = int(dt*60/step)
idxs = [True if ((t%24 >= times[0]) & (t%24 < times[1])) else False for (_,t,_) in grid.times ]

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, alpha in enumerate(n_if_needed):
            alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
            for i, ev in enumerate(evs):
                ev.n_if_needed = alphs[i]
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(alpha))
            _, m, _ = charging_sessions(grid)
            nsessions[b,pch,alpha] = m
            ta = [time.time()]
            # get charging profiles
            ch_profs = np.array([ev.charging for ev in evs])
            # reducing data to make it faster
            ch_profs = ch_profs[:, idxs]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                p = []
                pt = []
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    profs = ch_profs[idx, :].sum(axis=0)
                    p.append(profs.max())
                    pt.append(times[0] + (profs.argmax()%dsteps) * (step/60))
                p.sort()
                peak99[b,pch,alpha,nev] = p[int(nfleets * 0.99)-1]/nev
                peak95[b,pch,alpha,nev] = p[int(nfleets * 0.95)-1]/nev
                peak90[b,pch,alpha,nev] = p[int(nfleets * 0.90)-1]/nev
                peak50[b,pch,alpha,nev] = p[int(nfleets * 0.50)-1]/nev
                peakmean[b,pch,alpha,nev] = np.mean(p)/nev
                peaktime[b,pch,alpha,nev] = np.mean(pt)
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(alpha, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
t.append(time.time())
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))


peak99 = peak_to_series_and_save(peak99, namesave='peak99urban.csv')
peak95 = peak_to_series_and_save(peak95, namesave='peak95urban.csv')
peak90 = peak_to_series_and_save(peak90, namesave='peak90urban.csv')
peak50 = peak_to_series_and_save(peak50, namesave='peak50urban.csv')
peakmean = peak_to_series_and_save(peakmean, namesave='peakmeanurban.csv')
peaktime = peak_to_series_and_save(peaktime, namesave='peaktimeurban.csv', name='Peaktime')
nsessions = peak_to_series_and_save(nsessions, namesave='nsessionsurban.csv', name='nsessions', indexnames=['Batt_size', 'Charger_power', 'alpha'])


#%% Plot peak load results
#peak = pd.read_csv(folder_outputs + 'peak95_adEN.csv', engine='python', index_col=[0,1,2,3]).Peak_load
peak = pd.read_csv(folder_outputs + 'peak99_ToU_adEN.csv', engine='python', index_col=[0,1,2,3]).Peak_load

# Vary all chargers, all batt sizes
f, ax = plt.subplots()
ax.set_xscale('log')
battery = [25,50,75]
pch = [3.6,7]
alpha = 0.5
ls = ['-','--',':','-.','-','--']
for i, p in enumerate(pch):
    for j, b in enumerate(battery):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,alpha,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label='{:.1f} kVA; {} kWh'.format(p,b))
plt.legend()
plt.xlabel('Fleet size')
plt.ylabel('Power [kW]')
plt.xlim((1,10000))
plt.ylim((0,7.5))
plt.grid(linestyle='--', axis='x')

# Vary all chargers, all alphas
for b in peak.index.levels[0]:
    f, ax = plt.subplots()
    ax.set_xscale('log')
    #b = 50
    pch = [3.6,7,10]
    alpha = peak.index.levels[2]
    ls = ['-','--',':','-.',(0, (3, 1, 1, 1)),(0, (3, 1, 2, 1)) ]
    for i, p in enumerate(pch):
        for j, a in enumerate(alpha):
            #peak[battery, charger, alpha, :]
            k=peak.loc[b,p,a,:]
            plt.plot(k.index.get_level_values('fleet_size'), k, 
                     linestyle=ls[j%6], label=r'{:.1f} kVA; $\alpha$={}'.format(p,a))
    plt.legend()
    plt.xlabel('Fleet size')
    plt.ylabel('Power [kW]')
    plt.xlim((1,10000))
    plt.ylim((0,11))
    plt.grid(linestyle='--', axis='x')
    f.suptitle('{} kWh batteries'.format(b))

#%% Plot peak load ToU
peak = pd.read_csv(folder_outputs + 'peak99_adEN.csv', engine='python', index_col=[0,1,2,3]).Peak_load
peaktou = pd.read_csv(folder_outputs + 'peak99_ToU_adEN.csv', engine='python', index_col=[0,1,2,3]).Peak_load

## Vary all chargers, all batt sizes
#f, ax = plt.subplots()
ax.set_xscale('log')
battery = [25,50,75]
pch = [3.6,7]
alpha = 1.31
ls = ['-','--',':','-.']

for j, b in enumerate(battery):
    f, ax = plt.subplots()
    ax.set_xscale('log')
    for i, p in enumerate(pch):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,alpha,:]
        tou=peaktou.loc[b,p,alpha,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[0], label='{:.1f} kW; Non-systematic'.format(p,b))
        plt.plot(tou.index.get_level_values('fleet_size'), tou, 
                 linestyle=ls[1], label='{:.1f} kW; Systematic'.format(p,b))
    plt.legend()
    plt.xlabel('Fleet size')
    plt.ylabel('Power [kW]')
    plt.xlim((1,10000))
    plt.ylim((0,7.5))
    plt.grid(linestyle='--', axis='x')
    f.suptitle('{} kWh batteries'.format(b))
    
#%% Plot peak load Uncontrolled - paper format  
lvl = '95_adEN' # 95, mean, 95_ToU, etc
peak = pd.read_csv(folder_outputs + 'peak{}.csv'.format(lvl), engine='python', index_col=[0,1,2,3]).Peak_load

# One fig, two subplots. Left three batt sizes, alpha average. Right, 50kW, all alphas
f, axs = plt.subplots(1,2, sharey=True)
ax = axs[0]
ax.set_xscale('log')
plt.sca(ax)
battery = [25,50,75]
pch = [3.6,7.2,10.7]
alpha = 1.31
lkw = [3.7,7.4,11]
ls = [':','-','--',':','-.','-','--']
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for i, p in enumerate(pch):
    for j, b in enumerate(battery):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,alpha,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label='{} kVA; {} kWh'.format(lkw[i],b),
                 color=colors[j + 3*i])
plt.legend(fontsize=8)
plt.xlabel('Fleet size')
plt.ylabel('Power [kW/EV]')
plt.xlim((1,10000))
plt.ylim((0,12))
plt.grid(linestyle='--', axis='x')
ax.set_title('(a) Average plug-in behavior', y=-0.25)

# Vary all chargers, all alphas
ax = axs[1]
plt.sca(ax)
ax.set_xscale('log')
b = 50
pch = [3.6,7.2,10.7]
alpha = [0.5,1.31,3.4,10000]
labelalpha = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
ls = ['-','--',':','-.',(0, (3, 1, 1, 1)),(0, (3, 1, 2, 1)) ]
colors = ['tab:blue', 'tab:orange', 'tab:green',  'maroon',
          'tab:red', 'tab:purple', 'tab:brown', 'darkblue', 
          'tab:pink',  'tab:gray', 'tab:olive', 'tab:cyan']
ls=[':','--','-.','-']
for i, p in enumerate(pch):
    for j, a in enumerate(alpha):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,a,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label=r'{} kVA; {}'.format(lkw[i],labelalpha[j]),
                 color=colors[(j + 4*i)%9])
plt.legend(fontsize=8)
plt.xlabel('Fleet size')
#plt.ylabel('Power [kW/EV]')
plt.xlim((1,10000))
plt.ylim((0,12))
plt.grid(linestyle='--', axis='x')
ax.set_title('(b) 50 kWh battery', y=-0.25)

    
#f.set_size_inches(11,4.76)    
f.set_size_inches(7.4, 3.3)  

plt.tight_layout()
folder_img = r'c:\user\U546416\Pictures\PlugInModel\Coincidence\Randalpha\\'
#plt.savefig(folder_img + 'peakload_vpapersize_{}.png'.format(lvl))
#plt.savefig(folder_img + 'peakload_vpapersize_{}.pdf'.format(lvl))

#%% Plot peak load - paper format  : 2 figs; 3 plots
lvl = '95_ToU' # 95, mean, 95_ToU, etc
peak = pd.read_csv(folder_outputs + 'peak{}.csv'.format(lvl), engine='python', index_col=[0,1,2,3]).Peak_load

# One fig, three subplots. Each subplot with a chpower level. Three batt sizes (25/50/75), alpha average per subplot
f0, axs = plt.subplots(1,3, sharey=True)
battery = [25,50,75]
pch = [3.6,7.2,10.7]
alpha = 1.31
lkw = [3.7,7.4,11]
ls = [':','-','--',':','-.','-','--']
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
abc = 'abc'
defg = 'defg'

dx = 0  #0.0455 for legend on the right 
dy = 0  # 0.06 for legend under plot
dyu = 0.06
yt = -0.35
ncol = 4
loclegend = 9

for i, p in enumerate(pch):
    ax=axs[i]
    ax.set_xscale('log')
    plt.sca(ax)
    for j, b in enumerate(battery):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,alpha,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label='{} kWh'.format(b),
                 color=colors[j])
#    plt.legend()
    plt.xlabel('Fleet size')
    if i==0:
        plt.ylabel('Power [kW/EV]')
    plt.xlim((1,10000))
    plt.ylim((0,12))
    plt.grid(linestyle='--', axis='x')
#    plt.text(x=2,y=11,s='{} kVA charger'.format(lkw[i]), fontweight='bold')
    ax.set_title('({}) Average plug-in\n     {} kVA charger'.format(abc[i], lkw[i]), y=yt)

f0.legend(ncol=ncol, loc=loclegend) # legend under plot
#f0.legend(ncol=1, loc=5) # legend on the right
f0.set_size_inches(7.4, 3.3)  #9,4
f0.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    ax.set_position([pos.x0-(dx*i), pos.y0+dy, pos.width-dx, pos.height-dy-dyu])
    

# One fig, three subplots. Each subplot with a chpower level. 4 alpha values (low,avg,high,syst), batt 50 kWh
f1, axs = plt.subplots(1,3, sharey=True)

b = 50
pch = [3.6,7.2,10.7]
alpha = [0.5,1.31,3.4,10000]
labelalpha = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
ls = ['-','--',':','-.',(0, (3, 1, 1, 1)),(0, (3, 1, 2, 1)) ]
colors = ['tab:blue', 'tab:orange', 'tab:green',  'maroon',
          'tab:red', 'tab:purple', 'tab:brown', 'darkblue', 
          'tab:pink',  'tab:gray', 'tab:olive', 'tab:cyan']
ls = [':','--','-.','-']
lkw = [3.7,7.4,11]
for i, p in enumerate(pch):
    ax=axs[i]
    ax.set_xscale('log')
    plt.sca(ax)
    for j, a in enumerate(alpha):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,a,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label=r'{}'.format(labelalpha[j]),
                 color=colors[j])
#plt.legend()
    plt.xlabel('Fleet size')
    if i==0:
        plt.ylabel('Power [kW/EV]')
    plt.xlim((1,10000))
    plt.ylim((0,12))
    plt.grid(linestyle='--', axis='x')
#    plt.text(x=2,y=11,s='{} kVA charger'.format(lkw[i]), fontweight='bold') 
    ax.set_title('({})  50 kWh battery \n     {} kVA charger'.format(defg[i], lkw[i]), y=yt)
        
f1.legend(ncol=ncol, loc=loclegend)
f1.set_size_inches(7.4, 3.3)  #(9,4) #11, 4.76)   
f1.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    ax.set_position([pos.x0-(dx*i), pos.y0+dy, pos.width-dx, pos.height-dy-dyu])
    
f0.tight_layout()
f1.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(f0.axes):
    pos = ax.get_position()
    ax.set_position([pos.x0-(dx*i), pos.y0+dy, pos.width-dx, pos.height-dy-dyu])
# resizing axs to leave space for legend
for i, ax in enumerate(f1.axes):
    pos = ax.get_position()
    ax.set_position([pos.x0-(dx*i), pos.y0+dy, pos.width-dx, pos.height-dy-dyu])
    
folder_img = r'c:\user\U546416\Pictures\PlugInModel\Coincidence\Randalpha\\'
f0.savefig(folder_img + 'peakload_vpapersize_5.1_{}.png'.format(lvl))
f0.savefig(folder_img + 'peakload_vpapersize_5.1_{}.pdf'.format(lvl))
f1.savefig(folder_img + 'peakload_vpapersize_5.2_{}.png'.format(lvl))
f1.savefig(folder_img + 'peakload_vpapersize_5.2_{}.pdf'.format(lvl))


#%% Plot peak load Rural-Avg-Urban - paper format  : 2 figs; 3 plots
lvl = '95' # 95, mean, 95_ToU, etc
v = '_adEN' # '' or '_adEN'
peaka = pd.read_csv(folder_outputs + 'peak{}{}.csv'.format(lvl,v), engine='python', index_col=[0,1,2,3]).Peak_load
peaku = pd.read_csv(folder_outputs + 'peak{}urban{}.csv'.format(lvl,v), engine='python', index_col=[0,1,2,3]).Peak_load
peakr = pd.read_csv(folder_outputs + 'peak{}rural{}.csv'.format(lvl,v), engine='python', index_col=[0,1,2,3]).Peak_load

# One fig, three subplots. Three charger power, alpha average, 50 kWh
f, axs = plt.subplots(1,3)
b = 50
pch = [3.6,7.2,10.7]
alpha = 1.31
ls = [':','-','--',':','-.','-','--']
lkw = [3.7,7.4,11]
colors = ['tab:blue', 'tab:orange', 'tab:green', 
          'tab:red', 'tab:purple', 'tab:brown', 
          'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

for i, p in enumerate(pch):
    ax=axs[i]
    ax.set_xscale('log')
    plt.sca(ax)
    k=peaku.loc[b,p,alpha,:]
    plt.plot(k.index.get_level_values('fleet_size'), k, 
             linestyle=ls[0], label='Urban'.format(b),
             color=colors[0])
    k=peaka.loc[b,p,alpha,:]
    plt.plot(k.index.get_level_values('fleet_size'), k, 
             linestyle=ls[1], label='Average'.format(b),
             color=colors[1])    
    k=peakr.loc[b,p,alpha,:]
    plt.plot(k.index.get_level_values('fleet_size'), k, 
             linestyle=ls[2], label='Rural'.format(b),
             color=colors[2])
#    plt.legend()
    plt.xlabel('Fleet size')
    plt.ylabel('Power [kW/EV]')
    plt.xlim((1,10000))
    plt.ylim((0,12))
    plt.grid(linestyle='--', axis='x')
    plt.text(x=2,y=11,s='{} kVA charger'.format(lkw[i]), fontweight='bold')
f.legend(ncol=3, loc=8)
f.set_size_inches(11,4.76)   
f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy])

folder_img = r'c:\user\U546416\Pictures\PlugInModel\Coincidence\Randalpha\\'    
plt.savefig(folder_img + 'peakload_urbanrural_v4.1_{}{}.png'.format(lvl,v))
plt.savefig(folder_img + 'peakload_urbanrural_v4.1_{}{}.pdf'.format(lvl,v))

## Vary all chargers, all alphas
#f, axs = plt.subplots(1,3)
#
#b = 50
#pch = [3.6,7.2,10.7]
#alpha = [0.5,1.31,3.4,10000]
#labelalpha = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
#ls = ['-','--',':','-.',(0, (3, 1, 1, 1)),(0, (3, 1, 2, 1)) ]
#colors = ['tab:blue', 'tab:orange', 'tab:green',  'maroon',
#          'tab:red', 'tab:purple', 'tab:brown', 'darkblue', 
#          'tab:pink',  'tab:gray', 'tab:olive', 'tab:cyan']
#ls = [':','--','-.','-']
#lkw = [3.7,7.4,11]
#for i, p in enumerate(pch):
#    ax=axs[i]
#    ax.set_xscale('log')
#    plt.sca(ax)
#    for j, a in enumerate(alpha):
#        #peak[battery, charger, alpha, :]
#        k=peak.loc[b,p,a,:]
#        plt.plot(k.index.get_level_values('fleet_size'), k, 
#                 linestyle=ls[j], label=r'{}'.format(labelalpha[j]),
#                 color=colors[j])
##plt.legend()
#    plt.xlabel('Fleet size')
#    plt.ylabel('Power [kW/EV]')
#    plt.xlim((1,10000))
#    plt.ylim((0,12))
#    plt.grid(linestyle='--', axis='x')
#    plt.text(x=2,y=11,s='{} kVA charger'.format(lkw[i]), fontweight='bold')
#        
#f.legend(ncol=4, loc=8)
#f.set_size_inches(11,4.76)   
#f.tight_layout()
#
## resizing axs to leave space for legend
#for i, ax in enumerate(axs):
#    pos = ax.get_position()
#    dy = 0.06
#    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy])
#    
#folder_img = r'c:\user\U546416\Pictures\PlugInModel\Coincidence\Randalpha\\'
#plt.savefig(folder_img + 'peakload_urbanrural_v4.2_{}{}.png'.format(lvl,v))
#plt.savefig(folder_img + 'peakload_urbanrural_v4.2_{}{}.pdf'.format(lvl,v))
#%% Smart charging potential

## Simulation parameters
nweeks = 12
ndays = nweeks*7
step = 60

nevs = 1000

# EV PARAMS:
# Battery
batts = np.arange(15,100.1,2.5)
# driving eff
driving_eff = ((batts * 0.09)+14)/100
# charging power [kw]
pcharger = (3.6,7.2,10.7)
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,0.66,1,1.31,2,3.4,10000)

# arrival and departure data from electric nation
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)
bins=np.arange(0,24.5,0.5)
adwd = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe = dict(pdf_a_d=arr_dep_we.values, bins=bins)     

# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs=grid.add_evs('test', nevs, 'dumb', 
                 charging_type='if_needed',
                 n_if_needed=n_if_needed, batt_size=50,
                 arrival_departure_data_wd=adwd, 
                 arrival_departure_data_we=adwe)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))
# TODO add ElNation deparr data

dur_ch = {} # Average duration of charging part of sessions for all charging sessions
dur_ch2 = {} # Average user duration of charging part of sessions for all users (first compute avg duration by user, then aggregate)
dur_ss = {} # Average duration of charging sessions for all charging sessions
dur_ss2 = {} # Average user duration of charging sessions for all users (first compute avg duration by user, then aggregate)
sc_pot = {} # Idle time of all charging sessions
sc_pot2 = {} # average idle time for users (first, compute idle time by user, then aggregate)
time_connected = {} # Average time connected, in hours/day/user
time_charging = {} # Average time spent charging, in hours/day/user
flex_kwhh = {} #
flex_kwh = {} #

for k, alpha in enumerate(n_if_needed):
    alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
    print('\talpha {}'.format(alpha))
    for i, ev in enumerate(evs):
        ev.n_if_needed = alphs[i]        
    for i, b in enumerate(batts):
        print('\t\tBatt {} kWh'.format(b))
        grid.set_evs_param(param='batt_size', value=b)
        grid.set_evs_param(param='driving_eff', value=driving_eff[i])
        for j, pch in enumerate(pcharger):
            print('\t\tCharging power {} kW'.format(pch))
            grid.set_evs_param(param='charging_power', value=pch)
            grid.reset()
            grid.do_days()
            _, m, _ = charging_sessions(grid)
            sc_pot[b, pch, alpha] = grid.get_global_data()['Flex_ratio']
            sc_pot2[b, pch, alpha] = np.mean([1-ev.charging.sum()/ev.off_peak_potential.sum() for ev in evs])
            dur_ch[b, pch, alpha] = sum([ev.charging.sum() for ev in evs])/sum([ev.ch_status.sum() for ev in evs])/(step/60)/pch
            dur_ch2[b, pch, alpha] = np.mean([ev.charging.sum()/ev.ch_status.sum() for ev in evs])/(step/60)/pch
            dur_ss[b, pch, alpha] = sum([ev.potential.sum() for ev in evs])/sum([ev.ch_status.sum() for ev in evs])/(step/60)/pch
            dur_ss2[b, pch, alpha] = np.mean([ev.potential.sum()/ev.ch_status.sum() for ev in evs])/(step/60)/pch
            time_connected[b, pch, alpha] = sum([ev.off_peak_potential.sum() for ev in evs])/pch/(step/60)/ndays/nevs
            time_charging[b, pch, alpha] = sum([ev.charging.sum() for ev in evs])/pch/(step/60)/ndays/nevs
            flex_kwhh[b, pch, alpha] = sum([(ev.up_flex-ev.dn_flex).sum() for ev in evs])/ndays/nevs
            flex_kwh[b, pch, alpha] = sum([(ev.up_flex-ev.dn_flex).sum() for ev in evs])/ndays/nevs/dur_ss[b,pch,alpha]
    print('\tSimulation finished for alpha {}'.format(alpha))
            # hours available per ch session
#            tch_session[b, pch, alpha] = [ev.charging.sum() for ev in evs]

def kpi_to_series_and_save(kpi, name='kpi', namesave=None):
    if namesave is None:
        namesave = name + '.csv'
    kpi = pd.Series(kpi)
    kpi.index.names = ['Batt_size', 'Charger_power', 'alpha']
    kpi.name = name
    kpi.to_csv(folder_outputs + namesave, header=True)
    return kpi

sc_pot = kpi_to_series_and_save(sc_pot, 'Flex_potential', 'flex_pot_adEN.csv')
sc_pot2 = kpi_to_series_and_save(sc_pot2, 'Flex_potential', 'flex_pot2_adEN.csv')
dur_ch = kpi_to_series_and_save(dur_ch, 'charge_duration', 'avgchdur_adEN.csv')
dur_ch2 = kpi_to_series_and_save(dur_ch2, 'charge_duration', 'avgchdur2_adEN.csv')
dur_ss = kpi_to_series_and_save(dur_ss, 'session_duration', 'avgssdur_adEN.csv')
dur_ss2 = kpi_to_series_and_save(dur_ss2, 'session_duration', 'avgssdur2_adEN.csv')
time_connected = kpi_to_series_and_save(time_connected, 'time_connected', 'time_connected_adEN.csv')
time_charging = kpi_to_series_and_save(time_charging, 'time_charging', 'time_charging_adEN.csv')
flex_kwh = kpi_to_series_and_save(flex_kwh, 'flex_kwh', 'flex_kwh_adEN.csv')
flex_kwhh = kpi_to_series_and_save(flex_kwhh, 'flex_kwhh', 'flex_kwhh_adEN.csv')



#%% Plot SC potential
case = '' #_adEN
sc_pot = pd.read_csv(folder_outputs + 'flex_pot{}.csv'.format(case), engine='python', index_col=[0,1,2]).Flex_potential
sc_pot2 = pd.read_csv(folder_outputs + 'flex_pot2{}.csv'.format(case), engine='python', index_col=[0,1,2]).Flex_potential
dur_ch = pd.read_csv(folder_outputs + 'avgchdur{}.csv'.format(case), engine='python', index_col=[0,1,2]).charge_duration
dur_ch2 = pd.read_csv(folder_outputs + 'avgchdur2{}.csv'.format(case), engine='python', index_col=[0,1,2]).charge_duration
time_connected = pd.read_csv(folder_outputs + 'time_connected{}.csv'.format(case), engine='python', index_col=[0,1,2]).time_connected
time_charging = pd.read_csv(folder_outputs + 'time_charging{}.csv'.format(case), engine='python', index_col=[0,1,2]).time_charging

# Vary all batts, plot diff chargers. alpha 0.5
f, ax = plt.subplots()
b = np.arange(15,100.1,2.5)
pch = [3.6,7.2,10.7]
alpha = 0.5
ls = ['-','--',':','-.']
for i, p in enumerate(pch):
    #sc_pot[battery, charger, alpha]
    k=sc_pot.loc[:,p,alpha]
    plt.plot(k.index.get_level_values('Batt_size'), k, 
             linestyle=ls[i], label=r'{} kVA'.format(lkw[i]))
plt.legend()
plt.xlabel('Battery size [kWh]')
plt.ylabel('Flexibility ratio [p.u.]')
plt.xlim((15,100))
plt.ylim((0,1))
plt.grid(linestyle='--')

# time connected vs time charging, vary all batts, alpha 0.5
for a in time_connected.index.levels[2]:
    f, ax = plt.subplots()
    b = np.arange(15,100.1,2.5)
    pch = [3.6,7]
#    alpha = 0.5
    ls = ['-','--',':','-.']
    for i, p in enumerate(pch):
        #sc_pot[battery, charger, alpha]
        k=time_connected.loc[:,p,a]
        plt.plot(k.index, k, 
                 linestyle=ls[i], label=r'Time connected - {} kVA'.format(lkw[i]))
        j=time_charging.loc[:,p,a]
        plt.plot(k.index, j, 
                 linestyle=ls[i], label=r'Time charging - {} kVA'.format(lkw[i]))
        plt.fill_between(k.index.get_level_values('Batt_size'), j,k, color='y', alpha=0.1)
    plt.legend()
    plt.xlabel('Battery size [kWh]')
    plt.ylabel('Hours/day')
    plt.xlim((15,100))
    plt.ylim((0,15))
    plt.grid(linestyle='--')
    f.suptitle(r'$\alpha$={}'.format(a))
    plt.text(30, (j[30] + k[30])/2, 'Idle time')


# Vary all batts, plot diff chargers. all alpha
f, ax = plt.subplots()
b = np.arange(15,100.1,2.5)
pch = [3.6,7]
alpha = [0,.5,1,1.31,2,100]
ls = ['-','--',':','-.',(0, (3, 1, 1, 1)),(0, (3, 1, 2, 1)) ]
for i, p in enumerate(pch):
    for j, a in enumerate(alpha):
        #sc_pot[battery, charger, alpha]
        k=sc_pot.loc[:,p,a]
        plt.plot(k.index.get_level_values('Batt_size'), k, 
                 linestyle=ls[j], label=r'{} kVA; $\alpha$={}'.format(p,a))
plt.legend()
plt.xlabel('Battery size [kWh]')
plt.ylabel('Flexibility ratio [p.u.]')
plt.xlim((15,100))
plt.ylim((0,1))
plt.grid(linestyle='--')

#%% Plot SC potential - paper version
folder_images = r'c:\user\U546416\Pictures\PlugInModel\SCPotential\\'

case = '_adEN' #_adEN

sc_pot = pd.read_csv(folder_outputs + 'flex_pot{}.csv'.format(case), engine='python', index_col=[0,1,2]).Flex_potential
sc_pot2 = pd.read_csv(folder_outputs + 'flex_pot2{}.csv'.format(case), engine='python', index_col=[0,1,2]).Flex_potential
dur_ch = pd.read_csv(folder_outputs + 'avgchdur{}.csv'.format(case), engine='python', index_col=[0,1,2]).charge_duration
dur_ch2 = pd.read_csv(folder_outputs + 'avgchdur2{}.csv'.format(case), engine='python', index_col=[0,1,2]).charge_duration
time_connected = pd.read_csv(folder_outputs + 'time_connected{}.csv'.format(case), engine='python', index_col=[0,1,2]).time_connected
time_charging = pd.read_csv(folder_outputs + 'time_charging{}.csv'.format(case), engine='python', index_col=[0,1,2]).time_charging
dur_ss = pd.read_csv(folder_outputs + 'avgssdur{}.csv'.format(case), engine='python', index_col=[0,1,2]).session_duration
flex_kwh = pd.read_csv(folder_outputs + 'flex_kwh{}.csv'.format(case), engine='python', index_col=[0,1,2]).flex_kwh
flex_kwhh = pd.read_csv(folder_outputs + 'flex_kwhh{}.csv'.format(case), engine='python', index_col=[0,1,2]).flex_kwhh

# time connected vs time charging, vary all batts, alpha 1.31
#for a in time_connected.index.levels[2]:
f, ax = plt.subplots()
b = np.arange(15,100.1,2.5)
pch = [3.6,7.2,10.7]
a = 1.31
ls = ['-','--',':','-.',(0, (3, 1, 1, 1)),(0, (3, 1, 2, 1))]
ylim = np.round(time_connected.loc[:,7.2,10000].mean()) + 1
k=time_connected.loc[:,7.2,a]
plt.plot(k.index, k, 
             linestyle=ls[0], label=r'Time connected - Non-systematic plug-in')
l=time_connected.loc[:,7.2,10000]
plt.plot(l.index, l, 
         linestyle=ls[1], label=r'Time connected - Systematic plug-in')
lkw = [3.7, 7.4, 11]
for i, p in enumerate(pch):
    #sc_pot[battery, charger, alpha]
    j=time_charging.loc[:,p,a]
    plt.plot(k.index, j, 
             linestyle=ls[i+2], label=r'Time charging - {} kVA'.format(lkw[i]))
    plt.fill_between(k.index.get_level_values('Batt_size'), j,k, color='y', alpha=0.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.95,0.99))
plt.xlabel('Battery size [kWh]')
plt.ylabel('Hours/day/EV')
plt.xlim((15,100))
plt.ylim((0,ylim))
plt.grid(linestyle='--')
#f.suptitle(r'$\alpha$={}'.format(a))
plt.text(30, (j[30] + k[30])/2, 'Idle time')

#plt.savefig(folder_images + 'idletime_{}_v2.pdf'.format(case))
#plt.savefig(folder_images + 'idletime_{}_v2.png'.format(case))

#%% Plot avg charging time per session -- Paper version

# Plot avg charging time per session three plots
labels = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
abc = 'abc'
ls = [':','--','-.','-']
alphas = [0.5,1.31,3.4,10000]
f,axs=plt.subplots(1,3, sharey=True)
ylim = round(dur_ss.mean()) + 1
xb_text = 55
for i, p in enumerate(pch):
    plt.sca(axs[i])
    ax = plt.gca()
    for k, a in enumerate(alphas):
        j=dur_ch.loc[:,p,a]
        plt.plot(j.index, j, linestyle=ls[k],
                 label=r'{}'.format(labels[k]))
    plt.plot(dur_ss.loc[:,p,a], '-.', markersize=3, label='Charging session duration')
#    plt.legend(loc=1)
    plt.xlabel('Battery size [kWh]')
    if i==0:
        plt.ylabel('Hours')
    plt.xlim((15,100))
    plt.ylim((0,ylim))
    plt.grid(linestyle='--', axis='y')    

    plt.annotate('', xy=(xb_text, dur_ss.loc[xb_text,p,a]), 
                 xytext=(xb_text, dur_ch.loc[xb_text,p,0.5]), 
                 arrowprops=dict(arrowstyle='<-'))
    plt.text(x=xb_text,
             y=8.2,#(dur_ss.loc[xb_text,p,a]+ dur_ch.loc[xb_text,p,0.5])/2,
             s='  Flexible time',
             verticalalignment='center')
    plt.annotate('', xy=(xb_text,0), 
                 xytext=(xb_text,dur_ch.loc[xb_text,p,10000]), 
                 arrowprops=dict(arrowstyle='<-'))
    plt.text(x=xb_text,
             y=dur_ch.loc[xb_text,p,10000]/2,
             s='  Charging time',
             verticalalignment='center')
#    plt.text(x=20,y=12.5,s='{} kVA charger'.format(lkw[i]), fontweight='bold')
    ax.set_title('({}) {} kVA charger'.format(abc[i], lkw[i]), y=-0.25)
f.legend(ncol=5, loc=9)
f.set_size_inches(7.43,3.3) #(9,4) #11,4.76)   
f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.00
    dyu = 0.06
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy-dyu])
f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.00
    dyu = 0.06
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy-dyu])
plt.savefig(folder_images + 'chargingtime{}_vpapersize.pdf'.format(case))
plt.savefig(folder_images + 'chargingtime{}_vpapersize.png'.format(case))


#%% Plot idle time per session - bcp de lignes
labels = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
alphas = [0.5,1.31,3.4,10000]
plt.figure()
for i, p in enumerate(pch):
    for k, a in enumerate(alphas):
        j=time_connected.loc[:,p,a]-time_charging.loc[:,p,a]
        plt.plot(j.index, j, 
                 linestyle=ls[i], 
                 label=r'{} kVA, {}'.format(lkw[i], labels[k]))
plt.legend(loc=1)
plt.xlabel('Battery size [kWh]')
plt.ylabel('Hours/day/EV')
plt.xlim((15,100))
plt.ylim((0,15))
plt.grid(linestyle='--')    

#plt.savefig(folder_images + 'idletime_all{}_v2.pdf'.format(case))
#plt.savefig(folder_images + 'idletime_all{}_v2.png'.format(case))

#%% Plot flex power (kW of connected EVs) -- Paper version

# Three subplots, one for each charger power level
labels = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
alphas = [0.5,1.31,3.4,10000]
ls = [':','--','-.','-']
flexpower = time_connected*time_connected.index.get_level_values(1)/dur_ss
f,axs=plt.subplots(1,3, sharey=True)
for i, p in enumerate(pch):
    plt.sca(axs[i])
    ax = plt.gca()
    for k, a in enumerate(alphas):
        j=flexpower.loc[:,p,a]
        plt.plot(j.index, j, 
                 linestyle=ls[k], 
                 label=r'{}'.format(labels[k]))
#    plt.legend(loc=1)
    plt.xlabel('Battery size [kWh]')
    if i==0:
        plt.ylabel('Power [kW/EV]')
    plt.xlim((15,100))
    plt.ylim((0,12))
    plt.grid(linestyle='--', axis='y') 
#    plt.text(x=20,y=11,s='{} kVA charger'.format(lkw[i]), fontweight='bold')
    ax.set_title('({}) {} kVA charger'.format(abc[i], lkw[i]), y=-0.25)
f.legend(ncol=5, loc=9)
f.set_size_inches(7.47,3.3)#(9,4) # (11,4.76)   
#f.tight_layout()

## resizing axs to leave space for legend
#for i, ax in enumerate(axs):
#    pos = ax.get_position()
#    dy = 0.06
#    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy])

f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.0
    dyu = 0.06
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy-dyu])
f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.0
    dyu = 0.063
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy-dyu])

plt.savefig(folder_images + 'flexpower{}_vpapersize.pdf'.format(case))
plt.savefig(folder_images + 'flexpower{}_vpapersize.png'.format(case))


#%% Plot stored energy per EV, one figure three plots -- Paper version
labels = ['Low plug-in', 'Average plug-in', 'High plug-in', 'Systematic']
alphas = [0.5,1.31,3.4,10000]
ls = [':','--','-.','-',(0, (3, 1.5, 1, 1.5,1,1.5,1,1.5))]
f,axs=plt.subplots(1,3, sharey=True)
for i, p in enumerate(pch):
    plt.sca(axs[i])
    ax=plt.gca()
    for k, a in enumerate(alphas):
        j=flex_kwh.loc[:,p,a]
        plt.plot(j.index, j, 
                 linestyle=ls[k], 
                 label=r'{}'.format(labels[k]))
    plt.xlabel('Battery size [kWh]')
    if i==0:
        plt.ylabel('Storage [kWh/EV]')
    plt.xlim((15,100))
    plt.ylim((0,50))
    plt.grid(linestyle='--', axis='y')    
#    plt.text(x=20,y=45,s='{} kVA charger'.format(lkw[i]), fontweight='bold')
    ax.set_title('({}) {} kVA charger'.format(abc[i], lkw[i]), y=-0.25)
    plt.plot(j.index, j.index*0.8, 'k', linestyle=ls[-1], label='Storage limit')
f.legend(ncol=5, loc=9)
f.set_size_inches(7.47,3.3)#(9,4) #(11,4.76)   
f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.00
    dyu = 0.06
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy-dyu])
f.tight_layout()

# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.00
    dyu = 0.06
    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy-dyu])


plt.savefig(folder_images + 'storage{}_vpapersize.pdf'.format(case))
plt.savefig(folder_images + 'storage{}_vpapersize.png'.format(case))

#%% plot feasible area for one EV

step = 10
dt = 18
hini = 16
nst = int(dt*60/step)
h = hini + np.array([i*step/60 for i in range(nst)])
up = np.zeros(nst)
dn = np.zeros(nst)
bmax = 40
bini = 25
bmin = 40*0.2
pch = 7
eff = 0.95
# computing up & dn trajectory
up[10:-10] = (bini + (pch/(60/step) * eff * np.ones(nst-20)).cumsum()).clip(bmin,bmax)
dn[10:-10] = (bini - (pch/(60/step) / eff * np.ones(nst-20)).cumsum()).clip(bmin,bmax)
rch = (bmax - (pch/(60/step) * eff * np.ones(nst-20)).cumsum())[::-1]
dn[10:-10] = np.max(np.array([dn[10:-10], rch]), axis=0)
up[9]=bini
dn[9] = bini

# computing a sample feasible path
dx = int((bmax - (bini-0.5*20))/1)
dx2 = int(len(h[10:-10])-20-20-dx)
fp = np.concatenate((np.ones(20)*bini, 
                         bini - 0.5*np.ones(20).cumsum(), 
                         bini-0.5*20+np.zeros(dx2), 
                         (bmax+1-1*np.ones(dx).cumsum())[::-1]))
#plt.figure()
#plt.plot(h,up, 'k--')
#plt.plot(h,dn, 'k--')
#plt.fill_between(h, up, dn, color='g', alpha=0.3, label='Accessible storage')
## plot feasible path
#dx = int((bmax - (bini-0.5*20))/1)
#dx2 = int(len(h[10:-10])-20-20-dx)
#plt.plot(h[10:-10], fp,
#        'r', label='Feasible trajectory')
#plt.xlim(h[0],h[-1])
#plt.ylim(0,bmax+5)
#plt.xticks([hini +i for i in range(dt)], [(hini +i)%24 for i in range(dt)])
#plt.legend(loc=4)
#plt.xlabel('Time [h]')
#plt.ylabel('Storage [kWh]')
##plt.text((h[0]+h[-1])/2, (bmax + bmin)/2, 'Accessible storage', horizontalalignment='center')
#plt.savefig(folder_images + 'storage1ev.pdf')
#plt.savefig(folder_images + 'storage1ev.png')
#
#
#plt.figure()
# computing power demand for trajectories
up_power = np.concatenate((np.zeros(10), (up[10:-10]-up[9:-11]) * (60/step)/eff, (np.zeros(10))))
dn_power = np.concatenate((np.zeros(10), (dn[10:-50]-dn[9:-51]) * (60/step)*eff, (dn[-50:-10]-dn[-51:-11]) * (60/step)/eff, (np.zeros(10))))
fp_power = np.concatenate((np.zeros(11), (fp[1:]-fp[:-1]) * (60/step) / eff, (np.zeros(10))))
#plt.plot(h, up_power, '-.', color='darkgreen', label='Upper storage bound')
#plt.plot(h, dn_power, '--', color='darkblue', label='Lower storage bound')
#plt.plot(h, fp_power, 'r', label='Feasible trajectory')
#plt.xlim(h[0],h[-1])
#plt.ylim(-pch*1.1, pch*1.1)
#plt.xticks([hini +i for i in range(dt)], [(hini +i)%24 for i in range(dt)])
#plt.legend(loc=4)
#plt.xlabel('Time [h]')
#plt.ylabel('Power [kW]')
#
#plt.savefig(folder_images + 'powertraj1ev.pdf')
#plt.savefig(folder_images + 'powertraj1ev.png')


# Plotting them in one fig
f, axs = plt.subplots(1,2)
plt.sca(axs[0])
ax = plt.gca()
# Plot storage
plt.plot(h,up, 'k--')
plt.plot(h,dn, 'k--')
plt.fill_between(h, up, dn, color='g', alpha=0.3, label='Accessible storage')
# plot feasible path
dx = int((bmax - (bini-0.5*20))/1)
dx2 = int(len(h[10:-10])-20-20-dx)
plt.plot(h[10:-10], fp,
        'r', label='Feasible trajectory')
plt.xlim(h[0],h[-1])
plt.ylim(0,bmax+2)
plt.xticks([hini +i for i in range(0,dt+1,2)], [(hini +i)%24 for i in range(0,dt+1,2)])
plt.legend(loc=4)
plt.xlabel('Time [h]')
plt.ylabel('Storage [kWh]')
ax.set_title('(a) Accesible storage', y=-0.25)
# plot trajectoryplt.sca(axs[1])
plt.sca(axs[1])
ax = plt.gca()
plt.plot(h, up_power, '-.', color='darkgreen', label='Upper storage bound')
plt.plot(h, dn_power, '--', color='darkblue', label='Lower storage bound')
plt.plot(h, fp_power, 'r', label='Feasible trajectory')
plt.xlim(h[0],h[-1])
plt.ylim(-pch*1.1, pch*1.1)
plt.xticks([hini +i for i in range(0,dt+1,2)], [(hini +i)%24 for i in range(0,dt+1,2)])
plt.legend(loc=4)
plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
ax.set_title('(b) Charging profile', y=-0.25)

f.set_size_inches(7.47,3.3)#(9,4) #(11,4.76) 
plt.tight_layout()

plt.savefig(folder_images + 'storage_traj1ev_vpapersize.pdf')
plt.savefig(folder_images + 'storage_traj1ev_vpapersize.png')

#%% plot feasible area for two EVs - no trajectory
f, axs = plt.subplots(1,2, sharey=True)

step = 10
dt = 18
hini = 16
nst = int(dt*60/step)
h = hini + np.array([i*step/60 for i in range(nst)])
up = np.zeros(nst)
dn = np.zeros(nst)
bmax = 40
bini = 25
bmin = 40*0.2
pch = 7
eff = 0.95
# computing up & dn trajectory
up[10:-10] = (bini + (pch/(60/step) * eff * np.ones(nst-20)).cumsum()).clip(bmin,bmax)
dn[10:-10] = (bini - (pch/(60/step) / eff * np.ones(nst-20)).cumsum()).clip(bmin,bmax)
rch = (bmax - (pch/(60/step) * eff * np.ones(nst-20)).cumsum())[::-1]
dn[10:-10] = np.max(np.array([dn[10:-10], rch]), axis=0)
up[9]=bini
dn[9] = bini

# computing a sample feasible path
dx = int((bmax - (bini-0.5*20))/1)
dx2 = int(len(h[10:-10])-20-20-dx)
fp = np.concatenate((np.ones(20)*bini, 
                         bini - 0.5*np.ones(20).cumsum(), 
                         bini-0.5*20+np.zeros(dx2), 
                         (bmax+1-1*np.ones(dx).cumsum())[::-1]))
# Plotting
plt.sca(axs[0])
ax = plt.gca()
# Plot storage
plt.plot(h,up, 'k--')
plt.plot(h,dn, 'k--')
plt.fill_between(h, up, dn, color='g', alpha=0.3, label='Accessible storage')
# plot feasible path
plt.plot(h[10:-10], fp,
        'r', label='Feasible trajectory')
plt.xlim(h[0],h[-1])
plt.ylim(0,bmax+2)
plt.xticks([hini +i for i in range(0,dt+1,2)], [(hini +i)%24 for i in range(0,dt+1,2)])
#plt.legend(loc=4)
plt.xlabel('Time [h]')
plt.ylabel('Storage [kWh]')
ax.set_title('(a) 40 kWh, 7,4 kVA', y=-0.25)

# Doing it all again for a different batt
bmax = 25
bini = 25/40*25
bmin = 25*0.2
pch = 3.5
up2 = np.zeros(nst)
dn2 = np.zeros(nst)
# computing up & dn trajectory
up2[10:-10] = (bini + (pch/(60/step) * eff * np.ones(nst-20)).cumsum()).clip(bmin,bmax)
dn2[10:-10] = (bini - (pch/(60/step) / eff * np.ones(nst-20)).cumsum()).clip(bmin,bmax)
rch2 = (bmax - (pch/(60/step) * eff * np.ones(nst-20)).cumsum())[::-1]
dn2[10:-10] = np.max(np.array([dn2[10:-10], rch2]), axis=0)
up2[9]=bini
dn2[9] = bini

chdn = pch*0.90
chup = pch*0.95
# computing a sample feasible path
d2x = int((bmax - (bini-chdn/(60/step)*20))/(chup/(60/step)))
d2x2 = int(len(h[10:-10])-20-20-d2x)
fp2 = np.concatenate((np.ones(20)*bini, 
                         bini - chdn/(60/step)*np.ones(20).cumsum(), 
                         bini - chdn/(60/step)*20+np.zeros(d2x2), 
                         (bmax-chup/(60/step)*np.ones(d2x).cumsum())[::-1]))


# Plotting them in one fig

plt.sca(axs[1])
ax = plt.gca()
# Plot storage
plt.plot(h,up2, 'k--')
plt.plot(h,dn2, 'k--')
plt.fill_between(h, up2, dn2, color='g', alpha=0.3, label='Accessible storage')
# plot feasible path
dx = int((bmax - (bini-0.5*20))/1)
dx2 = int(len(h[10:-10])-20-20-dx)
plt.plot(h[10:-10], fp2,
        'r', label='Feasible trajectory')
plt.xlim(h[0],h[-1])
plt.ylim(0,42)
plt.xticks([hini +i for i in range(0,dt+1,2)], [(hini +i)%24 for i in range(0,dt+1,2)])
plt.legend(loc=1)
plt.xlabel('Time [h]')
#plt.ylabel('Storage [kWh]')
ax.set_title('(b) 25 kWh, 3.7 kVA', y=-0.25)




f.set_size_inches(7.47,3.3)#(9,4) #(11,4.76) 
plt.tight_layout()

plt.savefig(folder_images + 'storage_2ev_vpapersize.pdf')
plt.savefig(folder_images + 'storage_2ev_vpapersize.png')


#%% Do simulation for flex power profile along one day
## Simulation parameters
nweeks = 12
ndays = nweeks*7
step = 15

nevs = 1000

# EV PARAMS:
# Battery
batts = np.array((25,50,75))
# driving eff
driving_eff = ((batts * 0.09)+14)/100
# charging power [kw]
pcharger = 7.3
# alpha parameter of non systematic charging
n_if_needed = (1.31, 10000)

# arrival and departure data from electric nation
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)
bins=np.arange(0,24.5,0.5)
adwd = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe = dict(pdf_a_d=arr_dep_we.values, bins=bins)     

# Create grid
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs = {}
for i, b in enumerate(batts):
    for alpha in n_if_needed:
        evs[b, alpha] =grid.add_evs('b{}a{}'.format(b,alpha), nevs, 'dumb', 
                             charging_type='if_needed', charging_power=pcharger,
                             n_if_needed=alpha, batt_size=b,
                             arrival_departure_data_wd=adwd, 
                             arrival_departure_data_we=adwe,
                             driving_eff=driving_eff[i])
        alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
        for j, ev in enumerate(evs[b, alpha]):
            ev.n_if_needed = alphs[j]

grid.do_days()
print('\tSimulation finished')

#%% plot flex power profile along the day
case='_adEN'
flexpower={}
for idxs, evset in evs.items():
    flexpower[idxs[0], idxs[1]] = np.sum(np.array([ev.potential for ev in evset]),axis=0)
flexpower = pd.DataFrame(flexpower) / nevs / pcharger * 1.51
#flexpower['day'] = [x//(24 * 60/step) for x in flexpower.index]
flexpower['hour'] = [(x/(60/step)) % 24 for x in flexpower.index]

avgprof  = flexpower.groupby('hour').mean()

plt.figure()
idx = avgprof.index
labels=['Non-systematic, 25 kWh',
        'Non-systematic, 50 kWh','Non-systematic, 75 kWh','Systematic']
cs = [(25,1.31),(50,1.31),(75,1.31),(50,10000.0)]
ls = ['--','-.',':','-',]
for i, c in enumerate(cs):
    plt.plot(idx, avgprof[c], linestyle=ls[i], label=labels[i])
    
plt.xlabel('Hour')
plt.ylabel('EV share')
plt.xlim(0,23.75)
plt.ylim(0,1.05)
plt.legend(loc=1)
plt.xticks(np.arange(0,25,4))
plt.tight_layout()
plt.gca().figure.set_size_inches((6.1,4.1))
plt.savefig(folder_images + 'profile_evs{}.pdf'.format(case))
plt.savefig(folder_images + 'profile_evs{}.png'.format(case))


#%% Smart charging potential for percentiles of EVs
#
### Simulation parameters
#nweeks = 12
#ndays = nweeks*7
#step = 60
#
#nevs = 1000
#
## EV PARAMS:
## Battery
#batts = np.arange(15,100.1,2.5)
## driving eff
#driving_eff = ((batts * 0.09)+14)/100
## charging power [kw]
#pcharger = (3.6,7.2,10.7)
## alpha parameter of non systematic charging
#n_if_needed = (0.5,1.31,2,10000)
## Percents: Indicators for x%s of 'better behaved' EV users
#percents = [90,70,50]
## arrival and departure data from electric nation
#folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
#arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
#arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)
#bins=np.arange(0,24.5,0.5)
#adwd = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
#adwe = dict(pdf_a_d=arr_dep_we.values, bins=bins)     
#
## Create grid
#grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
#evs=grid.add_evs('test', nevs, 'dumb', 
#                 charging_type='if_needed',
#                 n_if_needed=n_if_needed, batt_size=50,
#                 arr_dep_wd=adwd, arr_dep_we=adwe)
#ds = [float(ev.dist_wd) for ev in evs]
#print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))
## TODO add ElNation deparr data
#
#dur_ch = {} # Average duration of charging part of sessions for all charging sessions
#dur_ch2 = {} # Average user duration of charging part of sessions for all users (first compute avg duration by user, then aggregate)
#dur_ss = {} # Average duration of charging sessions for all charging sessions
#dur_ss2 = {} # Average user duration of charging sessions for all users (first compute avg duration by user, then aggregate)
#sc_pot = {} # Idle time of all charging sessions
#sc_pot2 = {} # average idle time for users (first, compute idle time by user, then aggregate)
#time_connected = {} # Average time connected, in hours/day/user
#time_charging = {} # Average time spent charging, in hours/day/user
#flex_kwhh = {} #
#flex_kwh = {} #
#
#for k, alpha in enumerate(n_if_needed):
#    alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=nevs)
#    print('\talpha {}'.format(alpha))
#    for i, ev in enumerate(evs):
#        ev.n_if_needed = alphs[i]        
#    for i, b in enumerate(batts):
#        print('\t\tBatt {} kWh'.format(b))
#        grid.set_evs_param(param='batt_size', value=b)
#        grid.set_evs_param(param='driving_eff', value=driving_eff[i])
#        for j, pch in enumerate(pcharger):
#            print('\t\tCharging power {} kW'.format(pch))
#            grid.set_evs_param(param='charging_power', value=pch)
#            grid.reset()
#            grid.do_days()
#            _, m, _ = charging_sessions(grid)
#            
#            # I'll select the 'best' evs by looking at how much plugged time they have, i choose the x% with highest plug-in
#            pt = [ev.off_peak_potential.sum() for ev in evs]
#            cut = np.percentile(pt, percents)
#            for l, p in enumerate(percents):
#                pevs = [ev for i, ev in enumerate(evs) if pt[i]>=cut[l]]
#                sc_pot[b, pch, alpha, p] = 1-sum([ev.charging.sum() for ev in pevs])/sum([ev.off_peak_potential.sum() for ev in pevs])
#                sc_pot2[b, pch, alpha, p] = np.mean([1-ev.charging.sum()/ev.off_peak_potential.sum() for ev in pevs])
#                dur_ch[b, pch, alpha, p] = sum([ev.charging.sum() for ev in pevs])/sum([ev.ch_status.sum() for ev in pevs])/pch/(step/60)
#                dur_ch2[b, pch, alpha, p] = np.mean([ev.charging.sum()/ev.ch_status.sum() for ev in pevs])/pch/(step/60)
#                dur_ss[b, pch, alpha, p] = sum([ev.potential.sum() for ev in pevs])/sum([ev.ch_status.sum() for ev in pevs])/(step/60)/pch
#                dur_ss2[b, pch, alpha, p] = np.mean([ev.potential.sum()/ev.ch_status.sum() for ev in pevs])/(step/60)/pch
#                time_connected[b, pch, alpha, p] = sum([ev.off_peak_potential.sum() for ev in pevs])/pch*(step/60)/ndays/nevs
#                time_charging[b, pch, alpha, p] = sum([ev.charging.sum() for ev in pevs])/pch*(step/60)/ndays/len(pevs)
#                flex_kwhh[b, pch, alpha, p] = sum([(ev.up_flex-ev.dn_flex).sum() for ev in pevs])/ndays/len(pevs)
#                flex_kwh[b, pch, alpha, p] = sum([(ev.up_flex-ev.dn_flex).sum() for ev in pevs])/ndays/len(pevs)/dur_ss[b,pch,alpha,p]
#    print('\tSimulation finished for alpha {}'.format(alpha))
#            # hours available per ch session
##            tch_session[b, pch, alpha] = [ev.charging.sum() for ev in evs]
#
#def kpi_to_series_and_save(kpi, name='kpi', namesave=None):
#    if namesave is None:
#        namesave = name + '.csv'
#    kpi = pd.Series(kpi)
#    kpi.index.names = ['Batt_size', 'Charger_power', 'alpha', 'perc']
#    kpi.name = name
#    kpi.to_csv(folder_outputs + namesave, header=True)
#    return kpi
#
#sc_pot = kpi_to_series_and_save(sc_pot, 'Flex_potential', 'flex_pot_perc.csv')
#sc_pot2 = kpi_to_series_and_save(sc_pot2, 'Flex_potential', 'flex_pot2_perc.csv')
#dur_ch = kpi_to_series_and_save(dur_ch, 'charge_duration', 'avgchdur_perc.csv')
#dur_ch2 = kpi_to_series_and_save(dur_ch2, 'charge_duration', 'avgchdur2_perc.csv')
#dur_ss = kpi_to_series_and_save(dur_ss, 'session_duration', 'avgssdur_perc.csv')
#dur_ss2 = kpi_to_series_and_save(dur_ss2, 'session_duration', 'avgssdur2_perc.csv')
#time_connected = kpi_to_series_and_save(time_connected, 'time_connected', 'time_connected_perc.csv')
#time_charging = kpi_to_series_and_save(time_charging, 'time_charging', 'time_charging_perc.csv')
#flex_kwh = kpi_to_series_and_save(flex_kwh, 'flex_kwh', 'flex_kwh_perc.csv')
#flex_kwhh = kpi_to_series_and_save(flex_kwhh, 'flex_kwhh', 'flex_kwhh_perc.csv')