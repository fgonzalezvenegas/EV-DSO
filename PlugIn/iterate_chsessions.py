# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:26:59 2020

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

folder_outputs = r'c:\user\U546416\Documents\PhD\Data\Plug-in Model\Results\\'
#%% Simulation to get ch_sessions per week
## Simulation parameters
nweeks = 5
ndays = nweeks*7
step = 60

nevs = 1000

# batterie sizes
batts = np.arange(15,100.1,2.5)
# driving efficiency
#md = (0.02/(25))
#b0 = 25
#eff0 = 0.16
#driving_eff = [eff0 + (b-b0)*md  for b in batts]
driving_eff = ((batts * 0.09)+14)/100
# n_if_needed
n0 = 0.6
b0 = 25
mn = (1.6-n0)/(50)
n_if_needed = [n0 + (b-b0)*mn for b in batts]
#n_if_needed = 0.5

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

for n in n_if_needed:
    for i, b in enumerate(batts):
        print('\tBatt {} kWh'.format(b))
        grid.set_evs_param(param='batt_size', value=batts[i])
        grid.set_evs_param(param='driving_eff', value=driving_eff[i])
        try: 
            len(n_if_needed)
            grid.set_evs_param(param='n_if_needed', value=n)
        except:
            pass
        grid.reset()
        grid.do_days()
        hists[b], mean[b], median[b] = charging_sessions(grid, stats=True)
        
    hists = pd.DataFrame(hists).T
    hists.index.name='Batt_size'
    stats = pd.DataFrame([mean, median], index=['Mean', 'Median']).T
    stats.index.name='Batt_size'
    
    #folder_save = r'c:\user\U546416\Pictures\ElectricNation\PlugIn\Tests\\'
    n = 'var' if type(n_if_needed)==list else n_if_needed
    hists.to_csv(folder_outputs + 'hists_alpha{}.csv'.format(n))
    stats.to_csv(folder_outputs + 'stats_alpha{}.csv'.format(n))

#%% Do Histogram of charging sessions

labels=['{} to {} per week'.format(i,i+1) for i in range(7)]
labels.append('More than 7')
#
f, ax = plt.subplots()
ax.stackplot(batts, hists.T*100, labels=labels, alpha=0.8)    
ax.set_xlim([15,97])
ax.set_ylim([0,100])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('Percentage of EVs')
ax.set_title('Charging sessions per week (cumulated)')
#ax.axvline(50, color='k', linestyle='--')
#ax.text(x=51, y=30, s='Peugeot e208')
plt.legend(loc=4)

#%% Identifying peak load
t = [time.time()]
## Simulation parameters
##### Each round of Battery/Charger/alpha for all fleet sizes (67 between 1-10k),
##### considering 1000 simulated fleets to get peak load takes around 25min
##### Reducing simulated fleets to 500 reduces time by 50%
nweeks = 10
ndays = nweeks*7
step = 30

nevs = 10000

# EV PARAMS:
# Battery
batts = (25,50,75)
# driving eff
driving_eff = [eff0 + (b-b0)*md  for b in batts]
# charging power [kw]
pcharger = (3.6,7)
# fleet sizes to study, between 1 to 1000
nevs_fleets = list(set([int(n) for n in np.logspace(np.log10(1), np.log10(10000), 67)]))
nevs_fleets.sort()
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,1,1.6,100)
# number of fleets to analyze for pmax
nfleets = 1000
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
                 arr_dep_wd=adwd, arr_dep_we=adwe)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))
# TODO add ElNation deparr data

peak = {}

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, nif in enumerate(n_if_needed):
            grid.set_evs_param(param='n_if_needed', value=nif)
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(nif))
            _, m, _ = charging_sessions(grid)
            ta = [time.time()]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                # get charging profiles
                ch_profs = np.array([ev.charging for ev in evs])
                pmax = pch * nev
                p=0
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    p = max(p,ch_profs[idx, :].sum(axis=0).max())
                    if p >= pmax:
                        break
                peak[b,pch,nif,nev] = p/nev
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(nif, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))
peak = pd.Series(peak)
peak.index.names = ['Batt_size', 'Charger_power', 'alpha', 'fleet_size']
peak.name = 'Peak_load'
peak.to_csv(folder_outputs + 'Peak_load.csv', header=True)

#%% Identifying peak load version2
t = [time.time()]
## Simulation parameters
nweeks = 10
ndays = nweeks*7
step = 30

nevs = 10000

conf = 0.99

# EV PARAMS:
# Battery
batts = (25,50,75)
# driving eff
driving_eff = [eff0 + (b-b0)*md  for b in batts]
# charging power [kw]
pcharger = (3.6,7)
# fleet sizes to study, between 1 to 1000
nevs_fleets = list(set([int(n) for n in np.logspace(np.log10(1), np.log10(10000), 67)]))
nevs_fleets.sort()
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,1,1.6,100)
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
                 arr_dep_wd=adwd, arr_dep_we=adwe)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

peak = {}

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, nif in enumerate(n_if_needed):
            grid.set_evs_param(param='n_if_needed', value=nif)
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(nif))
            _, m, _ = charging_sessions(grid)
            ta = [time.time()]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                # get charging profiles
                ch_profs = np.array([ev.charging for ev in evs])
                p = []
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    p.append(ch_profs[idx, :].sum(axis=0).max())
                p.sort()
                peak[b,pch,nif,nev] = p[int(nfleets * conf)-1]/nev
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(nif, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
t.append(time.time())
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))
peak = pd.Series(peak)
peak.index.names = ['Batt_size', 'Charger_power', 'alpha', 'fleet_size']
peak.name = 'Peak_load'
peak.to_csv(folder_outputs + 'Peak_load_v2.csv', header=True)

#%% Identifying peak load version2 - ToU charging
t = [time.time()]
## Simulation parameters
nweeks = 10
ndays = nweeks*7
step = 30

nevs = 10000

conf = 0.99

# EV PARAMS:
# Battery
batts = (25,50,75)
# driving eff
driving_eff = [eff0 + (b-b0)*md  for b in batts]
# charging power [kw]
pcharger = (3.6,7)
# fleet sizes to study, between 1 to 1000
nevs_fleets = list(set([int(n) for n in np.logspace(np.log10(1), np.log10(10000), 67)]))
nevs_fleets.sort()
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,1,1.6,100)
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
                 batt_size=50,
                 arr_dep_wd=adwd, arr_dep_we=adwe,
                 tou_ini=22, tou_end=8, tou=True)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))

peak = {}

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, nif in enumerate(n_if_needed):
            grid.set_evs_param(param='n_if_needed', value=nif)
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(nif))
            _, m, _ = charging_sessions(grid)
            ta = [time.time()]
            for j, nev in enumerate(nevs_fleets):
                if nev in [28,231,705,1072,2222,4977,7564]:
                    ta.append(time.time())
                    print('\t\t\tTesting {} EV fleets, dt {:.0f}s'.format(nev, ta[-1]-ta[-2]))
                # TODO compute pmax for given nev
                # get charging profiles
                ch_profs = np.array([ev.charging for ev in evs])
                p = []
                # get peak load for f fleets of nev size
                for f in range(nfleets):
                    idx = np.random.choice(range(nevs), size=nev, replace=False)
                    p.append(ch_profs[idx, :].sum(axis=0).max())
                p.sort()
                peak[b,pch,nif,nev] = p[int(nfleets * conf)-1]/nev
            ta.append(time.time())
            print('\t\tFinished alpha {}, dt {}:{:.0f}'.format(nif, int((ta[-1]-ta[0])/60),(ta[-1]-ta[0])%60))
    t.append(time.time())
    print('\tFinished battery {}, dt {}:{:.0f}'.format(b, int((t[-1]-t[-2])/60),(t[-1]-t[-2])%60))
t.append(time.time())
print('Finished all, dt {}:{:.0f}'.format(int((t[-1]-t[0])/60),(t[-1]-t[0])%60))
peak = pd.Series(peak)
peak.index.names = ['Batt_size', 'Charger_power', 'alpha', 'fleet_size']
peak.name = 'Peak_load'
peak.to_csv(folder_outputs + 'Peak_load_ToU.csv', header=True)


#%% Plot peak load results
#peak = pd.read_csv(folder_outputs + 'Peak_load.csv', engine='python', index_col=[0,1,2,3]).Peak_load
#peak = pd.read_csv(folder_outputs + 'Peak_load_v2.csv', engine='python', index_col=[0,1,2,3]).Peak_load
peak = pd.read_csv(folder_outputs + 'Peak_load_ToU.csv', engine='python', index_col=[0,1,2,3]).Peak_load

# Vary all chargers, all batt sizes
f, ax = plt.subplots()
ax.set_xscale('log')
battery = [25,50,75]
pch = [3.6,7]
alpha = 0.5
ls = ['-','--',':','-.']
for i, p in enumerate(pch):
    for j, b in enumerate(battery):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,alpha,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label='{:.1f} kW; {} kWh'.format(p,b))
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
    pch = [3.6,7]
    alpha = peak.index.levels[2]
    ls = ['-','--',':','-.',(0, (3, 1, 1, 1))]
    for i, p in enumerate(pch):
        for j, a in enumerate(alpha):
            #peak[battery, charger, alpha, :]
            k=peak.loc[b,p,a,:]
            plt.plot(k.index.get_level_values('fleet_size'), k, 
                     linestyle=ls[j], label=r'{:.1f} kW; $\alpha$={}'.format(p,a))
    plt.legend()
    plt.xlabel('Fleet size')
    plt.ylabel('Power [kW]')
    plt.xlim((1,10000))
    plt.ylim((0,7.5))
    plt.grid(linestyle='--', axis='x')
    f.suptitle('{} kWh batteries'.format(b))

#%% 
peak = pd.read_csv(folder_outputs + 'Peak_load_v2.csv', engine='python', index_col=[0,1,2,3]).Peak_load
peaktou = pd.read_csv(folder_outputs + 'Peak_load_ToU.csv', engine='python', index_col=[0,1,2,3]).Peak_load

## Vary all chargers, all batt sizes
f, ax = plt.subplots()
ax.set_xscale('log')
battery = [25,50,75]
pch = [3.6,7]
alpha = 0.5
ls = ['-','--',':','-.']

for j, b in enumerate(battery):
    f, ax = plt.subplots()
    ax.set_xscale('log')
    for i, p in enumerate(pch):
        #peak[battery, charger, alpha, :]
        k=peak.loc[b,p,alpha,:]
        tou=peaktou.loc[b,p,alpha,:]
        plt.plot(k.index.get_level_values('fleet_size'), k, 
                 linestyle=ls[j], label='{:.1f} kW; Non-systematic'.format(p,b))
        plt.plot(tou.index.get_level_values('fleet_size'), tou, 
                 linestyle=ls[j], label='{:.1f} kW; Systematic'.format(p,b))
        plt.plot(tou.index.get_level_values('fleet_size'), tou, 
                 linestyle=ls[j], label='{:.1f} kW; Systematic'.format(p,b))
plt.legend()
plt.xlabel('Fleet size')
plt.ylabel('Power [kW]')
plt.xlim((1,10000))
plt.ylim((0,7.5))
plt.grid(linestyle='--', axis='x')
f.suptitle('{} kWh batteries'.format(b))

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
driving_eff = [eff0 + (b-b0)*md  for b in batts]
# charging power [kw]
pcharger = (3.6,7)
# alpha parameter of non systematic charging
n_if_needed = (0,0.5,1,1.6,100)

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
                 arr_dep_wd=adwd, arr_dep_we=adwe)
ds = [float(ev.dist_wd) for ev in evs]
print('EV distances, mean {:.1f}, median {:.1f}'.format(np.mean(ds)*2, np.median(ds)*2))
# TODO add ElNation deparr data

dur_ch = {} # Average duration of charging sessions for all charging sessions
dur_ch2 = {} # Average user duration of charging sessions for all users (first compute avg duration by user, then aggregate)
sc_pot = {} # Idle time of all charging sessions
sc_pot2 = {} # average idle time for users (first, compute idle time by user, then aggregate)
time_connected = {} # Average time connected, in hours/day/user
time_charging = {} # Average time spent charging, in hours/day/user

for i, b in enumerate(batts):
    print('\tBatt {} kWh'.format(b))
    grid.set_evs_param(param='batt_size', value=b)
    grid.set_evs_param(param='driving_eff', value=driving_eff[i])
    for j, pch in enumerate(pcharger):
        print('\tCharging power {} kW'.format(pch))
        grid.set_evs_param(param='charging_power', value=pch)
        for k, nif in enumerate(n_if_needed):
            grid.set_evs_param(param='n_if_needed', value=nif)
            grid.reset()
            grid.do_days()
            print('\t\tSimulation finished for alpha {}'.format(nif))
            _, m, _ = charging_sessions(grid)
            sc_pot[b, pch, nif] = grid.get_global_data()['Flex_ratio']
            sc_pot2[b, pch, nif] = np.mean([1-ev.charging.sum()/ev.off_peak_potential.sum() for ev in evs])
            dur_ch[b, pch, nif] = sum([ev.charging.sum() for ev in evs])/sum([ev.ch_status.sum() for ev in evs])/pch*(step/60)
            dur_ch2[b, pch, nif] = np.mean([ev.charging.sum()/ev.ch_status.sum() for ev in evs])/pch*(step/60)
            time_connected[b, pch, nif] = sum([ev.off_peak_potential.sum() for ev in evs])/pch*(step/60)/ndays/nevs
            time_charging[b, pch, nif] = sum([ev.charging.sum() for ev in evs])/pch*(step/60)/ndays/nevs
            

sc_pot = pd.Series(sc_pot)
sc_pot.index.names = ['Batt_size', 'Charger_power', 'alpha']
sc_pot.name = 'Flex_potential'
sc_pot.to_csv(folder_outputs + 'flex_pot.csv', header=True)
sc_pot2 = pd.Series(sc_pot2)
sc_pot2.index.names = ['Batt_size', 'Charger_power', 'alpha']
sc_pot2.name = 'Flex_potential'
sc_pot2.to_csv(folder_outputs + 'flex_pot2.csv', header=True)
dur_ch = pd.Series(dur_ch)
dur_ch.index.names = ['Batt_size', 'Charger_power', 'alpha']
dur_ch.name = 'Charge_duration'
dur_ch.to_csv(folder_outputs + 'avgchdur.csv', header=True)
dur_ch2 = pd.Series(dur_ch2)
dur_ch2.index.names = ['Batt_size', 'Charger_power', 'alpha']
dur_ch2.name = 'Charge_duration'
dur_ch2.to_csv(folder_outputs + 'avgchdur2.csv', header=True)
time_connected = pd.Series(time_connected)
time_connected.index.names = ['Batt_size', 'Charger_power', 'alpha']
time_connected.name = 'Time_connected'
time_connected.to_csv(folder_outputs + 'time_connected.csv', header=True)
time_charging = pd.Series(time_charging)
time_charging.index.names = ['Batt_size', 'Charger_power', 'alpha']
time_charging.name = 'Time_charging'
time_charging.to_csv(folder_outputs + 'time_charging.csv', header=True)


#%% Plot SC potential
sc_pot = pd.read_csv(folder_outputs + 'flex_pot.csv', engine='python', index_col=[0,1,2]).Flex_potential
sc_pot2 = pd.read_csv(folder_outputs + 'flex_pot2.csv', engine='python', index_col=[0,1,2]).Flex_potential
dur_ch = pd.read_csv(folder_outputs + 'avgchdur.csv', engine='python', index_col=[0,1,2]).Charge_duration
dur_ch2 = pd.read_csv(folder_outputs + 'avgchdur2.csv', engine='python', index_col=[0,1,2]).Charge_duration
time_connected = pd.read_csv(folder_outputs + 'time_connected.csv', engine='python', index_col=[0,1,2]).Time_connected
time_charging = pd.read_csv(folder_outputs + 'time_charging.csv', engine='python', index_col=[0,1,2]).Time_charging

# Vary all batts, plot diff chargers. alpha 0.5
f, ax = plt.subplots()
b = np.arange(15,100.1,2.5)
pch = [3.6,7]
alpha = 0.5
ls = ['-','--',':','-.']
for i, p in enumerate(pch):
    #sc_pot[battery, charger, alpha]
    k=sc_pot.loc[:,p,alpha]
    plt.plot(k.index.get_level_values('Batt_size'), k, 
             linestyle=ls[i], label=r'{} kW'.format(p))
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
                 linestyle=ls[i], label=r'Time connected - {} kW'.format(p))
        j=time_charging.loc[:,p,a]
        plt.plot(k.index, j, 
                 linestyle=ls[i], label=r'Time charging - {} kW'.format(p))
        plt.fill_between(k.index.get_level_values('Batt_size'), j,k, color='y', alpha=0.1)
    plt.legend()
    plt.xlabel('Battery_size [p.u.]')
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
alpha = [0,.5,1,1.6]
ls = ['-','--',':','-.']
for i, p in enumerate(pch):
    for j, a in enumerate(alpha):
        #sc_pot[battery, charger, alpha]
        k=sc_pot.loc[:,p,a]
        plt.plot(k.index.get_level_values('Batt_size'), k, 
                 linestyle=ls[j], label=r'{} kW; $\alpha$={}'.format(p,a))
plt.legend()
plt.xlabel('Battery size [kWh]')
plt.ylabel('Flexibility ratio [p.u.]')
plt.xlim((15,100))
plt.ylim((0,1))
plt.grid(linestyle='--')

#%% Do all Histogram of charging sessions
#consolidate data
fullhists = pd.DataFrame()
fullstats = pd.DataFrame()
n_if_needed = [0,0.5,1,1.6,'var']
for n in n_if_needed:
    h = pd.read_csv(folder_outputs + 'hists_alpha{}.csv'.format(n), engine='python')
    if '8' in h.columns:
        h['7']=h['7']+h['8']
        h.drop('8', inplace=True, axis=1)
    if 'Unnamed: 0' in h.columns:
        h.columns = ['Batt_size'] + list(h.columns)[1:]
        h.set_index('Batt_size', drop=True).to_csv(folder_outputs + 'hists_alpha{}.csv'.format(n))
    s = pd.read_csv(folder_outputs + 'stats_alpha{}.csv'.format(n), engine='python')
    if 'Unnamed: 0' in s.columns:
        s.columns = ['Batt_size'] + list(s.columns)[1:]
        s.set_index('Batt_size', drop=True).to_csv(folder_outputs + 'stats_alpha{}.csv'.format(n))
    h['alpha'] = n
    s['alpha'] = n
    fullhists = fullhists.append(h)
    fullstats = fullstats.append(s)
fullhists.set_index(['alpha', 'Batt_size'], inplace=True, drop=True)
fullstats.set_index(['alpha', 'Batt_size'], inplace=True, drop=True)
fullhists.to_csv(folder_outputs + 'hists_all.csv')
fullstats.to_csv(folder_outputs + 'stats_all.csv')    
 
f, axstats = plt.subplots()
f, axmedian = plt.subplots()
labels=['{} to {} per week'.format(i,i+1) for i in range(7)]
labels.append('More than 7')
for n in n_if_needed:
    f, ax = plt.subplots()
    ax.stackplot(batts, fullhists.loc[n, :].T*100, labels=labels, alpha=0.8)    
    ax.set_xlim([15,97])
    ax.set_ylim([0,100])
    ax.set_xlabel('Battery size [kWh]')
    ax.set_ylabel('Percentage of EVs')
    ax.set_title(r'Charging sessions per week (cumulated); $\alpha$={}'.format(n))
    #ax.axvline(50, color='k', linestyle='--')
    #ax.text(x=51, y=30, s='Peugeot e208')
    plt.legend(loc=4)
    axstats.plot(batts, fullstats.loc[n,:].Mean, label=r'$\alpha$={}'.format(n))
    axmedian.plot(batts, fullstats.loc[n,:].Median, label=r'$\alpha$={}'.format(n))
axstats.legend()
axstats.set_xlabel('Battery size [kWh]')
axstats.set_ylabel('Charging sessions per week')
axstats.set_xlim(15,100)
axstats.grid(linestyle='--')
axmedian.legend()
axmedian.set_xlabel('Battery size [kWh]')
axmedian.set_ylabel('Charging sessions per week')
axmedian.set_xlim(15,100)
axmedian.grid(linestyle='--')
#%%
