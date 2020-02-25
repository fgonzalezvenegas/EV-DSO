# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:47:16 2019
Run 1000 EVs and do histograms of # of charging sessions
@author: U546416
"""

import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as stats
import util

ndays = 7
step = 15

bins_dist = np.linspace(0, 100, num=51)
dist_function = np.sin(bins_dist[:-1]/ 100 * np.pi * 2) + 1
dist_function[10:15] = [0, 0, 0 , 0 , 0]
pdfunc = (dist_function/sum(dist_function)).cumsum()
jours = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim', 'Lun']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon']
def set_labels_fr(ax):
    ax.set_xlabel('Jours')
    ax.legend(loc=2)
    ax.set_xticklabels(jours)
    ax.set_ylabel('Puissance [kW]')
def set_labels_eng(ax):
    ax.set_xlabel('Days')
    ax.legend(loc=3)
    ax.set_xticklabels(days)
    ax.set_ylabel('Power [MW]')
        
def charging_sessions(grid):
    # compute hist
    nsessions = np.asarray([ev.ch_status.sum() + (ev.extra_energy > 0).sum()
                            for ev in grid.get_evs()])
    h_bins = np.append(np.arange(1,9,1), 100)
    hs = np.histogram(nsessions, h_bins)
    return hs[0]/sum(hs[0])

def global_data(grid):
    return grid.get_global_data()

def V1GFF(grid):
    return grid.get_global_data()['Flex_ratio']


def iterate_batt_size(grid, b_start=10, b_end=100, step=1, fx=charging_sessions):
    batts = np.arange(b_start, b_end, step)
    nb = len(batts)
    hists = []
    for i in range(nb):
        # set batt size
        grid.set_evs_param('batt_size', batts[i])
        #simulate
        grid.do_days()
        # compute hist for given function
        hists.append(fx(grid))
        print(batts[i], hists[-1])
        # reset grid
        grid.reset()
    return np.asarray(hists)

def lognormal_vector(s, m, loc=0, start=0, end=100, step=1, normalize=False):
    bs = np.arange(0-loc, 100-loc,1)
    v = 1/(np.sqrt(2*np.pi) * bs * s ) * np.exp(-((np.log(bs) - m)**2)/(2*s*s))
    v[np.isnan(v)] = 0
    if normalize:
        v = v/sum(v)
    return v

def plot_lognormal(s, m, loc=0, ax='', **kwargs):
    if ax=='':
        f, ax = plt.subplots()
    lognormal = lognormal_vector(s, m, loc, start=0, end=100, step=1)
    ax.plot(np.arange(0,100,1), lognormal, **kwargs)
    return lognormal

    
#%% Simulate!
b_start = 10
b_end = 100
batts = np.arange(b_start, b_end, 1)

ev_data = {'charging_power' : 7.2,
           'charging_type' : 'if_needed_sunday',
           'target_soc' : 1}

grid = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data)

hists = iterate_batt_size(grid, b_start=b_start, b_end=b_end, fx=charging_sessions)

#%% Do plots !
f, ax = plt.subplots()
labels=[str(i) + ' per week' for i in range(1,8)]
labels.append('8 or more per week')
for i in range(7):
    ax.plot(batts, hists.cumsum(axis=1)[:,i]*100, label=labels[i])
ax.set_xlim([10,100])
ax.set_ylim([0,105])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('Percentage of EVs')
ax.set_title('Charging sessions per week (cumulated) [%]')
plt.legend()
plt.grid(linestyle='--')
#
f, ax = plt.subplots()
ax.stackplot(batts, np.transpose(hists)*100, labels=labels)    
ax.set_xlim([10,99])
ax.set_ylim([0,100])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('Percentage of EVs')
ax.set_title('Charging sessions per week (cumulated) [%]')
ax.axvline(50, color='k', linestyle='--')
ax.text(x=51, y=30, s='Peugeot e208')
plt.legend(loc=4)

#%% Run charging sessions for 
print('loading data')
#load_by_comm, load_profiles, SS = evmodel.load_conso_ss_data()
hh, hw = evmodel.load_hist_data(file_hist_home='HistHomeModal_SS.csv')

print('iterating')
ss_urban = 'VANVES'
ss_peri = 'VERFEIL'

cdf_urban = hh.loc[ss_urban].cumsum()/hh.loc[ss_urban].sum()
cdf_peri = hh.loc[ss_peri].cumsum()/hh.loc[ss_peri].sum()


grid_peri = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid_urban = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)

grid_urban.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', 
                  cdf_dist_wd=cdf_urban, cdf_dist_we=cdf_urban, **ev_data)
grid_peri.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', 
                  cdf_dist_wd=cdf_peri, cdf_dist_we=cdf_peri, **ev_data)

hists_urban = iterate_batt_size(grid_urban, b_start, b_end)
hists_peri = iterate_batt_size(grid_peri, b_start, b_end)


#%% Plot
# Plot densities
bs = np.arange(0,100,1)
s = 0.736
m = 2.75
lognormale = 1/(np.sqrt(2*np.pi) * bs * s ) * np.exp(-((np.log(bs) - m)**2)/(2*s*s))
lognormale[0] = 0
lognormale = lognormale / lognormale.sum()

f, ax = plt.subplots()
ax.plot([i for i in range(0,100,2)], hh.loc[ss_urban]/hh.loc[ss_urban].sum(), label='Urban')
ax.plot([i for i in range(0,100,2)], hh.loc[ss_peri]/hh.loc[ss_peri].sum(), label='Periurban')
ax.plot(bs, lognormale*2, label='Lognormal')

#%% Plot n_th charging sessions
f, ax = plt.subplots()
nthch = 2
ax.plot(batts, hists.cumsum(axis=1)[:,nthch-1]*100, label='National average')
ax.plot(batts, hists_urban.cumsum(axis=1)[:,nthch-1]*100, label='Urban')
ax.plot(batts, hists_peri.cumsum(axis=1)[:,nthch-1]*100, label='Peri-urban')
ax.set_xlim([10,100])
ax.set_ylim([0,105])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('Percentage of EVs')
ax.set_title('EVs charging at most {} times per week [%]'.format(nthch))
ax.axvline(50, color='k', linestyle='--')
ax.text(x=51, y=30, s='Peugeot e208')
plt.legend()
plt.grid(linestyle='--')



#%% fit and re compute with fitted lognormal
data_urban = np.asarray([evmodel.random_from_pdf(cdf_urban, bins=[i*2 for i in range(51)]) for j in range(1000)])
data_peri = np.asarray([evmodel.random_from_pdf(cdf_peri, bins=[i*2 for i in range(51)]) for j in range(1000)])
data_ln = np.asarray([evmodel.random_from_pdf(lognormale.cumsum(), bins=[i for i in range(101)]) for j in range(1000)])

hu = np.asarray([((data_urban >= i*2) & (data_urban < (i+1)*2)).sum() for i in range(50)])
hpu = np.asarray([((data_peri >= i*2) & (data_peri < (i+1)*2)).sum() for i in range(50)])
hln = np.asarray([((data_ln >= i*2) & (data_ln < (i+1)*2)).sum() for i in range(50)])

fitu = stats.lognorm.fit(data_urban)
fitp = stats.lognorm.fit(data_peri)
fitln = stats.lognorm.fit(data_ln, floc=0)

lnp = lognormal_vector(fitp[0], np.log(fitp[2]), fitp[1], normalize=True)
lnu = lognormal_vector(fitu[0], np.log(fitu[2]), fitu[1], normalize=True)
lnln = lognormal_vector(fitln[0], np.log(fitln[2]), fitln[1], normalize=True)

f, ax = plt.subplots()

ax.bar(x=[i*2-0.5 for i in range(50)], height=hu/hu.sum(), color='b')
ax.bar(x=[i*2 for i in range(50)], height=hpu/hpu.sum(), color='g')
ax.bar(x=[i*2+0.5 for i in range(50)], height=hln/hpu.sum(), color='r')
ax.plot(lnu*2, color='b')
ax.plot(lnp*2, color='g')
ax.plot(lnln*2, color='r')


#%% Instantiate grid and simulate
grid_periln = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid_urbanln = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)

grid_urbanln.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', 
                  cdf_dist_wd=cdf_urban, cdf_dist_we=cdf_urban, **ev_data)
grid_periln.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', 
                  cdf_dist_wd=cdf_peri, cdf_dist_we=cdf_peri, **ev_data)

hists_urbanln = iterate_batt_size(grid_urbanln, b_start, b_end, fx=charging_sessions)
hists_periln = iterate_batt_size(grid_periln, b_start, b_end, fx=charging_sessions)

#%% Plot n_th charging sessions
f, ax = plt.subplots()
nthch = 2
ax.plot(batts, hists.cumsum(axis=1)[:,nthch-1]*100, label='National average')
ax.plot(batts, hists_urban.cumsum(axis=1)[:,nthch-1]*100, label='Urban')
ax.plot(batts, hists_peri.cumsum(axis=1)[:,nthch-1]*100, label='Peri-urban')
ax.plot(batts, hists_urbanln.cumsum(axis=1)[:,nthch-1]*100, label='Urban_Fitted')
ax.plot(batts, hists_periln.cumsum(axis=1)[:,nthch-1]*100, label='Peri-urban_Fitted')
ax.set_xlim([10,100])
ax.set_ylim([0,105])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('Percentage of EVs')
ax.set_title('EVs charging at most {} times per week [%]'.format(nthch))
ax.axvline(50, color='k', linestyle='--')
ax.text(x=51, y=30, s='Peugeot e208')
plt.legend()
plt.grid(linestyle='--')

#f, ax = plt.subplots()
#plot_lognormal(fitu[0], np.log(fitu[2]), loc=fitu[1], ax=ax, label='Urban', color='b')
#plot_lognormal(fitp[0], np.log(fitp[2]), loc=fitp[1], ax=ax, label='Periurban', color='g')
#plot_lognormal(fitln[0], np.log(fitln[2]), loc=0, ax=ax, label='LogNormal', color='k')
#ax.bar(x=[i*2-0.5 for i in range(50)], height=hu/2000, color='b')
#ax.bar(x=[i*2 for i in range(50)], height=hpu/2000, color='g')
#ax.bar(x=[i*2+0.5 for i in range(50)], height=hln/2000, color='k')
#ax.legend()

#%% Simulate V1GFlexibility factor
grid7 = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid7.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', 
              charging_power= 7.2, charging_type='if_needed_sunday',                  
              arrival_departure_data_we = {'mu_arr':7, 'mu_dep':23, 'std_arr':2, 'std_dep':2},
              arrival_departure_data_wd = {'mu_arr':7, 'mu_dep':21, 'std_arr':2, 'std_dep':2})
grid3 = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid3.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', 
              charging_power= 3.6, charging_type='if_needed_sunday',
              arrival_departure_data_we = {'mu_arr':7, 'mu_dep':23, 'std_arr':2, 'std_dep':2},
              arrival_departure_data_wd = {'mu_arr':7, 'mu_dep':21, 'std_arr':2, 'std_dep':2})

gdata3 = iterate_batt_size(grid3, b_start=b_start, b_end=b_end, fx=global_data)
sc_factor3 = [gdata3[i]['Flex_ratio'] for i in range(len(gdata3))]
chen3 = [gdata3[i]['Tot_ev_charge'] for i in range(len(gdata3))]
chex3 = [gdata3[i]['Extra_charge'] for i in range(len(gdata3))]
gdata7 = iterate_batt_size(grid7, b_start=b_start, b_end=b_end, fx=global_data)
sc_factor7 = [gdata7[i]['Flex_ratio'] for i in range(len(gdata7))]
chen7 = [gdata7[i]['Tot_ev_charge'] for i in range(len(gdata7))]
chex7 = [gdata7[i]['Extra_charge'] for i in range(len(gdata7))]

#%% plot
f, ax = plt.subplots()
ax.plot(np.arange(10,100,1), sc_factor3, label='Charger 3.6 kW')
ax.plot(np.arange(10,100,1), sc_factor7, label='Charger 7.2 kW')
ax.set_title('Flexibility factor (' + '$FF_{V1G}$'+ ') according to battery sizes')
ax.set_xlim([10,100])
ax.set_ylim([0,1])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('$FF_{V1G}$')
plt.grid(linestyle='--')
plt.legend()

#%% Run charging sessions for Different target SOCs

print('iterating')
ev_data = {'charging_power' : 7.2,
           'charging_type' : 'if_needed_sunday'
           }

grid1 = EVmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid08 = EVmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid05 = EVmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)

grid1.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data, target_soc=1)
grid08.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data, target_soc=0.8)
grid05.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data, target_soc=0.5)


hists1 = iterate_batt_size(grid1, b_start, b_end)
hists08 = iterate_batt_size(grid08, b_start, b_end)
hists05 = iterate_batt_size(grid05, b_start, b_end)


#%%
#util.self_reload(EVmodel)
#import EVmodel
#print('iterating')
#ev_data = {'charging_power' : 7.2,
#           'charging_type' : 'if_needed_sunday'
#           }
#
##grid1 = EVmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
#grid08 = EVmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
##grid05 = EVmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
#
##grid1.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data, target_soc=1)
#grid08.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data, target_soc=0.8)
##grid05.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data, target_soc=0.5)
#
#
##hists1 = iterate_batt_size(grid1, b_start, b_end)
#hists08 = iterate_batt_size(grid08, b_start, b_end)
#hists05 = iterate_batt_size(grid05, b_start, b_end)
#%% Plot n_th charging sessions
f, ax = plt.subplots()
nthch = 2
ax.plot(batts, hists1.cumsum(axis=1)[:,nthch-1]*100, label='Target SOC=1.0')
ax.plot(batts, hists08.cumsum(axis=1)[:,nthch-1]*100, label='Target SOC=0.8')
ax.plot(batts, hists05.cumsum(axis=1)[:,nthch-1]*100, label='Target SOC=0.5')
ax.set_xlim([10,100])
ax.set_ylim([0,105])
ax.set_xlabel('Battery size [kWh]')
ax.set_ylabel('Percentage of EVs')
ax.set_title('EVs charging at most {} times per week [%]'.format(nthch))
ax.axvline(50, color='k', linestyle='--')
ax.text(x=51, y=30, s='Peugeot e208')
plt.legend()
plt.grid(linestyle='--')

