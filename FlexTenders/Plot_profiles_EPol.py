# -*- coding: utf-8 -*-
"""
Created on August 2020
Script that iterates over a range of EV fleet sizes.

Use case for Energy Policy paper

@author: U546416
"""


import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as sts
import time
import util
import flex_payment_funcs as fpf
import pandas as pd

#%% 0 Params
t = []
t.append(time.time())
# Case 1 : Company fleet IN-BUILDING
    # This means they will charge during the night
# Case 2: Commuters with medium probability of plugin (n=1)
# Case 3: Commuters with medium probability of plugin (n=5)
# Case x: JPL

n_evs = 30
ovn=True

# FLEET PARAMS
batt_size = 50
ch_power = 7

t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))  

# Load arr_dep_distr
folder_acn = r'c:\user\U546416\Documents\PhD\Data\Caltech_ACN\Outputs\\'
JPL_arr_dep = pd.read_csv(folder_acn + 'JPL_wd.csv', index_col=0)
Cal_arr_dep = pd.read_csv(folder_acn + 'Caltech_wd.csv', index_col=0) 
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)

general_params = dict(charging_power=ch_power,
                      batt_size=batt_size,
                      ovn=ovn,                  #because we're doing overnight
                      target_soc = 0.8) 

# Arrival departure data for high plug in. Modifies schedules to have more overnight connections
adwd_HP = arr_dep_wd.copy().values
adwe_HP = arr_dep_we.copy().values
for i in range(48):
    for j in range(i, 48):
        adwd_HP[i,j] = adwd_HP[i,j]/2 
        adwe_HP[i,j] = adwe_HP[i,j]/2 
adwd_HP = adwd_HP/adwd_HP.sum()
adwe_HP = adwe_HP/adwe_HP.sum()

# creating structures to be used by EVmodel for Commuters
bins=np.arange(0,24.5,0.5)
adwd_LP = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe_LP = dict(pdf_a_d=arr_dep_we.values, bins=bins)     
adwd_HP = dict(pdf_a_d=adwd_HP, bins=bins)
adwe_HP = dict(pdf_a_d=adwe_HP, bins=bins)

n_proba_high=2  # This gives 5.5 plugs per week
n_proba_low=0.15 #Approx factor to get a median of 2.5 times plug in ratio, w/ 40kWh & O.Borne data

commuter_params_LP = dict(arrival_departure_data_wd=adwd_LP,
                          arrival_departure_data_we=adwe_LP,
                          n_if_needed=n_proba_low)
commuter_params_HP = dict(arrival_departure_data_wd=adwd_HP,
                          arrival_departure_data_we=adwe_HP,
                          n_if_needed=n_proba_high)
              
# Data for Company fleet.
# Using data from Parker project - Forsyning.
# See Lea Sass Berthou Ms Thesis at DTU, "Flexibility Profiles for EV Users"
# See also conf paper "Added Value of Individual Flexibility Profiles of Electric Vehicle Users For
# Ancillary Services", PBA et al, 2018

# Departure (morning). Lognormal with mu=1.04, sigma=0.53, shift=5h30
shift = 5.5
x = np.arange(shift,24,0.5)
dep_time = np.concatenate((np.zeros(int(shift*2)), sts.lognorm.pdf(x-shift, s=0.53, loc=0, scale=np.exp(1.04))))
#plt.plot(np.concatenate(([0],x)), np.concatenate(([0], dep_time)))

# Arrival (afternoon). Normal with mu=13.2, sigma=1.82
x = np.arange(0,24, 0.5)
arr_time = sts.norm.pdf(x, loc=13.2, scale=1.82)
#plt.plot(x, arr_time)

# Doing the 2d plot
ad_comp = np.array([arr_time[i] * dep_time for i in range(len(arr_time))])
for i in range(48):
    for j in range(i, 48):
        ad_comp[i,j] = 0 
ad_comp = ad_comp/ad_comp.sum()

# Energy usage = daily needs in kWh are lognorm with mu=1.44, sigma=0.79
# Transformed at daily distances, considering driving efficiency of 0.2kWh/km:
# dist/trip follows a lognorm of mu=1.44+ln(1/0.2/2), sigma=0.79
#dist = stats.lognorm.pdf(bins/0.4, loc=0, s=0.79, scale=np.exp(1.44)/0.4)
#dist = dist / dist.sum()
#plt.plot(bins, energy)
#plt.plot(bins/0.4, dist)

company_params = dict(arrival_departure_data_we=dict(pdf_a_d=ad_comp,
                                                     bins=bins),
                      arrival_departure_data_wd=dict(mu_arr=15, mu_dep=9,
                                                     std_arr=1, std_dep=1),
                      dist_wd=dict(loc=0, s=0.79, scale=np.exp(1.44)/0.4),
                      dist_we=dict(cdf=sts.norm.cdf(np.arange(1,100,2), 
                                                      loc=0.01, 
                                                      scale=0.0005)),
                      n_if_needed=100  #100, always, 0, never,                      
                      )
t = []
t.append(time.time())



# DSO SERVICE PARAMS:
service_time= [30,60,120]        # minutes for which the service should be provided
# I will select how many EVs do i need
min_bid = 10            # kW
#av_window = [17, 21]    # Availability window
av_days = 'wd'          # Weekdays (wd), weekends (we) only, all
t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))                             


#%% 1 Compute EVs simulation: 
t.append(time.time())             
# SIMS PARAMS:
nweeks = 50
ndays = 7 * nweeks + 1 # 50 weeks, + one extra day
step = 5 # minutes

# DO SIMULATIONS (takes about 1-2min)
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
grid.add_evs(nameset='Company', n_evs=n_evs, ev_type='dumb', 
             flex_time=service_time,
             **general_params,
             **company_params)
grid.add_evs(nameset='Commuter_HP', n_evs=n_evs, ev_type='dumb', 
             flex_time=service_time,
             **general_params,
             **commuter_params_HP)
grid.add_evs(nameset='Commuter_LP', n_evs=n_evs, ev_type='dumb', 
             flex_time=service_time,
             **general_params,
             **commuter_params_LP)
for i, ev in enumerate(grid.evs_sets['Commuter_LP']):
    grid.evs_sets['Commuter_HP'][i].dist_we = ev.dist_we
    grid.evs_sets['Commuter_HP'][i].dist_wd = ev.dist_wd
    
grid.do_days()
grid.plot_ev_load(day_ini=7, days=14)
grid.plot_flex_pot(day_ini=7, days=14)
t.append(time.time())                              
print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))

#%%
# Do fleets:
nfleets= 5
av_window = [0, 24] 
aw_s = str(av_window[0]) + '_' + str(av_window[1])
if ovn:
    shift = 12
else:
    shift = 0    
av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
if av_window == [0, 24]:
    av_window_idxs = [0, int(24*60/step)]
av_days = 'weekdays'

sets = grid.ev_sets
ch_profs = {}
dn_profs = {}
fch = {}
fdn = {}
for s in sets:
    ch_profs[s], dn_profs[s] = fpf.get_ev_profs(grid, ovn=ovn, av_days=av_days, nameset=s)
    fch[s], fdn[s] = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], 
                                         nfleets=1, nevs_fleet=30)

#%% Plot avg profiles and range
conf = 0.95
x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

NUM_COLORS = 6
cm = plt.get_cmap('Paired')
colors = [cm(1.*(i*2+1)/(NUM_COLORS*2)) for i in range(NUM_COLORS)]

(k, nd, ns) = fch[s].shape

f,ax = plt.subplots()
for i, s in enumerate(sets):
    ax.plot(x,fch[s][0].mean(axis=0), color=colors[i], label='Charging ' + s)
    ax.fill_between(x, np.sort(fch[s][0], axis=0)[int(nd*conf)], 
                    np.sort(fch[s][0], axis=0)[int(nd*(1-conf))],
                    alpha=0.1, color=colors[i], label='_90% range')

plt.xlim(0,24)
plt.ylim(0,120)
plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.xticks(xticks, (xticks-12)%24)
plt.grid('--', alpha=0.5)
plt.axvspan(5,8, color='yellow', alpha=0.3, label='Evening window')
plt.legend()
#%% Avg profs with V2G potential
sets_str = [s.replace('_', ' ') for s in sets]
mrks = ['x','','o']
mrksize = [4,0,3]
ls = ['--',':','-.']
f,ax = plt.subplots()
for i, s in enumerate(sets):
    ax.plot(x,fch[s][0].mean(axis=0), color=colors[i], label='Charging ' + sets_str[i], 
            marker=mrks[i], markersize=mrksize[i], markevery=12)
    ax.fill_between(x, np.sort(fch[s][0], axis=0)[int(nd*conf)], 
                    np.sort(fch[s][0], axis=0)[int(nd*(1-conf))],
                    alpha=0.1, color=colors[i], label='_90% range')

for i, s in enumerate(sets):
    ax.plot(x, fdn[s][0][0].mean(axis=0), '--', color=colors[i], label='V2G potential ' + sets_str[i],
            linestyle=ls[i])
    ax.fill_between(x, np.sort(fdn[s][0][0], axis=0)[int(nd*conf)], 
                np.sort(fdn[s][0][0], axis=0)[int(nd*(1-conf))],
                alpha=0.1, color=colors[i], label='_90% range')
plt.xlim(0,24)
plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
plt.xticks(xticks, (xticks-12)%24)
plt.grid('--', alpha=0.5)
axv = plt.axvspan(5,8, color='yellow', alpha=0.3, label='Evening window')
axv.set_hatch('//')
axv.set_edgecolor('olive')
plt.legend(loc=1)
plt.ylim(-220,220)

# Add arrows
p1 = 16.5
p2 = 19
idxp1 = int((p1-12)%24 * 60 / step)
idxp2 = int((p2-12)%24 * 60 / step)
yp1 = [fch['Company'][0].mean(axis=0)[idxp1], fdn['Company'][0][0].mean(axis=0)[idxp1]]
yp2 = [fch['Commuter_HP'][0].mean(axis=0)[idxp2], 0]

plt.annotate('', xy=((p1-12)%24, yp1[0]), xytext=((p1-12)%24, yp1[1]), arrowprops=dict(arrowstyle='<->'))
plt.text(x=(p1+0.5-12)%24,y=-175,s='V2G Flexibility of Company fleet')
plt.annotate('', xy=((p2-12)%24, yp2[0]), xytext=((p2-12)%24, yp2[1]), arrowprops=dict(arrowstyle='<->', color='k'), zorder=10)
plt.text(x=(p2-12-0.5)%24,y=-20,s='V1G Flexibility of Commuter fleet')
