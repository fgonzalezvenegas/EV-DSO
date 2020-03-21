# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:06:56 2020
Plot baselines

@author: U546416
"""



import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as stats
import time
import util
import flex_payment_funcs

#%% 0 Params
t = []
t.append(time.time())
# Case 1 : Company fleet IN-BUILDING
    # This means they will charge during the night
# FLEET PARAMS
batt_size = 40
ch_power = 7

# DSO SERVICE PARAMS:
service_time= 30        # minutes for which the service should be provided
t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))  

# Particular params
mu_dist_company = 40
sigma_dist_company = 5


general_params = dict(charging_power=ch_power,
                      batt_size=batt_size,
                      flex_time=service_time,
                      up_dn_flex=True,
                      ovn=True,
                      target_soc = 0.8) #because we're doing overnight

company_params = dict(arrival_departure_data_we=dict(mu_arr=15, mu_dep=15,
                                                     std_arr=0, std_dep=0),
                      arrival_departure_data_wd=dict(mu_arr=15, mu_dep=9,
                                                     std_arr=1, std_dep=1),
                      dist_wd=dict(cdf=stats.norm.cdf(np.arange(1,100,2), 
                                                      loc=mu_dist_company, 
                                                      scale=sigma_dist_company)),
                      dist_we=dict(cdf=stats.norm.cdf(np.arange(1,100,2), 
                                                      loc=0.01, 
                                                      scale=0.0005)),
                      n_if_needed=100  #100, always, 0, never,                      
                      )
# Use default values for weekends and for distances (lognormal dist, O.Borne)
commuter_params = dict(arrival_departure_data_wd=dict(mu_arr=17.5, mu_dep=8,
                                                     std_arr=2, std_dep=2))
n_proba_high=5
n_proba_low=1                

n_evs = 20 
                           
#%% 1 Compute EVs simulation: 
t.append(time.time())             
# SIMS PARAMS:
ndays = 7 * 10 + 1 # 50 weeks, + one extra day
step = 5 # minutes

# DO SIMULATIONS
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
grid.add_evs(nameset='Company', n_evs=n_evs, ev_type='dumb', 
             **general_params,
             **company_params)

grid.add_evs(nameset='Commuter_HP', n_evs=n_evs, ev_type='dumb', 
             **general_params,
             **commuter_params,
             n_if_needed=n_proba_high)

grid.add_evs(nameset='Commuter_LP', n_evs=n_evs, ev_type='dumb', 
             **general_params,
             **commuter_params,
             n_if_needed=n_proba_low)
grid.do_days()
t.append(time.time())                              
print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))

for key, val in grid.get_ev_data().items():
    print(str(key) + '= ' + str(val))
grid.plot_ev_load(days=7)
#%% Flex service params

av_w_ini = 17
av_w_end = 20
idx_ini = int((av_w_ini-12) * 60 / step)
idx_end = int((av_w_end-12) * 60 / step)

#%% Evaluation params:

ch_company, dn_company = flex_payment_funcs.get_ev_profs(grid, nameset='Company')
ch_comm_h, dn_comm_h = flex_payment_funcs.get_ev_profs(grid, nameset='Commuter_HP')
ch_comm_l, dn_comm_l = flex_payment_funcs.get_ev_profs(grid, nameset='Commuter_LP')

fleet_ch = np.array([ch_company.sum(axis=0), 
                     ch_comm_h.sum(axis=0), 
                     ch_comm_l.sum(axis=0)])
fleet_dn = np.array([dn_company.sum(axis=0), 
                     dn_comm_h.sum(axis=0), 
                     dn_comm_l.sum(axis=0)])

baseline_enedis = flex_payment_funcs.get_baselines(fleet_ch, bl='Enedis', ndays_bl=45, step=5)
baseline_UKPN_day = flex_payment_funcs.get_baselines(fleet_ch, bl='UKPN', ndays_bl=45, step=5)
baseline_UKPN_evening = np.zeros(baseline_enedis.shape)
baseline_UKPN_evening[:,:,idx_ini:idx_end] = flex_payment_funcs.get_baselines(fleet_ch[:, :, idx_ini:idx_end], bl='UKPN', ndays_bl=45, step=5)
#
#%% All baselines, gros bordel
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter LP']

x = np.arange(0,24,5/60)

d = int(np.random.randint(0, 45,1))
for i in range(3):
    plt.plot(x, baseline_enedis[i,0,:], color=c[i], linestyle='-', label=labels[i]+ ' panel baseline')
 #   plt.plot(x, baseline_UKPN_day[i,0,:], color=c[i], linestyle=':', label=labels[i] + ' unique baseline')
 #   plt.plot(x, fleet_ch[i, d, :], color=c[i], linestyle='-.', label=labels[i] + ' realisation')
plt.legend()

#%%  Average baselines for three fleets
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter LP']
ls = ['-',':','-.']

x = np.arange(0,24,5/60)
for i in range(3):
    plt.plot(x, fleet_ch[i].mean(axis=0), color=c[i], linestyle=ls[i], label=labels[i])
plt.legend()
plt.axis([0,24, 0, 7*20*1.2])

plt.xlabel('Hours')
plt.ylabel('Power [kw]')
plt.xticks(np.arange(0,24,2), np.arange(12,36,2)%24)

#%%  Average baselines for three fleets
f, ax = plt.subplots()
c = ['b', 'r', 'g']
c2 = ['royalblue', 'orange', 'lightgreen']
labels = ['Company', 'Commuter HP', 'Commuter LP']
ls = ['-','--','-.']
m = ['o','.','*']

x = np.arange(0,24,5/60)
for i in range(3):
    plt.plot(x, fleet_ch[i].mean(axis=0), color=c[i], linestyle=ls[i], label=labels[i] + ' average profile')
for i in range(3):
    plt.plot(x, fleet_dn[i].mean(axis=0), color=c2[i], linestyle=':', marker=m[i], markersize=2, label=labels[i] + ' down flexibility')
plt.legend()
plt.axis([0,24, -7*20*1.2, 7*20*1.2])

plt.xlabel('Hours')
plt.ylabel('Power [kw]')
plt.xticks(np.arange(0,24,2), np.arange(12,36,2)%24)

#%% Baselines for company 
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter LP']
ls = ['--',':','-.']

i=2
x = np.arange(0,24,5/60)
plt.plot(x, fleet_ch[i, d], color='k', linestyle='-', alpha=0.9, label='Realisation')
plt.plot(x, baseline_enedis[i,0], color=c[0], linestyle=ls[0], label='30 min baseline')
plt.plot(x, baseline_UKPN_evening[i,0], color=c[1], linestyle=ls[1], label='Evening window baseline')
plt.plot(x, baseline_UKPN_day[i,0], color=c[2], linestyle=ls[2], label='Full day window baseline')

plt.legend()
plt.axis([0,24, 0, 7*10*2])
plt.xlabel('Hours')
plt.ylabel('Power [kw]')
plt.xticks(np.arange(0,24,2), np.arange(12,36,2)%24)
plt.xlabel('Hours')
plt.ylabel('Power [kw]')
plt.xticks(np.arange(0,24,2), np.arange(12,36,2)%24)


#%% Baselines for company  + down flex
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter LP']
ls = ['--',':','-.']

i=0
x = np.arange(0,24,5/60)
plt.plot(x, fleet_ch[i, d], color='k', linestyle='-', alpha=0.9, label='Realisation')
plt.plot(x, baseline_enedis[i,0], color=c[0], linestyle=ls[0], label='30 min baseline')
plt.plot(x, baseline_UKPN_evening[i,0], color=c[1], linestyle=ls[1], label='Evening window baseline')
plt.plot(x, baseline_UKPN_day[i,0], color=c[2], linestyle=ls[2], label='Full day window baseline')
plt.plot(x, fleet_dn[i,d], color='grey', linestyle=':',marker='*', markersize=4, alpha=0.9, label='Feasible 30 minutes flexibility')

plt.legend()
plt.axis([0,24, -7*10*2*1.1, 7*10*2])
plt.xlabel('Hours')
plt.ylabel('Power [kw]')
plt.xticks(np.arange(0,24,2), np.arange(12,36,2)%24)
