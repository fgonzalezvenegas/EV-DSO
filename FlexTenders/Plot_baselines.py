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
import pandas as pd

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
                      ovn=True,
                      target_soc = 0.8) #because we're doing overnight

company_params = dict(arrival_departure_data={'wd' : dict(mu_arr=15, mu_dep=9,
                                                     std_arr=1, std_dep=1),
                                              'we' : dict(mu_arr=15, mu_dep=9,
                                                     std_arr=1, std_dep=1)},
                      dist_wd=dict(cdf=stats.norm.cdf(np.arange(1,100,2), 
                                                      loc=mu_dist_company, 
                                                      scale=sigma_dist_company)),
                      dist_we=dict(cdf=stats.norm.cdf(np.arange(1,100,2), 
                                                      loc=0.01, 
                                                      scale=0.0005)),
                      alpha=100  #100, always, 0, never,                      
                      )
# Use default values for weekends and for distances (lognormal dist, O.Borne)
commuter_params = dict(arrival_departure_data = {'wd' : dict(mu_arr=17.5, mu_dep=8,
                                                        std_arr=2, std_dep=2)})
n_proba_high=5
n_proba_low=1                

n_evs = 20 
                           
#%% 1 Compute EVs simulation: 
t.append(time.time())             
# SIMS PARAMS:
ndays = 7 * 10 + 1 # 10 weeks, + one extra day
step = 5 # minutes

# DO SIMULATIONS
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
grid.add_evs(nameset='Company', n_evs=n_evs, ev_type='dumb', 
             **general_params,
             **company_params)

grid.add_evs(nameset='Commuter_HP', n_evs=n_evs, ev_type='dumb', 
             **general_params,
             **commuter_params,
             alpha=n_proba_high)

grid.add_evs(nameset='Commuter_LP', n_evs=n_evs, ev_type='dumb', 
             **general_params,
             **commuter_params,
             alpha=n_proba_low)
for i, ev in enumerate(grid.evs_sets['Commuter_LP']):
    ev.dist_wd = grid.evs_sets['Commuter_HP'][i].dist_wd
    ev.dist_we = grid.evs_sets['Commuter_HP'][i].dist_we

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


#%% Base load params
t.append(time.time())
baseload = False
shift = 12
max_load = 5
if baseload:
    load = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\Conso\conso-inf36_profiles.csv',
                       engine='python', index_col=0)
    load = load['RES1 (+ RES1WE)'] / load['RES1 (+ RES1WE)'].max() * max_load
    load.index = pd.to_datetime(load.index)
    load = util.get_max_load_week(load)
    load = util.interpolate(load, step=step, method='polynomial', order=3)
    n = int(60/step)*24
    
    load = load[int(n*(3-shift/24)):int(n*(4-shift/24))]
else:
    load=0
t.append(time.time())                              
print('More params, t={:.2f} seg'.format(t[-1]-t[-2]))
#%% Evaluation params:

ch_company, dn_company = flex_payment_funcs.get_ev_profs(grid, nameset='Company')
ch_comm_h, dn_comm_h = flex_payment_funcs.get_ev_profs(grid, nameset='Commuter_HP')
ch_comm_l, dn_comm_l = flex_payment_funcs.get_ev_profs(grid, nameset='Commuter_LP')
(nevs, ndays, nsteps) = ch_comm_h.shape

# This is to include (or not) the baseload into the EV baseline computation
if baseload:
    loadevs = np.tile(load.values, (nevs, ndays,1)).sum(axis=0)
else:
    loadevs = 0
    
fleet_ch = np.array([ch_company.sum(axis=0) + loadevs, 
                     ch_comm_h.sum(axis=0) + loadevs, 
                     ch_comm_l.sum(axis=0) + loadevs])
fleet_dn = np.array([dn_company[0].sum(axis=0) + loadevs, 
                     dn_comm_h[0].sum(axis=0) + loadevs, 
                     dn_comm_l[0].sum(axis=0) + loadevs])

baseline_enedis = flex_payment_funcs.get_baselines(fleet_ch, bl='Enedis', ndays_bl=45, step=5)
baseline_UKPN_day = flex_payment_funcs.get_baselines(fleet_ch, bl='UKPN', ndays_bl=45, step=5)
baseline_UKPN_evening = np.zeros(baseline_enedis.shape)
baseline_UKPN_evening[:,:,idx_ini:idx_end] = flex_payment_funcs.get_baselines(fleet_ch[:, :, idx_ini:idx_end], bl='UKPN', ndays_bl=45, step=5)
#
#%% All baselines, gros bordel
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter MP']

x = np.arange(0,24,5/60)

d = int(np.random.randint(0, 45,1))
for i in range(3):
    plt.plot(x, baseline_enedis[i,0,:], color=c[i], linestyle='-', label=labels[i]+ ' panel baseline')
    plt.plot(x, baseline_UKPN_day[i,0,:], color=c[i], linestyle=':', label=labels[i] + ' unique baseline')
    plt.plot(x, fleet_ch[i, d, :], color=c[i], linestyle='-.', label=labels[i] + ' realisation')
plt.legend()

#%%  Average profiles for three fleets
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter MP']
ls = ['-',':','-.']

x = np.arange(0,24,5/60)
for i in range(3):
    plt.plot(x, fleet_ch[i].mean(axis=0), color=c[i], linestyle=ls[i], label=labels[i])
plt.plot(x, load*20, color='k', label='Baseline')
plt.legend()
plt.axis([0,24, 0, (7+5)*20])

plt.xlabel('Hours')
plt.ylabel('Power [kw]')
plt.xticks(np.arange(0,24,2), np.arange(12,36,2)%24)

#%%  Average profiles for three fleets + V2G potential
f, ax = plt.subplots()
c = ['b', 'r', 'g']
c2 = ['royalblue', 'orange', 'lightgreen']
labels = ['Company', 'Commuter HP', 'Commuter MP']
ls = ['-','--','-.']
m = ['o','.','*']

x = np.arange(0,24,5/60)
plt.axhline(y=0, xmin=0, xmax=100, color='k', linestyle='--', lw=0.5, alpha=1)
plt.axvspan(5,8, facecolor='yellow', label='Evening Window', alpha=0.5)

for i in range(3):
    plt.plot(x, fleet_ch[i].mean(axis=0), color=c[i], linestyle=ls[i], label=labels[i] + ' average profile')
for i in range(3):
    plt.plot(x, fleet_dn[i].mean(axis=0), color=c2[i], linestyle=':', marker=m[i], markersize=2, label=labels[i] + ' V2G potential')
plt.legend()

plt.axis([0,24, -150,150])

#
plt.xlabel('Time of day [h]')
plt.ylabel('Power [kW]')
plt.xticks(np.arange(0,25,2), np.arange(12,37,2)%24)

#%%
p1 = 17.5
p2 = 19
idxp1 = int((p1-12)%24 * 60 / step)
idxp2 = int((p2-12)%24 * 60 / step)
yp1 = [fleet_ch[0].mean(axis=0)[idxp1], fleet_dn[0].mean(axis=0)[idxp1]]
yp2 = [fleet_ch[2].mean(axis=0)[idxp2], 0]

plt.annotate('', xy=((p1-12)%24, yp1[0]), xytext=((p1-12)%24, yp1[1]), arrowprops=dict(arrowstyle='<->'))
plt.text(x=(p1+0.5-12)%24,y=-54,s='V2G Flexibility of Company fleet')
plt.annotate('', xy=((p2-12)%24, yp2[0]), xytext=((p2-12)%24, yp2[1]), arrowprops=dict(arrowstyle='<->', color='k'), zorder=10)
plt.text(x=(p2-12-0.5)%24,y=-15,s='V1G Flexibility of Commuter fleet')

#%% Baselines for company  (i=1); commuter HP (i=1); commuter LP (i=2)

c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter MP']
ls = ['--',':','-.']

i=0
x = np.arange(0,24,5/60)

f, ax = plt.subplots()
d=28.
#d = np.random.randint(fleet_ch.shape[1])
plt.axhline(y=0, xmin=0, xmax=100, color='k', linestyle='--', lw=0.5, alpha=1)
plt.axvspan(5,8, facecolor='yellow', label='Evening Window', alpha=0.5)
plt.plot(x, fleet_ch[i, d], color='k', linestyle='-', alpha=0.7, label='Realization')
plt.plot(x, baseline_enedis[i,0], color=c[0], linestyle=ls[0], label='30-min baseline')
plt.plot(x, baseline_UKPN_evening[i,0], color=c[1], linestyle=ls[1], label='Unique-value baseline')
#plt.plot(x, baseline_UKPN_evening[i,0], color=c[1], linestyle=ls[1], label='Unique-value baseline, Evening window')
#plt.plot(x, baseline_UKPN_day[i,0], color=c[2], linestyle=ls[2], label='Unique-value baseline, Full-day window')

plt.legend(framealpha=1)
plt.axis([0,24, 0, 7*10*2.5])
plt.xlabel('Hours')
plt.ylabel('Power [kW]')
plt.xticks(np.arange(0,25,2), np.arange(12,37,2)%24)
plt.xlim(0,12)
f.set_size_inches(3.5,3)
plt.tight_layout()
 plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\CIRED\baseline_thesis.pdf')
  plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\CIRED\baseline_thesis.jpg', dpi=300)
#%% Baselines for company  + down flex
f, ax = plt.subplots()
c = ['b', 'r', 'g']
labels = ['Company', 'Commuter HP', 'Commuter MP']
ls = ['--',':','-.']

i=0
x = np.arange(0,24,5/60)
d = np.random.randint(fleet_ch.shape[1])
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

#%% Average flex
flex_V2G = baseline_enedis - fleet_dn
flex_V1G = baseline_enedis - fleet_dn.clip(min=0)

mean_flex = np.array([flex_V2G.mean(axis=(1,2)), flex_V2G[:,:,idx_ini:idx_end].mean(axis=(1,2)),
             flex_V1G.mean(axis=(1,2)), flex_V1G[:,:,idx_ini:idx_end].mean(axis=(1,2))])/20



