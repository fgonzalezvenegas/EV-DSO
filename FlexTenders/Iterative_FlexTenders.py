# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:34:48 2020

@author: U546416
"""


import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as stats
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

n_evs = 1000
ovn=True

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
                      ovn=ovn,         #because we're doing overnight
                      target_soc = 0.8) 

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

t = []
t.append(time.time())



# DSO SERVICE PARAMS:
service_time= 30        # minutes for which the service should be provided
# I will select how many EVs do i need
min_bid = 50            # kW
av_window = [16, 20]    # Availability window
av_days = 'wd'          # Weekdays (wd), weekends (wd) only, all
t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))                             
#%% 1 Compute EVs simulation: 
t.append(time.time())             
# SIMS PARAMS:
ndays = 7 * 50 + 1 # 50 weeks, + one extra day
step = 5 # minutes

# DO SIMULATIONS
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
#grid.add_evs(nameset='Company', n_evs=n_evs, ev_type='dumb', 
#             **general_params,
#             **company_params)
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
#%% Evaluation params:
t.append(time.time())    
# Service params:
av_window = [0, 24] 
aw_s = str(av_window[0]) + '_' + str(av_window[1])
if ovn:
    shift = 12
else:
    shift = 0    
#av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
av_window_idxs = [0, 288]
av_window_vector = fpf.get_av_window_vector(av_window, step, ovn)

# Random params
nfleets = 1000 # Number of fleets to simulate
nscenarios = 1000
# Payment params:
# Pot size (in €/MWh)
# Equivalent payment in €/kW
eur_kw = 50
days_of_service = 20 * 3
nevents = 10

# Expected payment in €/MWh, considering the same value for availability & utilisation
exp_payment = eur_kw / (days_of_service * (av_window[1]-av_window[0])/1000 + nevents * (service_time/60) / 1000)

av_payment=exp_payment
ut_payment=exp_payment 


#
t.append(time.time())                              
print('More params, t={:.2f} seg'.format(t[-1]-t[-2]))
#%% Extract data
t.append(time.time())

nameset='Commuter_LP'
ch_profs, dn_profs = fpf.get_ev_profs(grid, ovn=ovn, av_days=av_days, baseline='i', nameset=nameset)
ch_profs_av_w = fpf.get_av_window(ch_profs, av_window_idxs)
dn_profs_av_w = fpf.get_av_window(dn_profs, av_window_idxs)
print('profiles')
(nevs, ndays, nsteps) = ch_profs.shape

#%% Iterate on number of evs
fleet_range = np.arange(3,100,2)
nrange = len(fleet_range)

stats_V1G_UKPN = []
stats_V2G_UKPN = []
stats_V1G_Enedis = []
stats_V2G_Enedis = []

cols = [j + '_' + k for j in ['Bids', 'Payments'] for k in ['Avg', 'min', 'max', 'perc_h', 'perc_l']]


tt = [time.time()]
for nevs_fleet in range(3,100,2):
    fleet_ch_profs, fleet_dn = fpf.get_fleet_profs(ch_profs_av_w, dn_profs_av_w, 
                                               nfleets=nfleets, nevs_fleet=nevs_fleet)
    print('\tFleet profiles')
    baseline_UKPN = fpf.get_baselines(fleet_ch_profs, bl='UKPN', ndays_bl=100)
    baseline_Enedis = fpf.get_baselines(fleet_ch_profs, bl='Enedis', ndays_bl=100)
    print('\tBaselines')
    flex_V1G = fpf.get_flex_wrt_bl(fleet_dn, baseline_UKPN, V2G=False)
    flex_V2G = fpf.get_flex_wrt_bl(fleet_dn, baseline_UKPN, V2G=True)
    flex_V1G_Enedis = fpf.get_flex_wrt_bl(fleet_dn, baseline_Enedis, V2G=False)
    flex_V2G_Enedis = fpf.get_flex_wrt_bl(fleet_dn, baseline_Enedis, V2G=True)
    print('\tFlexibility profs')
    V1G_bids, V1G_payments = fpf.compute_payments(flex_V1G, av_payment, ut_payment, 
                                             nevents, days_of_service, service_time=service_time, 
                                             min_delivery=0.6, min_bid=50, nscenarios=nscenarios)
    V2G_bids, V2G_payments = fpf.compute_payments(flex_V2G, av_payment, ut_payment, 
                                             nevents, days_of_service, 
                                             service_time=service_time, step=step,
                                             min_delivery=0.6, min_bid=50, nscenarios=nscenarios)
    V1G_bids_En, V1G_payments_En = fpf.compute_payments(flex_V1G_Enedis, av_payment, ut_payment, 
                                         nevents, days_of_service, service_time=service_time, 
                                         min_delivery=0.6, min_bid=50, nscenarios=nscenarios)
    V2G_bids_En, V2G_payments_En = fpf.compute_payments(flex_V2G_Enedis, av_payment, ut_payment, 
                                             nevents, days_of_service, 
                                             service_time=service_time, step=step,
                                             min_delivery=0.6, min_bid=50, nscenarios=nscenarios)
    print('\tPayments')
    statsb = fpf.get_stats(V1G_bids)
    statsp = fpf.get_stats(V1G_payments)
    stats_V1G_UKPN.append(statsb + statsp)
    statsb = fpf.get_stats(V2G_bids)
    statsp = fpf.get_stats(V2G_payments)
    stats_V2G_UKPN.append(statsb + statsp)
    statsb = fpf.get_stats(V1G_bids_En)
    statsp = fpf.get_stats(V1G_payments_En)
    stats_V1G_Enedis.append(statsb + statsp)
    statsb = fpf.get_stats(V2G_bids_En)
    statsp = fpf.get_stats(V2G_payments_En)
    stats_V2G_Enedis.append(statsb + statsp)
    print('\t stats')
    tt.append(time.time())
    print('Number of evs {}, t={:.2f} seg'.format(nevs_fleet, tt[-1]-tt[-2]))

stats_V1G_UKPN = pd.DataFrame(stats_V1G_UKPN, columns=cols, index=fleet_range)
stats_V2G_UKPN = pd.DataFrame(stats_V2G_UKPN, columns=cols, index=fleet_range)
stats_V1G_Enedis = pd.DataFrame(stats_V1G_Enedis, columns=cols, index=fleet_range)
stats_V2G_Enedis = pd.DataFrame(stats_V2G_Enedis, columns=cols, index=fleet_range)

t.append(time.time())
(h,m,s) = util.sec_to_time(t[-1]-t[-2])
print('Done sim, time {}h{}m{:.0f}s'.format(h, m, s))

#%% Save data
stats_V1G_UKPN.to_csv(r'Results\\' + nameset + '_' + aw_s + '_' + 'V1G_UKPN.csv')
stats_V2G_UKPN.to_csv(r'Results\\' + nameset + '_' + aw_s + '_' + 'V2G_UKPN.csv')
stats_V1G_Enedis.to_csv(r'Results\\' + nameset + '_' + aw_s + '_' + 'V1G_Enedis.csv') 
stats_V2G_Enedis.to_csv(r'Results\\' + nameset + '_' + aw_s + '_' + 'V2G_Enedis.csv') 

#%%
x= fleet_range
plt.subplots()
plt.plot(x, stats_V2G_UKPN.Bids_Avg, 'b-', label='V2G UKPN')
plt.plot(x, stats_V1G_UKPN.Bids_Avg, 'b--', label='V1G UKPN')
plt.plot(x, stats_V2G_Enedis.Bids_Avg, 'g-', label='V2G Enedis')
plt.plot(x, stats_V1G_Enedis.Bids_Avg, 'g--', label='V1G Enedis')
plt.legend()
plt.title('Fleet Bid')
plt.xlabel('Fleet size')
plt.ylabel('Bid [kW]')

#%%
plt.subplots()
plt.plot(x, stats_V2G_UKPN.Bids_Avg/x, 'b-', label='V2G UKPN')
plt.plot(x, stats_V1G_UKPN.Bids_Avg/x, 'b--', label='V1G UKPN')
plt.plot(x, stats_V2G_Enedis.Bids_Avg/x, 'g-', label='V2G Enedis')
plt.plot(x, stats_V1G_Enedis.Bids_Avg/x, 'g--', label='V1G Enedis')
plt.legend()
plt.title('Bid per EV')
plt.xlabel('Fleet size')
plt.ylabel('Bid [kW]')
#%%
plt.subplots()
plt.plot(x, stats_V2G_UKPN.Payments_Avg/x, 'b-', label='V2G UKPN')
plt.plot(x, stats_V1G_UKPN.Payments_Avg/x, 'b--', label='V1G UKPN')
plt.plot(x, stats_V2G_Enedis.Payments_Avg/x, 'g-', label='V2G Enedis')
plt.plot(x, stats_V1G_Enedis.Payments_Avg/x, 'g--', label='V1G Enedis')
plt.legend()
plt.title('Annual revenue per EV')
plt.xlabel('Fleet size')
plt.ylabel('Revenue [€]')

#%%

plt.subplots()
plt.errorbar(x, stats_V2G_UKPN.Payments_Avg/x, 
             yerr=[(stats_V2G_UKPN.Payments_Avg - stats_V2G_UKPN.Payments_perc_l)/x, 
                   (stats_V2G_UKPN.Payments_perc_h - stats_V2G_UKPN.Payments_Avg)/x],
                   elinewidth=0.8, capsize=1.5, ecolor='k', color='b',
                   label='V2G UKPN')
plt.errorbar(x, stats_V2G_Enedis.Payments_Avg/x, 
             yerr=[(stats_V2G_Enedis.Payments_Avg - stats_V2G_Enedis.Payments_perc_l)/x, 
                   (stats_V2G_Enedis.Payments_perc_h - stats_V2G_Enedis.Payments_Avg)/x], 
                   elinewidth=0.8, capsize=1.5, ecolor='k', color='g',
                   label='V2G Enedis')
plt.legend()
plt.title('Annual revenue per EV')
plt.xlabel('Fleet size')
plt.ylabel('Revenue [€]')
