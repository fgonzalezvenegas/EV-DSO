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

n_evs = 1000
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


##%% 1 Compute EVs simulation: 
#t.append(time.time())             
## SIMS PARAMS:
#nweeks = 50
#ndays = 7 * nweeks + 1 # 50 weeks, + one extra day
#step = 5 # minutes
#
## DO SIMULATIONS (takes about 1-2min)
#grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
#grid.add_evs(nameset='Company', n_evs=n_evs, ev_type='dumb', 
#             flex_time=service_time,
#             **general_params,
#             **company_params)
##grid.add_evs(nameset='Commuter_HP', n_evs=n_evs, ev_type='dumb', 
##             flex_time=service_time,
##             **general_params,
##             **commuter_params_HP)
##grid.add_evs(nameset='Commuter_LP', n_evs=n_evs, ev_type='dumb', 
##             flex_time=service_time,
##             **general_params,
##             **commuter_params_LP)
#grid.do_days()
#grid.plot_ev_load(day_ini=7, days=14)
#grid.plot_flex_pot(day_ini=7, days=14)
#t.append(time.time())                              
#print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))
#%% Evaluation params:
t.append(time.time())    
# Service params:

# Random params
nfleets = 1000 # Number of fleets to simulate
nscenarios = 500 # number of scenarios to simulate activations
# Payment params:
# Pot size (in €/MWh)
# Equivalent payment in €/kW
eur_kw = 50
days_of_service = 20 * 3
nevents = 10
#
t.append(time.time())                              
print('More params, t={:.2f} seg'.format(t[-1]-t[-2]))
#minbid = 50

#%% Extract data
av_w = [[17,20],[0,24]]
for s in ['Commuter_LP', 'Commuter_HP', 'Company']:
    print(s)
    t.append(time.time())             
    # SIMS PARAMS:
    nweeks = 50
    ndays = 7 * nweeks + 1 # 50 weeks, + one extra day
    step = 5 # minutes
    
    # DO SIMULATIONS (takes about 1-2min)
    grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)

    if s == 'Commuter_HP':
        grid.add_evs(nameset='Commuter_HP', n_evs=n_evs, ev_type='dumb', 
                     flex_time=service_time,
                     **general_params,
                     **commuter_params_HP)
    if s == 'Company':
        grid.add_evs(nameset='Company', n_evs=n_evs, ev_type='dumb', 
                 flex_time=service_time,
                 **general_params,
                 **company_params)
    if s == 'Commuter_LP':
        grid.add_evs(nameset='Commuter_LP', n_evs=n_evs, ev_type='dumb', 
                     flex_time=service_time,
                     **general_params,
                     **commuter_params_LP)
    grid.do_days()
#    grid.plot_ev_load(day_ini=7, days=14)
#    grid.plot_flex_pot(day_ini=7, days=14)
    t.append(time.time())                              
    print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))
    
    for a in av_w:
#        if (s == 'Commuter_LP') & (a == [17,20]):
#            continue
        print('Starting with availability window {}'.format(a))
        #av_window = [17, 20] 
        aw_s = str(a[0]) + '_' + str(a[1])
        shift = 12   
        av_window_idxs = [int(((a[0]-shift)%24)*60/step), int(((a[1]-shift)%24)*60/step)]
        if a == [0, 24]:
            av_window_idxs = [0, int(24*60/step)]
        t.append(time.time())
        
        # Extracting data
        nameset=grid.ev_sets[0]
        ch_profs, dn_profs = fpf.get_ev_profs(grid, ovn=ovn, av_days=av_days, nameset=nameset)
        (nevs, ndays, nsteps) = ch_profs.shape
        ch_profs_av_w = fpf.get_av_window(ch_profs, av_window_idxs)
        if type(dn_profs) == dict:
            dn_profs_av_w = dict()
            for i in dn_profs:
                dn_profs_av_w[i] = fpf.get_av_window(dn_profs[i], av_window_idxs)
        else:
            dn_profs_av_w = fpf.get_av_window(dn_profs, av_window_idxs)
        #baseloads = fpf.get_av_window(np.tile(load, (nfleets, ndays, 1)), av_window_idxs)
        
        t.append(time.time())
        print('EV profiles extracted, t={:.2f} seg'.format(t[-1]-t[-2]))
        
        ##%% Iterate on number of evs per fleet
        
        fleet_range = [10,500]
        nrange = 25
        x = np.logspace(np.log10(fleet_range[0]), np.log10(fleet_range[1]), num=nrange).round(0)
    #    x = [10,30,50,70]
        conf_threshold = [0.5,0.9,0.99]
        penalty_threshold = [0.6, 0.8]
        penalty_values = [0, eur_kw * 0.35, eur_kw]
        minbid = 10
        
        stats_VxG = {}
    
        t.append(time.time())
        cols = [j + '_' + k for j in ['Bids', 'Payments', 'UnderDel'] for k in ['Avg', 'min', 'max', 'perc_h', 'perc_l']]
        for nevs_fleet in x:
        #    if nevs_fleet <= 137:
        #        continue
            print('Computing fleet with {} EVs'.format(nevs_fleet))
            tt = [time.time()]    
            fleet_ch_profs, fleet_dn = fpf.get_fleet_profs(ch_profs_av_w, dn_profs_av_w, 
                                                       nfleets=nfleets, nevs_fleet=int(nevs_fleet))
            tt.append(time.time())
            print('\tFleet profiles, t={:.2f} seg'.format(tt[-1]-tt[-2]))
            baseline = fpf.get_baselines(fleet_ch_profs, bl='UKPN', ndays_bl=10)
        #    baseline_Enedis = fpf.get_baselines(fleet_ch_profs+baseloads, bl='Enedis', ndays_bl=100)
            tt.append(time.time())
            print('\tBaselines, t={:.2f} seg'.format(tt[-1]-tt[-2]))
            # Positive flex is power towards the system
            flex_V1G = fpf.get_flex_wrt_bl(fleet_dn, baseline, V2G=False)
            flex_V2G = fpf.get_flex_wrt_bl(fleet_dn, baseline, V2G=True)
        #    flex_V1G_Enedis = fpf.get_flex_wrt_bl(fleet_dn, baseline_Enedis, baseload, V2G=False)
        #    flex_V2G_Enedis = fpf.get_flex_wrt_bl(fleet_dn, baseline_Enedis, baseload, V2G=True)
            tt.append(time.time())
            print('\tFlexibility profs, t={:.2f} seg'.format(tt[-1]-tt[-2]))
            for i, f in enumerate(service_time):
                # Expected payment in €/MWh, considering the same value for availability & utilisation
                exp_payment = eur_kw / (days_of_service * (a[1]-a[0])/1000 + nevents * (f/60) / 1000)
                
                av_payment=exp_payment
                ut_payment=exp_payment 
                for j in conf_threshold:
                    for k in penalty_threshold:
                        params = dict(av_payment=av_payment, ut_payment=ut_payment,
                                      nevents=nevents, days_of_service=days_of_service,
                                      conf=j, service_time=j,
                                      min_delivery=k, min_bid=minbid, nscenarios=nscenarios)
                        V1G_bids, V1G_payments, V1G_und = fpf.compute_payments(flex_V1G[i], 
                                                                               **params)
                        V2G_bids, V2G_payments, V2G_und = fpf.compute_payments(flex_V2G[i], 
                                                                           **params)
                    
                        fpf_params = dict(percentile=95)
                        statsb = fpf.get_stats(V1G_bids, **fpf_params)
                        statsu = fpf.get_stats(V1G_und, **fpf_params)
                        for p in penalty_values:
                            V1G_payments = V1G_payments - V1G_bids.repeat(nscenarios) * V1G_und * p
                            statsp = fpf.get_stats(V1G_payments, **fpf_params)
                            stats_VxG['v1g', nevs_fleet, f, j, k, p] = (statsb + statsp + statsu)
                        statsb = fpf.get_stats(V2G_bids, **fpf_params)
                        statsp = fpf.get_stats(V2G_payments, **fpf_params)
                        statsu = fpf.get_stats(V2G_und, **fpf_params)
                        for p in penalty_values:
                            V2G_payments = V2G_payments - V2G_bids.repeat(nscenarios) * V2G_und * p
                            statsp = fpf.get_stats(V2G_payments, **fpf_params)
                            stats_VxG['v2g', nevs_fleet, f, j, k, p] = (statsb + statsp + statsu)
            del V1G_bids, V1G_payments, V1G_und, V2G_bids, V2G_payments, V2G_und, flex_V2G, flex_V1G, fleet_ch_profs, fleet_dn
            
            tt.append(time.time())
            print('\tStats, t={:.2f} seg'.format(tt[-1]-tt[-2]))
            
            print('Number of evs {}, t={:.2f} seg\n'.format(nevs_fleet, tt[-1]-tt[0]))
        
        stats_VxG = pd.DataFrame(stats_VxG, index=cols).T
        stats_VxG.index.names = ['VxG', 'nevs', 'service_duration', 'confidence', 'penalty_threshold', 'penalties']
        # transform data in per EV
        for i, j in stats_VxG.iterrows():
            stats_VxG.loc[i,:] = stats_VxG.loc[i,:] / i[1]
            
        
        t.append(time.time())
        (h,m,s) = util.sec_to_time(t[-1]-t[-2])
        print('Done sim, time {}h{}m{:.0f}s'.format(h, m, s))
        
        ##%% Save data
        folder = r'C:\Users\u546416\AnacondaProjects\EV-DSO\FlexTenders\EnergyPolicy\\'
        
        filehead = 'full_' + nameset + '_' + aw_s + '_'
        stats_VxG.to_csv(folder + filehead + 'VxG.csv')
        print('saving')
    
#%% Plot avg profiles and baselines
f, ax = plt.subplots()
for i in range(ndays):
    ax.plot(fleet_ch_profs[0,i]/x[-1], alpha=0.3)
    ax.plot(baseline[0,0]/x[-1], label='Baseline')
plt.title('Possible realizations and Baseline')
f, axs = plt.subplots(2)
axs[0].plot(baseline[0,0]/x[-1], label='Baseline')
axs[0].plot(fleet_ch_profs[0,0]/x[-1], label='Realization')
for i in range(ndays):
    axs[1].plot(flex_V2G[0][0,i]/x[-1], label='_', alpha=0.3)
nf, nd, ns = flex_V2G[0].shape
for j in conf_threshold:
    nconf = int(nd*ns*(1-j))
    bid = np.sort(flex_V2G[0].reshape(nf, ns*nd), axis=1)[0, nconf]/x[-1]
    axs[1].plot([0, ns], np.ones(2) * bid, '--', label='Bid at {} confidence'.format(j))
axs[0].legend()
axs[1].legend()
f.suptitle('V2G')

f, axs = plt.subplots(2)
axs[0].plot(baseline[0,0]/x[-1], label='Baseline')
axs[0].plot(fleet_ch_profs[0,0]/x[-1], label='Realization')
for i in range(ndays):
    axs[1].plot(flex_V1G[0][0,i]/x[-1], label='_', alpha=0.3)
nf, nd, ns = flex_V1G[0].shape
for j in conf_threshold:
    nconf = int(nd*ns*(1-j))
    bid = np.sort(flex_V1G[0].reshape(nf, ns*nd), axis=1)[0, nconf]/x[-1]
    axs[1].plot([0, ns], np.ones(2) * bid, '--', label='Bid at {} confidence'.format(j))
axs[1].legend(loc=2)
axs[0].legend(loc=2)
f.suptitle('V1G')

#%% Plot Average Payments per EV, Error bars
#idx = pd.IndexSlice
#
#
#for f in service_time:
#    plt.subplots()
#    for j in conf_threshold:
#        v2g = stats_V2G.loc[idx[:,f,j,0.6,0], idx[:]]
#        plt.errorbar(x, v2g.Payments_Avg, 
#                     yerr=[(v2g.Payments_Avg - v2g.Payments_perc_l), 
#                           (v2g.Payments_perc_h - v2g.Payments_Avg)],
#                           elinewidth=0.8, capsize=1.5, ecolor='k', color='b',
#                           label='V2G at {} confidence'.format(j))
#    plt.legend()
#    plt.title('Annual revenue per EV,\n{} min service time'.format(f))
#    plt.xlabel('Fleet size')
#    plt.ylabel('Revenue [€/EV.y]')
##plt.axis((0,100, 0,500))

#%% Plot average payments per EV, with shaded area for confidence
idx = pd.IndexSlice
max_nevs = stats_V2G.index.max()[0]

NUM_COLORS = 6
cm = plt.get_cmap('Paired')
colors = [cm(1.*(i*2+1)/(NUM_COLORS*2)) for i in range(NUM_COLORS)]

folder_figs = r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\\'
    
for f in service_time:
    plt.subplots()
    for i, j in enumerate(conf_threshold):
        v2g = stats_V2G.loc[idx[:,f,j,0.6,0], idx[:]]
        plt.plot(x, v2g.Payments_Avg, linewidth=1.5, color=colors[i],
                           label='V2G at {} confidence'.format(j))
        plt.fill_between(x, v2g.Payments_perc_l, v2g.Payments_perc_h, 
                         alpha=0.2, color=colors[i], label='_90% range')
        v1g = stats_V1G.loc[idx[:,f,j,0.6,0], idx[:]]
        plt.plot(x, v1g.Payments_Avg, linewidth=1.5, color=colors[i], linestyle='--',
                           label='V1G at {} confidence'.format(j))
        plt.fill_between(x, v1g.Payments_perc_l, v1g.Payments_perc_h, 
                         alpha=0.1, color=colors[i], label='_90% range')
    plt.legend()
    plt.title('Annual revenue per EV,\n{}min service time'.format(f))
    plt.xlabel('Fleet size')
    plt.ylabel('Revenue [€/EV.y]')
    plt.xlim(0,max_nevs)
    plt.ylim(0,np.round(stats_V2G.Payments_perc_h.max(),-1)+10)
    plt.grid(linestyle='--', alpha=0.8)
    plt.savefig(folder_figs + '{}_Rev_{}m_{}.png'.format(nameset,j,aw_s))

#%% Plot average payments per EV, with shaded area for confidence - comparison of penalties
idx = pd.IndexSlice

NUM_COLORS = 6
cm = plt.get_cmap('Paired')
colors = [cm(1.*(i*2+1)/(NUM_COLORS*2)) for i in range(NUM_COLORS)]

folder_figs = r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\\'
    
for f in service_time:
    plt.subplots()
    for i, j in enumerate(conf_threshold):
        v2g = stats_V2G.loc[idx[:,f,j,0.6,0], idx[:]]
        plt.plot(x, v2g.Payments_Avg, linewidth=1.5, color=colors[i],
                           label='V2G at {} confidence, no penalties'.format(j))
        plt.fill_between(x, v2g.Payments_perc_l, v2g.Payments_perc_h, 
                         alpha=0.1, color=colors[i], label='_90% range')
        p = 17.5
        v2g = stats_V2G.loc[idx[:,f,j,0.8,p], idx[:]]
        plt.plot(x, v2g.Payments_Avg, linewidth=1.5, color=colors[i], linestyle='--',
                           label='V2G at {} confidence, penalties @{}%'.format(j, int(p*2)))
        plt.fill_between(x, v2g.Payments_perc_l, v2g.Payments_perc_h, 
                         alpha=0.1, color=colors[i], label='_90% range')
    plt.legend()
    plt.title('Annual revenue per EV,\n{}min service time'.format(f))
    plt.xlabel('Fleet size')
    plt.ylabel('Revenue [€/EV.y]')
    plt.xlim(0,max_nevs)
    plt.ylim(min(0, np.round(stats_V2G.Payments_perc_l.min(),-1)-10),np.round(stats_V2G.Payments_perc_h.max(),-1)+10)
    plt.grid(linestyle='--', alpha=0.8)
    plt.savefig(folder_figs + '{}_Rev_p_{}m_{}.png'.format(nameset,j, aw_s))


#%% Plot Avg Bids per EV
for f in service_time:
    plt.subplots()
    for i, j in enumerate(conf_threshold):
        v2g = stats_V2G.loc[idx[:,f,j,0.6,0], idx[:]]
        plt.plot(x, v2g.Bids_Avg, linewidth=1.5, color=colors[i],
                           label='V2G bid {} confidence'.format(j))
        plt.fill_between(x, v2g.Bids_perc_l, v2g.Bids_perc_h, 
                         alpha=0.1, color=colors[i], label='_90% range')
#        plt.errorbar(x, v2g.Bids_Avg, 
#                     yerr=[(v2g.Bids_Avg - v2g.Bids_perc_l), 
#                           (v2g.Bids_perc_h - v2g.Bids_Avg)],
#                           elinewidth=0.8, capsize=1.5, color=colors[i],
#                           label='{} confidence'.format(j))
    plt.plot(x, v1g.Bids_Avg, linewidth=1.5, color=colors[i+1],
                           label='V1G bid'.format(j))
    plt.fill_between(x, v1g.Bids_perc_l, v1g.Bids_perc_h, 
                         alpha=0.1, color=colors[i+1], label='_90% range')
    plt.legend()
    plt.title('Bid per EV,\n{}min service time'.format(f))
    plt.xlabel('Fleet size')
    plt.ylabel('Bid [kW]')
    plt.xlim(0,max_nevs)
    plt.ylim(0,np.ceil(stats_V2G.Bids_perc_h.max()))
    plt.grid(linestyle='--', alpha=0.8)
    plt.savefig(folder_figs + '{}_Bid_{}m_{}.png'.format(nameset,j,aw_s))
    
#for f in service_time:
#    plt.subplots()
#    for i, j in enumerate(conf_threshold):
#        v1g = stats_V1G.loc[idx[:,f,j], idx[:]]
#        plt.errorbar(x, v1g.Bids_Avg, 
#                     yerr=[(v1g.Bids_Avg - v1g.Bids_perc_l), 
#                           (v1g.Bids_perc_h - v1g.Bids_Avg)],
#                           elinewidth=0.8, capsize=1.5, color=colors[i],
#                           label='{} confidence'.format(j))
#    plt.legend()
#    plt.title('Bid per EV with V1G,\n{}min service time'.format(f))
#    plt.xlabel('Fleet size')
#    plt.ylabel('Bid [kW]')
#    plt.ylim(0,np.round(stats_V1G.Bids_perc_h.max(),0)+1)
#    plt.grid(linestyle='--', alpha=0.8)
#        #plt.axis((0,100, 0,10))


#%% Plot Avg Under-delivery

for f in service_time:
    plt.subplots()
    for i, j in enumerate(conf_threshold):
        v2g = stats_V2G.loc[idx[:,f,j,0.6,0], idx[:]]
        plt.plot(x, v2g.UnderDel_Avg * 100, linewidth=1.5, color=colors[i],
                           label='{} confidence, u.d. threshold 60%'.format(j))
        plt.fill_between(x, v2g.UnderDel_perc_l  * 100, v2g.UnderDel_perc_h * 100, 
                         alpha=0.1, color=colors[i], label='_90% range')
        
        v2g = stats_V2G.loc[idx[:,f,j,0.8,17.5], idx[:]]
        plt.plot(x, v2g.UnderDel_Avg * 100, '--', linewidth=1.5, color=colors[i],
                           label='{} confidence, u.d. threshold 80%'.format(j))
        plt.fill_between(x, v2g.UnderDel_perc_l  * 100, v2g.UnderDel_perc_h * 100, 
                         alpha=0.1, color=colors[i], label='_90% range')
#        plt.errorbar(x, v2g.UnderDel_Avg * 100, 
#                     yerr=[(v2g.UnderDel_Avg - v2g.UnderDel_perc_l) * 100, 
#                           (v2g.UnderDel_perc_h - v2g.UnderDel_Avg) * 100],
#                           elinewidth=0.8, capsize=1.5, ecolor='k', color='b',
#                           label='{} confidence'.format(j))
#                
#    plt.plot(x, v1g.UnderDel_Avg * 100, linewidth=1.5, color=colors[i+1],
#                           label='V1G'.format(j))
    plt.fill_between(x, v1g.UnderDel_perc_l * 100, v1g.UnderDel_perc_h * 100, 
                     alpha=0.1, color=colors[i+1], label='_90% range')
    plt.legend()
    plt.title('Under-delivery,\n{}min service time'.format(f))
    plt.xlabel('Fleet size')
    plt.ylabel('Under-delivery [%]')
    plt.xlim(0,max_nevs)
#    plt.ylim
    plt.grid(linestyle='--', alpha=0.8)
    plt.savefig(folder_figs + '{}_UD_{}_{}m.png'.format(nameset,j,aw_s))
        
#for f in service_time:
#    plt.subplots()
#    for i, j in enumerate(conf_threshold):
#        v1g = stats_V1G.loc[idx[:,f,j], idx[:]]
#        plt.errorbar(x, v1g.UnderDel_Avg * 100, 
#                     yerr=[(v1g.UnderDel_Avg - v1g.UnderDel_perc_l) * 100, 
#                           (v1g.UnderDel_perc_h - v1g.UnderDel_Avg) * 100],
#                           elinewidth=0.8, capsize=1.5, ecolor='k', color='b',
#                           label='{} confidence'.format(j))
#                
#    plt.legend()
#    plt.title('Under-delivery (<60%) with V1G,\n{}min service time'.format(f))
#    plt.xlabel('Fleet size')
#    plt.ylabel('Under-delivery [%]')


#%% Load data

folder = r'EnergyPolicy//'
av_w  = '0_24'
nameset = 'Commuter_LP'
filehead = nameset + '_' + aw_s + '_'
stats_V1G = pd.read_csv(folder + filehead + 'V1G.csv', index_col=[0,1,2,3,4])
stats_V2G = pd.read_csv(folder + filehead + 'V2G.csv', index_col=[0,1,2,3,4])


#%% 
sets = ['Commuter_HP', 'Commuter_LP', 'Company']
folder = r'EnergyPolicy//'
av_w  = ['0_24', '17_20']
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
res = {}
res2 = {}

for s in sets:
    for a in av_w:
        filehead = s + '_' + a + '_'
        stats = pd.read_csv(folder + filehead + 'V1G.csv', index_col=[0,1,2,3,4])
        res[s, a] = stats[bds].loc[30,30,0.5,0.6,0]
        res2[s, a] = stats[bds].loc[70,30,0.5,0.6,0]
        print(s,a, stats[bds].loc[30,30,0.5,0.6,0].values)
        