# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 03:07:29 2019

@author: U546416
"""


import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as stats
import time.time as time

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

#%% Useful functions
    
def split_by_days(data, step=15):
    """ Returns an array splited by days
    Data: 2-dim array, with shape [nEVs, data(ev)]
    Step: Length of step size of data (minutes, int)
    Shift: Hours to be shifted (hours<24, int)
    """
    steps_day = int(60/step * 24)
    ndim = np.ndim(data)
#    if ndim == 1:
#        ndays = int(len(data) / steps_day)
#        return np.reshape(data, [ndays, steps_day])
#    else:
    nevs, lendata = np.shape(data)
    ndays = int(lendata / steps_day)
    return np.reshape(data, [nevs, ndays, steps_day])
    
def drop_days(data, days_before=7, days_after=1, step=15, drop_we=False, shift=0):
    """ Returns the data array without buffer days (before and after)
    """
    
        
    idx_tini = ((days_before * 24) + shift) * int(60/step)
    
    idx_tend = ((days_after * 24) + (24 - shift))* int(60/step)
#    if np.ndim(data) == 1:
#        lendata = len(data)
#        drops = data[idx_tini:lendata-idx_tend]
#    else:
    nevs, lendata = np.shape(data)
    drops =  data[:,idx_tini:lendata-idx_tend]
    
    if drop_we:
        drops = drop_weekends(drops, step)
    
    return drops
    
def drop_weekends(data, step=15):
    """ Returns the data array without weekends.
    It assumes data starts on a monday
    """
    steps_day = int(60/step * 24)
    ndim = np.ndim(data)
#    if ndim == 1:
#        ndays = int(len(data) / steps_day)
#        nweeks = int(ndays / 7)
#    
#        weekdays = np.concatenate([data[i*7*steps_day:(i*7+5)*steps_day] for i in range(nweeks)])
#    else:
    nevs, lendata = np.shape(data)
    ndays = int(lendata / steps_day)
    nweeks = int(ndays / 7)
    weekdays = np.array([np.concatenate([data[ev, i*7*steps_day:(i*7+5)*steps_day] for i in range(nweeks)]) for ev in range(nevs)])
    return weekdays
    
#%% Schéma
""" 
1 - def params:
    To simulate availability/charging patterns
    - Pborne
    - Taille Bat
    - V1G/V2G
    - Proba plug-in
    - Type de flotte:
        Particulier:
            Low/medium plug-in proba
            variable km/day
        Company:
            High plug in proba = always
            Higher km/day
    - SOC target (maybe for later)

2 - Def profiles:
    Do sims for 2 weeks, N cars
    Obtain availability and charging profiles (save?, depends on run time)
    
3 - Compute Up dn flex
    - Depends on V1G/V2G
    
4 - Compute Baseline

5 - Compute KPI:
    Min # of EVs for min bid
    Revenue per ev
    
    
"""   

    
#%% 1- PARAMS!
"""
# Cases: 
Company fleet - only overnight
Company building - only daytime; like JPL
Private fleet, low plug in - overnight
Private fleet, high plug in - overnight

Company site: Overnight company fleet + day workers fleet

Fleet params:
    Batt_size
    Pborne
    Plug-in proba, from 0 (always no) to 100 (always yes) 
    Daily kms
    Arrival departure
    SOC target = 0.8 (to see effect afterwards)
    
Company fleet
- Batt_size = 40 kwh
- Pborne = 7kW (sensi a 11kw)
- Proba plug in = 100
- Daily kms: 
     Data from O. Borne thesis
         Trip distance (one way):
             Normal distr, mean=40, std_dev = 5
     Andersen 2019 (parker):
         Lognormal fit with mean 6kWh/day (+-35km/day); mean=1.44; stddev=0.79
         
     eCube pour PSA - 25000km/year => +-80km/day considering 6/7 days a week

- Arrival departure:
    Data from O. Borne thesis
         Departure:
             Normal distr, mean=8, std_dev = 1
         Arrival: 
             Normal distr for departure from work, mean=14, std_dev = 1,
                 Actual arrival to home depending on avg speed, mean 15km/h, std_dev = 5
     Andersen 2019 (parker):
         Departure
             Shifted lognormal (+ 5.5), mean=1.04; stddev=0.53

Building fleet,
- Batt size = 40 kWh (what if diffs EVs?)
- Pborne = 7kW
- Proba plugin = high? But saying at most there are x spots?
- Arrival - staying time according to JPL


         
Private fleet, low plug in
- Batt_size = 40 kwh
- Pborne = 3.6kW (sensi a 11kw)
- Proba plug in = 50
- Daily kms: 
    Lognormal France, +-19km mean
    Sensi on 
- Arrival departure:
    Data from O. Borne thesis
         Departure:
             Normal distr, mean=8, std_dev = 1
         Arrival: 
             Normal distr for departure from work, mean=14, std_dev = 1,
                 Actual arrival to home depending on avg speed, mean 15km/h, std_dev = 5
     Andersen 2019 (parker):
         Departure
             Shifted lognormal (+ 5.5), mean=1.04; stddev=0.53
Private fleet, high plug in

     

 
 Company = 100; Private low = 50; Private high = 90
 Daily kms:


"""
#%% 0 Params
# Case 1 : Company fleet IN-BUILDING
    # This means they will charge during the night
# FLEET PARAMS
nameset = 'company'
batt_size = 40
ch_power = 7
n_plugin_proba = 100  #100, always, 0, never

mu_dist = 40
sigma_dist=5

distance = {'cdf': stats.norm.cdf(np.arange(1,100,2), loc=mu_dist, scale=sigma_dist)}

arrival_departure_data_wd = {'mu_arr':15, 'mu_dep':9,
                             'std_arr':1, 'std_dep':1}
arrival_departure_data_we = {'mu_arr':15, 'mu_dep':15,
                             'std_arr':0, 'std_dep':0}
ovn = True
soc_target = 0.8

n_evs = 1000 


# DSO SERVICE PARAMS:
service_time= 30        # minutes for which the service should be provided
# I will select how many EVs do i need
min_bid = 50            # kW
av_window = [16, 20]    # Availability window
av_days = 'wd'          # Weekdays (wd), weekends (wd) only, all                            
                             
#%% 1 Compute EVs simulation:            
# SIMS PARAMS:
ndays = 7 * 50 + 1 # 50 weeks, + one extra day
step = 5 # minutes

# DO SIMULATIONS
grid = EVmodel.Grid(ndays=ndays, step=step)
grid.add_evs(nameset=nameset, n_evs=n_evs, ev_type='dumb', 
             arrival_departure_data_we=arrival_departure_data_we,
             arrival_departure_data_wd=arrival_departure_data_wd,
             dist_wd=distance,
             dist_we=0,
             target_soc=0.8,
             ovn=ovn,
             charging_power=ch_power,
             batt_size=batt_size,
             n_if_needed=n_plugin_proba,
             up_dn_flex=True,
             flex_time=service_time
             )
grid.do_days()                             
    
                         
#%% 2 Select a subset of m evs: 
evs = grid.get_evs()
av_profs = np.array([ev.off_peak_potential for ev in evs])
ch_profs = np.array([ev.charging for ev in evs])

# Possible kWs that could be proposed to DSO, for a flex service (not yet taking into account baselines)

#up_profs_meantraj  = np.array([ev.up_flex_kw_meantraj for ev in evs])     
#up_profs_immediate = np.array([ev.up_flex_kw_immediate for ev in evs])     
#up_profs_delayed   = np.array([ev.up_flex_kw_delayed for ev in evs])     
#dn_profs_meantraj  = np.array([ev.dn_flex_kw_meantraj for ev in evs])     
dn_profs_immediate = np.array([ev.dn_flex_kw_immediate for ev in evs])     
#dn_profs_delayed   = np.array([ev.dn_flex_kw_delayed for ev in evs])     

                         
#%% 3-new profs at nev = x
                             
#split by days
if ovn:
    shift = 12
    days_after=0
else:
    shift = 0
    days_after=1
if av_days in ['wd', 'weekdays']:
    drop_we = True
else:
    drop_we = False

# Drop buffer days and weekends (if needed)    
# Possible kWs profiles that could be proposed to DSO, split by days. An array of shape (nevs, ndays, nsteps (per day)) 
    
#up_profs_meantraj  = split_by_days(drop_days(up_profs_meantraj, days_before=7, 
#                                             days_after=days_after, step=step, 
#                                             drop_we=drop_we, shift=shift), step=step)
#up_profs_immediate = split_by_days(drop_days(up_profs_immediate, days_before=7, 
#                                             days_after=days_after, step=step, 
#                                             drop_we=drop_we, shift=shift), step=step)
#up_profs_delayed   = split_by_days(drop_days(up_profs_delayed, days_before=7,                                              
#                                             days_after=days_after, step=step, 
#                                             drop_we=drop_we, shift=shift), step=step)  
#dn_profs_meantraj  = split_by_days(drop_days(dn_profs_meantraj, days_before=7,
#                                             days_after=days_after, step=step, 
#                                             drop_we=drop_we, shift=shift), step=step)
dn_profs_immediate = split_by_days(drop_days(dn_profs_immediate, days_before=7, 
                                             days_after=days_after, step=step, 
                                             drop_we=drop_we, shift=shift), step=step)
#dn_profs_delayed   = split_by_days(drop_days(dn_profs_delayed, days_before=7, 
#                                             days_after=days_after, step=step, 
#                                             drop_we=drop_we, shift=shift), step=step)

ch_profs = split_by_days(drop_days(ch_profs, days_before=7,
                                   days_after=days_after, step=step, 
                                   drop_we=drop_we, shift=shift), step=step)


n_evs, ndays, nsteps = dn_profs_immediate.shape

# Compute aggregated profiles for EV fleets

nev_fleet = 10 # number of EVs in the fleet
nfleets = 1000 # Number of fleets to simulate
nint = [np.random.randint(0,n_evs,nev_fleet) for i in range(nfleets)] # Fleets, based on combinations of EV indexes 

# up & dn profiles
#fleet_up_meantraj = np.array([up_profs_meantraj[n].sum(axis=0) for n in nint])
#fleet_dn_meantraj = np.array([dn_profs_meantraj[n].sum(axis=0) for n in nint])
#fleet_up_immediate = np.array([up_profs_immediate[n].sum(axis=0) for n in nint])
fleet_dn_immediate = np.array([dn_profs_immediate[n].sum(axis=0) for n in nint])
# ... etc do for the other ones

# Charging profiles for fleet
fleet_ch_profs = np.array([ch_profs[n].sum(axis=0) for n in nint])

# Compute at x% confidence of aggregate fleet            
confidence_level = 0.95
idx_conf = int(confidence_level * ndays)
                                    
#fleet_up_meantraj_atconf = np.sort(fleet_up_meantraj, axis=1)[:, ndays-idx_conf, :]                             
#fleet_dn_meantraj_atconf = np.sort(fleet_dn_meantraj, axis=1)[:, idx_conf, :]   
#fleet_up_immediate_atconf = np.sort(fleet_up_immediate, axis=1)[:, ndays-idx_conf, :]                             
fleet_dn_immediate_atconf = np.sort(fleet_dn_immediate, axis=1)[:, idx_conf, :]                             
# ... etc do for the other ones                     
                             


#%% Do some plots

# Plot of x% Confidence for all fleets

x = [i*step/60 + shift for i in range(int(24 * 60 / step))]
plt.subplots()
for i in range(nfleets):
    plt.plot(x, fleet_up_meantraj_atconf[i], alpha=0.2)                             
    plt.plot(x, fleet_dn_meantraj_atconf[i], alpha=0.2)  
plt.xticks([i+shift for i in range(25)], [str((i+shift)%24) for i in range(25)])
plt.title('Up and down possible {} min flexibility for fleet, with {:1.2f} confidence\nNumber of EVs per fleet {}'.format(
        service_time, confidence_level, nev_fleet))

#%% Plot of Avg x% Confidence level for simulated fleets, vs 100% confidence
x = [i*step/60 + shift for i in range(int(24 * 60 / step))]
plt.subplots()
plt.plot(x, fleet_up_meantraj_atconf.mean(axis=0), alpha=1, label='Up {:1.2f} confidence'.format(confidence_level))                             
plt.plot(x, np.mean(np.sort(fleet_up_meantraj, axis=1)[:, 0, :], axis=0), alpha=1, label='Up 1.00 confidence')
plt.plot(x, np.mean(np.sort(fleet_up_meantraj, axis=1)[:, int(ndays/2), :], axis=0), alpha=1, label='Up 0.50 confidence')  
plt.plot(x, fleet_dn_meantraj_atconf.mean(axis=0), alpha=1, label='Down {:1.2f} confidence'.format(confidence_level))                             
plt.plot(x, np.mean(np.sort(fleet_dn_meantraj, axis=1)[:, -1, :], axis=0), alpha=1, label='Down 1.00 confidence') 
plt.plot(x, np.mean(np.sort(fleet_dn_meantraj, axis=1)[:, int(ndays/2), :], axis=0), alpha=1, label='Down 0.50 confidence') 
plt.xticks([i+shift for i in range(25)], [str((i+shift)%24) for i in range(25)])
plt.legend()
plt.title('Up and down possible {} min flexibility, at different confidence levels.'.format(service_time) +
          '\n {} fleets simulated, {} EVs per fleet'.format(nfleets, nev_fleet))
plt.xlabel('Hours')
plt.ylabel('Fleet output [kW]')


#%% 4- Compute Baselines

# UKPN baseline: n representative days, plus a uniform BL during the availability window
ndays_bl = 10
d = np.random.randint(0,ndays, ndays_bl)
av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
UKPN_bl   =  fleet_ch_profs[:, d, av_window_idxs[0]:av_window_idxs[1]].mean(axis=(1,2))

av_window_vector = np.concatenate((np.zeros(av_window_idxs[0]), 
                                   np.ones(av_window_idxs[1]-av_window_idxs[0]),
                                   np.zeros(nsteps - av_window_idxs[1])))

# As a matrix of dim (nfleets, ndays, nsteps)
av_window_matrix = np.tile(av_window_vector, (nfleets, ndays, 1))
# As a matrix of dim (nfleets, ndays, nsteps)
ukpn_bls = (av_window_matrix * np.tile(UKPN_bl, (nsteps, ndays, 1)).T)

# Enedis bl: Panel of similar users. 
# Lets say we take the average for the fleet (as it is supposed to be representative)
Enedis_bl =  fleet_ch_profs.mean(axis=1)
enedis_bls = np.zeros((nfleets, ndays, nsteps))
for f in range(nfleets):
    enedis_bls[f] = np.tile(Enedis_bl[f], (ndays, 1))
enedis_bls = enedis_bls * av_window_matrix

# Flex wrt baselines

V1G_flex_UKPN = ukpn_bls - av_window_matrix * fleet_dn_immediate.clip(min=0)
V2G_flex_UKPN = ukpn_bls - av_window_matrix * fleet_dn_immediate
V1G_flex_Enedis = enedis_bls - av_window_matrix * fleet_dn_immediate.clip(min=0)
V2G_flex_Enedis = enedis_bls - av_window_matrix * fleet_dn_immediate

#%% Plot Enedis & UKPN baselines
plt.subplots()
for i in range(10):
    plt.plot(ukpn_bls[i,0,:], 'r')
    plt.plot(enedis_bls[i,0,:], 'b')
    plt.plot(-V2G_flex_UKPN[i].mean(axis=0), 'r--')
    plt.plot(-V2G_flex_Enedis[i].mean(axis=0), 'b--')

#%% 5 - Compute money money

av_payment = 200 # €/MW.h
ut_payment = 200 # €/MWh

nevents = 10

event_duration = 1 #h

days_of_service = 50

min_bid = 50

# define kws. Option 1: mean value of expected flex kWs (per fleet?)
V1G_bid_UKPN = V1G_flex_UKPN[:,:, av_window_idxs[0]:av_window_idxs[1]].mean(axis=(1,2))
V2G_bid_UKPN = V2G_flex_UKPN[:,:, av_window_idxs[0]:av_window_idxs[1]].mean(axis=(1,2))
V1G_bid_Enedis = V1G_flex_Enedis[:,:, av_window_idxs[0]:av_window_idxs[1]].mean(axis=(1,2))
V2G_bid_Enedis = V2G_flex_Enedis[:,:, av_window_idxs[0]:av_window_idxs[1]].mean(axis=(1,2))

# Cut bids under minimum bid
V1G_bid_UKPN = V1G_bid_UKPN * (V1G_bid_UKPN > min_bid)
V2G_bid_UKPN = V2G_bid_UKPN * (V2G_bid_UKPN > min_bid)
V1G_bid_Enedis = V1G_bid_Enedis * (V1G_bid_Enedis > min_bid)
V2G_bid_Enedis = V2G_bid_Enedis * (V2G_bid_Enedis > min_bid)


# evaluate delivery
len_service = int(service_time/step)
nscenarios = 1000
d = np.random.randint(0, ndays, (nscenarios, nevents))
t = np.random.randint(av_window_idxs[0], av_window_idxs[1] - len_service, (nscenarios, nevents))


V1G_delivery_UKPN  = np.zeros((nfleets, nscenarios, nevents))
V2G_delivery_UKPN  = np.zeros((nfleets, nscenarios, nevents))
V1G_delivery_Enedis  = np.zeros((nfleets, nscenarios, nevents))
V2G_delivery_Enedis  = np.zeros((nfleets, nscenarios, nevents))
for s in range(nscenarios):
    for e in range(nevents):
        V1G_delivery_UKPN[:,s,e] = V1G_flex_UKPN[:, d[s,e], t[s, e]:t[s,e]+len_service+1].mean(axis=1)
        V2G_delivery_UKPN[:,s,e] = V2G_flex_UKPN[:, d[s,e], t[s, e]:t[s,e]+len_service+1].mean(axis=1)
        V1G_delivery_Enedis[:,s,e] = V1G_flex_Enedis[:, d[s,e], t[s, e]:t[s,e]+len_service+1].mean(axis=1)
        V2G_delivery_Enedis[:,s,e] = V2G_flex_Enedis[:, d[s,e], t[s, e]:t[s,e]+len_service+1].mean(axis=1)
        
# Payments:
# To simplify. Payment on energy + reduction of av payment on delivered energy / contracted energy.
#  If delivers/contracted < 0.6, no payment 
# For simplicity no penalties considered

min_delivery = 0.6

V1G_delivery_pu_UKPN = V1G_delivery_UKPN / np.tile(V1G_bid_UKPN, (nevents, nscenarios, 1)).T
V2G_delivery_pu_UKPN = V2G_delivery_UKPN / np.tile(V2G_bid_UKPN, (nevents, nscenarios, 1)).T
V1G_delivery_pu_Enedis = V1G_delivery_Enedis / np.tile(V1G_bid_Enedis, (nevents, nscenarios, 1)).T
V2G_delivery_pu_Enedis = V2G_delivery_Enedis / np.tile(V2G_bid_Enedis, (nevents, nscenarios, 1)).T

V1G_delivery_pu_UKPN = (V1G_delivery_pu_UKPN).clip(max=1) * (V1G_delivery_pu_UKPN > min_delivery)
V2G_delivery_pu_UKPN = (V2G_delivery_pu_UKPN).clip(max=1) * (V2G_delivery_pu_UKPN > min_delivery)
V1G_delivery_pu_Enedis = (V1G_delivery_Enedis).clip(max=1) * (V1G_delivery_pu_Enedis > min_delivery)
V2G_delivery_pu_Enedis = (V2G_delivery_Enedis).clip(max=1) * (V2G_delivery_pu_Enedis > min_delivery)    
    
expected_payment = av_payment * (av_window[1]-av_window[0]) * days_of_service + (ut_payment * service_time/60 * nevents)

V1G_payments_UKPN = np.tile(V1G_bid_UKPN, (nscenarios,1)).T * V1G_delivery_pu_UKPN.mean(axis=2) * expected_payment
V2G_payments_UKPN = np.tile(V2G_bid_UKPN, (nscenarios,1)).T * V2G_delivery_pu_UKPN.mean(axis=2) * expected_payment
V1G_payments_Enedis = np.tile(V1G_bid_UKPN, (nscenarios,1)).T * V1G_delivery_pu_Enedis.mean(axis=2) * expected_payment
V2G_payments_Enedis = np.tile(V2G_bid_Enedis, (nscenarios,1)).T * V2G_delivery_pu_Enedis.mean(axis=2) * expected_payment



#%%
av_profs = np.array([ev.off_peak_potential for ev in grid.get_evs()])
ch_profs = np.array([ev.charging for ev in grid.get_evs()])

# new profs at nev = x
nev = 10
nint = [np.random.randint(0,1000,nev) for i in range(1000)]
new_av_profs = np.array([av_profs[n].sum(axis=0) for n in nint])
new_ch_profs = np.array([ch_profs[n].sum(axis=0) for n in nint])

#plot
f, ax2 = plt.subplots()
f, ax1 = plt.subplots()
for i in range(int(len(nint)/5)):
    ax1.plot(new_av_profs[i])
    ax2.plot(new_ch_profs[i])
ax1.plot(new_av_profs.min(axis=0))
ax1.plot(new_ch_profs.mean(axis=0),'k--')
#%% 
ax1.plot(new_ch_profs.mean(axis=0),'k--')
#av_profs_at = 


#%% Simulate company EV fleet, overnight charging!

b_start = 10
b_end = 100
batts = np.arange(b_start, b_end, 1)

ndays = 7
step = 1

# Data from O. Borne thesis
# Normal distr, mean=40, std_dev = 5

cdf = stats.norm.cdf(x=np.linspace(1,100,100),loc=40, scale=5)
ev_data = {'charging_power' : 7.2,
           'charging_type' : 'all_days',
           'arrival_departure_data_wd' : {'mu_arr':16, 'mu_dep':8,
                                          'std_arr':1, 'std_dep':0.5},
           'arrival_departure_data_we' : {'mu_arr':7, 'mu_dep':6.95,
                                          'std_arr':0, 'std_dep':0},
           'dist_wd' : {'cdf': cdf, 'bins' : np.linspace(0,100,101)},
           'n_if_needed' : 0}

grid = evmodel.Grid(ndays=ndays, step=step, name='Grid', verbose=False)
grid.add_evs(nameset='Std', n_evs=1000, ev_type='dumb', **ev_data)
grid.do_days()  

#hists = iterate_batt_size(grid, b_start=b_start, b_end=b_end, fx=charging_sessions)
#%%
av_profs = np.array([ev.off_peak_potential for ev in grid.get_evs()])
ch_profs = np.array([ev.charging for ev in grid.get_evs()])

# new profs at nev = x
nev = 10
nint = [np.random.randint(0,1000,nev) for i in range(1000)]
new_av_profs = np.array([av_profs[n].sum(axis=0) for n in nint])
new_ch_profs = np.array([ch_profs[n].sum(axis=0) for n in nint])

av_w = [16,20] # limits for availability window

avg_ch = new_ch_profs.mean(axis=0)
av_wind = np.ones(int(60*24/step)) # availability window
d = int(60*24/step)
av_wind[0: int(av_w[0]* (60/step))] = 0
av_wind[int(av_w[1]* (60/step)):] = 0
ukpn_bl = sum(av_wind * avg_ch[0+d:int(24*60/step)+d])/av_wind.sum() * av_wind
h = np.linspace(step/60,24,int(24*60/step))

av_ordered = new_av_profs
av_ordered.sort(axis=0)
perc = 0.9
av_perc = av_ordered[int(len(av_ordered)*(1-perc))]
#plot
#f, ax2 = plt.subplots()
f, ax1 = plt.subplots()
#for i in range(int(len(nint)/5)):
#    ax1.plot(h,new_av_profs[i][0+d:int(24*60/step)+d])
##    ax2.plot(new_ch_profs[i])
#ax1.plot(h, av_perc[0+d:int(24*60/step)+d], label='Availability at 90%')
#ax1.plot(h,new_av_profs.min(axis=0)[0+d:int(24*60/step)+d], label='Availability profiles')
av_ch = new_ch_profs.mean(axis=0)[0+d:int(24*60/step)+d]
nonrew = av_ch * av_wind
rew = np.array([av_ch, ukpn_bl]).min(axis=0)
ax1.plot(h,av_ch,'r', label='Average charging profile')
ax1.plot(h,ukpn_bl, 'k--', label='UKPN baseline')
ax1.set_title('UKPN Baseline computed for 10 EV fleet\nAvailability Window between 16 to 20h')
ax1.set_xlabel('Time [h]')
ax1.set_ylabel('Power [MW]')
ax1.stackplot(h, av_wind*1000,color='yellow', labels=['Availability Window'])
ax1.stackplot(h, nonrew,color='green', labels=['Non rewarded flexibility'])
ax1.stackplot(h, ukpn_bl,color='c', labels=['Rewarded for nothing'])
ax1.stackplot(h, rew,color='b', labels=['Rewarded flexibility'])
ax1.set_ylim(0,75)
ax1.legend(loc=3)

#%% 

av_profs_at = 