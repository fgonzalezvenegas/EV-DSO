# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:39:56 2020
Functions to compute revenue payments
@author: U546416
"""


import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as stats
import time
import util



#%% Useful functions
    
def split_by_days(data, step=5):
    """ Returns an array splited by days
    Data: 2-dim array, with shape [nEVs, data(ev)]
    Step: Length of step size of data (minutes, int)
    Shift: Hours to be shifted (hours<24, int)
    """
    steps_day = int(60/step * 24)
#    ndim = np.ndim(data)
#    if ndim == 1:
#        ndays = int(len(data) / steps_day)
#        return np.reshape(data, [ndays, steps_day])
#    else:
    nevs, lendata = np.shape(data)
    ndays = int(lendata / steps_day)
    return np.reshape(data, [nevs, ndays, steps_day])
    
def drop_days(data, days_before=7, days_after=1, step=5, 
              drop_we=False, shift=0):
    """ Returns the data array without buffer days (before and after)
    """
    
        
    idx_tini = ((days_before * 24) + shift) * int(60/step)
    
    idx_tend = ((days_after * 24) + (24 - shift))* int(60/step)
    if np.ndim(data) == 3: # This means there are more than one flex time profiles
        nevs, nflex, lendata = np.shape(data)
        drops =  data[:,:,idx_tini:lendata-idx_tend]
    else:
        nevs, lendata = np.shape(data)
        drops =  data[:,idx_tini:lendata-idx_tend]
    
    if drop_we:
        drops = drop_weekends(drops, step)
    
    return drops
    
def drop_weekends(data, step=5):
    """ Returns the data array without weekends.
    It assumes data starts on a monday
    """
    steps_day = int(60/step * 24)
#    ndim = np.ndim(data)
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
    
def get_av_window(profs, av_window_idxs):
    return profs[:,:,av_window_idxs[0]:av_window_idxs[1]]

def get_ev_profs(grid, nameset='all', ovn=True, av_days='wd', step=5):
    
    if nameset=='all':
        evs = grid.get_evs()
    else:
        evs = grid.evs_sets[nameset]
    ch_profs = np.array([ev.charging for ev in evs])
    
    # Possible kWs that could be proposed to DSO, for a flex service (not yet taking into account baselines)
    try:
        dn_profs = np.array([ev.dn_flex_kw for ev in evs])
    except:
        dn_profs = None
        dn_profs_multi=None
#    elif baseline in ['delayed', 'del', 'd']:
#        dn_profs =  np.array([ev.dn_flex_kw_delayed for ev in evs])
#    elif baseline in ['meantraj', 'mean', 'm', 'mean_trajectory', 'mt']:
#        dn_profs =  np.array([ev.dn_flex_kw_delayed for ev in evs])
    
        
         
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
    if dn_profs is None:
        pass
    else:
        if dn_profs.ndim == 3:
            dn_profs_multi = {}
            for i in range(dn_profs.shape[1]):
                dn_profs_multi[i] = split_by_days(drop_days(dn_profs[:,i,:], days_before=7, 
                                                 days_after=days_after, step=step, 
                                                 drop_we=drop_we, shift=shift), step=step)
        else:
            dn_profs_multi = split_by_days(drop_days(dn_profs, days_before=7, 
                                             days_after=days_after, step=step, 
                                             drop_we=drop_we, shift=shift), step=step)
     
    ch_profs = split_by_days(drop_days(ch_profs, days_before=7,
                                       days_after=days_after, step=step, 
                                       drop_we=drop_we, shift=shift), step=step)
    return ch_profs, dn_profs_multi
 
def get_fleet_profs(ch_profs, dn_profs, nevs_fleet, nfleets=1000):      
    (n_evs, ndays, nsteps) = ch_profs.shape
    
    # Compute aggregated profiles for EV fleets
    nint = np.array([np.random.choice(range(n_evs),size=nevs_fleet, replace=False)
                    for i in range(nfleets)]) # Fleets, based on combinations of EV indexes
    
    fleet_ch_profs = np.zeros((nfleets, ndays, nsteps))
    if type(dn_profs) == dict:
        fleet_dn = {k : np.zeros((nfleets, ndays, nsteps)) 
                    for k in dn_profs.keys()}
    else:
        fleet_dn = []    
    for i in range(nfleets):
#     up & dn profiles
        for j in range(nevs_fleet):
            fleet_ch_profs[i] += ch_profs[nint[i,j]]
            if dn_profs is None:
                pass
            else:
                if type(dn_profs) == dict:
                    for k in dn_profs.keys():
                        fleet_dn[k][i] += dn_profs[k][nint[i,j]]
                else:
                    fleet_dn[i] += dn_profs[nint[i,j]]
                    
#    else:
#        fleet_dn = np.array([dn_profs[nint[i]].sum(axis=0) for i in range(nfleets)])
#    # Charging profiles for fleet
#    fleet_ch_profs = np.array([ch_profs[nint[i]].sum(axis=0) for i in range(nfleets)])
    return fleet_ch_profs, fleet_dn


def get_baselines(fleet_ch_profs, bl='UKPN', ndays_bl=10, step=5):
    """ 
    UKPN baseline: n representative days, plus a uniform BL during the availability window
    Enedis bl: Panel of similar users. 
    Lets say we take the average for the fleet (as it is supposed to be representative)
    """
    (nfleets, ndays, nsteps) = fleet_ch_profs.shape
    d = np.random.choice(ndays, ndays_bl, replace=False)
    if bl in ['UKPN', 'unique']:
        # UKPN baseline: n representative days, plus a uniform BL during the availability window
        UKPN_bl = (fleet_ch_profs[:, d, :]).mean(axis=(1,2))
    
        # As a matrix of dim (nfleets, ndays, nsteps)
        ukpn_bls = np.tile(UKPN_bl, (nsteps, ndays, 1)).T
        return ukpn_bls

    elif bl in ['Enedis', '30min', 'panel', 'Panel']:
#        Enedis bl: Panel of similar users. 
#    Lets say we take the half-hourly average for the fleet (as it is supposed to be representative)
        nhh = int(nsteps * step / 30)
        nsteps_hh = int(30/step)
        enedis_bls = np.zeros((nfleets, ndays, nsteps))
        for i in range(nhh):
            a = fleet_ch_profs[:,d,i*nsteps_hh:(i+1)*nsteps_hh].mean(axis=(1,2))
            enedis_bls[:,:,i*nsteps_hh:(i+1)*nsteps_hh] = np.tile(a, (nsteps_hh, ndays, 1)).T
        return enedis_bls
    
def get_flex_wrt_bl(fleet_dn, baseline, baseload=0, V2G=True):
    if type(fleet_dn) == dict:
        if V2G:    
            return {k: baseline - baseload - fleet_dn[k]
                    for k in fleet_dn.keys()}
        else: # V1G flex
            return {k: baseline - baseload - fleet_dn[k].clip(min=0)
                    for k in fleet_dn.keys()}
    if V2G:    
        return baseline - baseload - fleet_dn
    else: # V1G flex
        return baseline - baseload - fleet_dn.clip(min=0)

def compute_payments(flex, av_payment, ut_payment, 
                     nevents, days_of_service, conf=0.9,
                     service_time=30, step=5, penalty=0,
                     min_delivery=0.6, min_bid=50, nscenarios=1000):
    """ 
    """
    (nfleets, ndays, nsteps) = flex.shape
#    nsteps_service = int(service_time / step) # Number of steps
    len_av_window = nsteps * step / 60 # Hours
    # define kws. Option 1: mean value of expected flex kWs (per fleet?)
#    flex_bid = flex.mean(axis=(1,2))
    
    # define kws. Option 2: mean of profile at 90%?
    # It is one bid per fleet
#    nconf = int((1-conf) * ndays)
#    #sorts by day, select profile at nconf and do avg
#    flex_bid = np.mean(np.sort(flex, axis=1)[:,nconf,:], axis=1) 
    # option 2: among all days and times, it needs to fulfill the confidence level
    nconf = int((1-conf) * nsteps * ndays)
    flex_bid = np.sort(flex.reshape((nfleets, ndays*nsteps)), axis=1)[:, nconf]
    
    # Cut bids under minimum bid
    flex_bid = flex_bid * (flex_bid > min_bid)
    
    # evaluate delivery
    d = np.random.randint(0, ndays, (nscenarios, nevents))
    t = np.random.randint(0, nsteps, (nscenarios, nevents))
    
    flex_delivery  = flex[:,d,t]
    
    # Payments:
    # To simplify. Payment on energy + reduction of av payment on delivered energy / contracted energy.
    #  If delivers/contracted < 0.6, no payment 

    flex_delivery_pu = flex_delivery / np.tile(flex_bid, (nevents, nscenarios, 1)).T
    flex_delivery_pu = (flex_delivery_pu).clip(max=1) * (flex_delivery_pu > min_delivery)

    undelivery_ratio = (flex_delivery_pu < min_delivery).mean(axis=(2))

    # correcting nans that arise when flex_bid == 0
    flex_delivery_pu = np.nan_to_num(flex_delivery_pu)
    
    # expected payment for each firm kW    
    expected_payment = (av_payment * len_av_window * days_of_service) / 1000 + (ut_payment * service_time/60 * nevents) / 1000
    
    # Penalties considered for activations with under-delivery<min_delivery%. 
    # Penalties = Expected payment * penalty [pu]
    flex_payments = (np.tile(flex_bid, (nscenarios,1)).T * flex_delivery_pu.mean(axis=2) * expected_payment - 
                     np.tile(flex_bid, (nscenarios,1)).T * undelivery_ratio * expected_payment * penalty)
    
    return flex_bid, flex_payments.flatten(), undelivery_ratio.flatten()

def get_av_window_vector(av_window, step, ovn=True):
    if ovn:
        shift = 12
    else:
        shift = 0
    nsteps = int(24 * 60 / step)
    av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
    av_window_vector = np.concatenate((np.zeros(av_window_idxs[0]), 
                                   np.ones(av_window_idxs[1]-av_window_idxs[0]),
                                   np.zeros(nsteps - av_window_idxs[1])))
    return av_window_vector

def get_av_window_matrix(av_window_vector, nfleets, ndays):
    # As a matrix of dim (nfleets, ndays, nsteps)
    av_window_matrix = np.tile(av_window_vector, (nfleets, ndays, 1))
    return av_window_matrix


def get_stats(data, percentile=90):
    return (np.mean(data), np.min(data), np.max(data),
            np.percentile(data, percentile), np.percentile(data, 100-percentile), 
            np.sort(data)[0:int(len(data)*(100-percentile)/100)].mean()) # CVaR
