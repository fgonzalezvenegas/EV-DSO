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
    
def drop_days(data, days_before=7, days_after=1, step=5, drop_we=False, shift=0):
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

def get_ev_profs(grid, nameset='all', ovn=True, av_days='wd', baseline='immediate', step=5):
    
    if nameset=='all':
        evs = grid.get_evs()
    else:
        evs = grid.evs[nameset]
    ch_profs = np.array([ev.charging for ev in evs])
    
    # Possible kWs that could be proposed to DSO, for a flex service (not yet taking into account baselines)
    if baseline in ['imm', 'im', 'i', 'immediate', 'dumb']:
        dn_profs = np.array([ev.dn_flex_kw_immediate for ev in evs])
    elif baseline in ['delayed', 'del', 'd']:
        dn_profs =  np.array([ev.dn_flex_kw_delayed for ev in evs])
    elif baseline in ['meantraj', 'mean', 'm', 'mean_trajectory', 'mt']:
        dn_profs =  np.array([ev.dn_flex_kw_delayed for ev in evs])
         
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
        
    dn_profs = split_by_days(drop_days(dn_profs, days_before=7, 
                                         days_after=days_after, step=step, 
                                         drop_we=drop_we, shift=shift), step=step)
 
    ch_profs = split_by_days(drop_days(ch_profs, days_before=7,
                                       days_after=days_after, step=step, 
                                       drop_we=drop_we, shift=shift), step=step)
    return ch_profs, dn_profs
 
def get_fleet_profs(ch_profs, dn_profs, nevs_fleet, nfleets=1000):      
    (n_evs, ndays, nsteps) = dn_profs.shape
    
    # Compute aggregated profiles for EV fleets
    nint = [np.random.randint(0,n_evs,nevs_fleet) for i in range(nfleets)] # Fleets, based on combinations of EV indexes 
    
    # up & dn profiles
    fleet_dn = np.array([dn_profs[n].sum(axis=0) for n in nint])
    # Charging profiles for fleet
    fleet_ch_profs = np.array([ch_profs[n].sum(axis=0) for n in nint])
    return fleet_ch_profs, fleet_dn

def get_baselines(fleet_ch_profs, bl='UKPN', ndays_bl=10, step=5):
    """ 
    UKPN baseline: n representative days, plus a uniform BL during the availability window
    Enedis bl: Panel of similar users. 
    Lets say we take the average for the fleet (as it is supposed to be representative)
    """
    (nfleets, ndays, nsteps) = fleet_ch_profs.shape
    d = np.random.choice(ndays, ndays_bl, replace=False)
    if bl in ['UKPN']:
        # UKPN baseline: n representative days, plus a uniform BL during the availability window
        UKPN_bl = (fleet_ch_profs[:, d, :]).mean(axis=(1,2))
    
        # As a matrix of dim (nfleets, ndays, nsteps)
        ukpn_bls = np.tile(UKPN_bl, (nsteps, ndays, 1)).T
        return ukpn_bls

    elif bl in ['Enedis']:
#        Enedis bl: Panel of similar users. 
#    Lets say we take the half-hourly average for the fleet (as it is supposed to be representative)
        nhh = int(nsteps * step / 30)
        nsteps_hh = int(30/step)
        enedis_bls = np.zeros((nfleets, ndays, nsteps))
        for i in range(nhh):
            a = fleet_ch_profs[:,d,i*nsteps_hh:(i+1)*nsteps_hh].mean(axis=(1,2))
            enedis_bls[:,:,i*nsteps_hh:(i+1)*nsteps_hh] = np.tile(a, (nsteps_hh, ndays, 1)).T
        return enedis_bls
    
def get_flex_wrt_bl(fleet_dn, baseline, V2G=True):
    if V2G:    
        return baseline - fleet_dn
    else: # V1G flex
        return baseline - fleet_dn.clip(min=0)

def compute_payments(flex, av_payment, ut_payment, 
                     nevents, days_of_service, conf=0.9,
                     service_time=30, step=5,
                     min_delivery=0.6, min_bid=50, nscenarios=1000):
    """ 
    """
    (nfleets, ndays, nsteps) = flex.shape
    nsteps_service = int(service_time / step) # Number of steps
    len_av_window = nsteps * step / 60 # Hours
    # define kws. Option 1: mean value of expected flex kWs (per fleet?)
#    flex_bid = flex.mean(axis=(1,2))
    
    # define kws. Option 2: mean of profile at 90%?
    nconf = int((1-conf) * ndays)
    flex_bid = np.mean(np.sort(flex, axis=1)[:,nconf,:], axis=1) #sorts by day, select profile at nconf and do avg
    
    
    # Cut bids under minimum bid
    flex_bid = flex_bid * (flex_bid > min_bid)
    
    # evaluate delivery
    d = np.random.randint(0, ndays, (nscenarios, nevents))
    t = np.random.randint(0, nsteps - nsteps_service, (nscenarios, nevents))
    
    flex_delivery  = np.zeros((nfleets, nscenarios, nevents))
    
    for s in range(nscenarios):
        for e in range(nevents):
            flex_delivery[:,s,e] = flex[:, d[s,e], t[s, e]:t[s,e]+nsteps_service+1].mean(axis=1)
            
    # Payments:
    # To simplify. Payment on energy + reduction of av payment on delivered energy / contracted energy.
    #  If delivers/contracted < 0.6, no payment 
    # For simplicity no penalties considered
    
    flex_delivery_pu = flex_delivery / np.tile(flex_bid, (nevents, nscenarios, 1)).T
    
    flex_delivery_pu = (flex_delivery_pu).clip(max=1) * (flex_delivery_pu > min_delivery)
    
    # expected payment for each firm kW    
    expected_payment = (av_payment * len_av_window * days_of_service) / 1000 + (ut_payment * service_time/60 * nevents) / 1000
    
    flex_payments = np.tile(flex_bid, (nscenarios,1)).T * flex_delivery_pu.mean(axis=2) * expected_payment
    return flex_bid, flex_payments.flat

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
    return np.mean(data), np.min(data), np.max(data), np.percentile(data, percentile), np.percentile(data, 100-percentile)
