# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:39:37 2019
Runs EV model over a week
Does some graphs in french/english

@author: U546416
"""
import numpy as np
from matplotlib import pyplot as plt
import model as evmodel

ndays = 8
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
        
#%% Computes 4 combinations of dumb charging 
#ev_data1 = {'EV Load': {'type' : 'dumb',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type' : 'if_needed'}}}
ev_data2 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days'}}}
ev_data3 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday'}}}
#ev_data3 = {'NonCoordonne': {'type' : 'dumb',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type' : 'if_needed',
#                         'charging_power' : 7.2}}}
#ev_data4 = {'NonCoordonne': {'type' : 'dumb',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type': 'all_days',
#                         'charging_power' : 7.2}}}
#grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step)
#grid1.do_days()
grid2 = evmodel.Grid(ev_data2, ndays=ndays, step=step)
grid2.do_days()
grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step)
for i in range(1000):
    grid3.evs['EV Load'][i].dist_wd = grid2.evs['EV Load'][i].dist_wd
    grid3.evs['EV Load'][i].dist_we = grid2.evs['EV Load'][i].dist_we
grid3.do_days()
#grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step)
#grid3.do_days()
#grid4 = evmodel.Grid(ev_data4, ndays=ndays, step=step)
#grid4.do_days()
#f, ([ax1, ax2],  [ax3, ax4]) = plt.subplots(2,2)
#ylim = 7.2*1000
f, (ax1, ax2) = plt.subplots(1,2)
ylim = 3.7*1000
grid2.plot_evload(ax=ax1, ylim=ylim, title='Non Coordonnée, Tous les jours')
#grid1.plot_evload(ax=ax2, ylim=ylim, title='Non Coordonnée, Si besoin')
grid3.plot_evload(ax=ax2, ylim=ylim, title='Non Coordonnée, Si besoin-Dim')
#set_labels_fr(ax1)
#set_labels_fr(ax2)
#set_labels_fr(ax3)
#grid4.plot_evload(ax=ax3, ylim=ylim, title='Dumb, 7.2kW, All days')
#grid3.plot_evload(ax=ax4, ylim=ylim, title='Dumb, 7.2kW, If needed')
print('all_days', grid2.compute_global_data())
#print('if_needed', grid1.compute_global_data())
print('sunday', grid3.compute_global_data())

f, (ax1, ax2) = plt.subplots(1,2)
grid2.plot_flex_pot(ax=ax1, title='Non Coordonnée, Tous les jours')
#grid1.plot_evload(ax=ax2, ylim=ylim, title='Non Coordonnée, Si besoin')
grid3.plot_flex_pot(ax=ax2, title='Non Coordonnée, Si besoin-Dim')
#%%
f, ax = plt.subplots(1,1)
flex_alldays = grid2.ev_up_flex['Total'] - grid2.ev_dn_flex['Total']
flex_ifneedd = grid3.ev_up_flex['Total'] - grid3.ev_dn_flex['Total']
ax.plot(flex_alldays, label='All days')
ax.plot(flex_ifneedd, label='If needed')
ax.set_xlabel('Time [h]')
ax.set_ylabel('Flexible storage capacity [kWh]')
ax.legend(loc=2)
ax.set_ylim([0, 35e3])
ax.set_xlim([0, 24*4*8])
#%% Computes 2 Peak-Off Peak, and modulated dumb and ToU

ev_data3 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 3.6,
                         'tou_ini' : 23,
                         'tou_end' : 7}}}
ev_data4 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type': 'all_days',
                         'charging_power' : 3.6,
                         'tou_ini' : 23,
                         'tou_end' : 7}}}
#ev_data5 = {'Dumb_Modulated': {'type' : 'mod',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type' : 'if_needed',
#                         'charging_power' : 3.6}}}
#ev_data6 = {'Dumb_Modulated': {'type' : 'mod',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type': 'all_days',
#                         'charging_power' : 3.6}}}
#ev_data7 = {'ToU_Modulated': {'type' : 'mod',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type' : 'if_needed',
#                         'charging_power' : 3.6,
#                         'tou_ini' : 23,
#                         'tou_end' : 7}}}
#ev_data8 = {'ToU_Modulated': {'type' : 'mod',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type': 'all_days',
#                         'charging_power' : 3.6,
#                         'tou_ini' : 23,
#                         'tou_end' : 7}}}


grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step)
grid3.do_days()
grid4 = evmodel.Grid(ev_data4, ndays=ndays, step=step)
grid4.do_days()
#grid5 = evmodel.Grid(ev_data5, ndays=ndays, step=step)
#grid5.do_days()
#grid6 = evmodel.Grid(ev_data6, ndays=ndays, step=step)
#grid6.do_days()
#grid7 = evmodel.Grid(ev_data7, ndays=ndays, step=step)
#grid7.do_days()
#grid8 = evmodel.Grid(ev_data8, ndays=ndays, step=step)
#grid8.do_days()

f, (ax1, ax2) = plt.subplots(1,2)
ylim = 3.7*1000
#grid2.plot_evload(ax=ax1, ylim=ylim, title='Dumb, 3.6kW, All days')
#grid1.plot_evload(ax=ax3, ylim=ylim, title='Dumb, 3.6kW, If needed')
grid4.plot_evload(ax=ax1, ylim=ylim, title='ToU, Tous les jours')
grid3.plot_evload(ax=ax2, ylim=ylim, title='ToU, Si besoin')
set_labels_fr(ax1)
set_labels_fr(ax2)

#f, ([ax1, ax2],  [ax3, ax4]) = plt.subplots(2,2)
#grid2.plot_evload(ax=ax1, ylim=ylim, title='Dumb, 3.6kW, All days')
#grid1.plot_evload(ax=ax3, ylim=ylim, title='Dumb, 3.6kW, If needed')
#grid6.plot_evload(ax=ax2, ylim=ylim, title='Modulated, 3.6kW, All days')
#grid5.plot_evload(ax=ax4, ylim=ylim, title='Modulated, 3.6kW, If needed')
#
#f, ([ax1, ax2],  [ax3, ax4]) = plt.subplots(2,2)
#grid4.plot_evload(ax=ax1, ylim=ylim, title='ToU, 3.6kW, All days')
#grid3.plot_evload(ax=ax3, ylim=ylim, title='ToU, 3.6kW, If needed')
#grid8.plot_evload(ax=ax2, ylim=ylim, title='ToU Modulated, 3.6kW, All days')
#grid7.plot_evload(ax=ax4, ylim=ylim, title='ToU Modulated, 3.6kW, If needed')

#%% Computes 4 combinations of dumb charging different battery size
ev_data1 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24}}}
ev_data2 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 40}}}
ev_data3 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 60}}}
ev_data4 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 7.2,
                         'batt_size' : 24}}}
ev_data5 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 7.2,
                         'batt_size' : 40}}}
ev_data6 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 7.2,
                         'batt_size' : 60}}}
grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step)
grid1.do_days()
grid2 = evmodel.Grid(ev_data2, ndays=ndays, step=step)
grid2.do_days()
grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step)
grid3.do_days()
grid4 = evmodel.Grid(ev_data4, ndays=ndays, step=step)
grid4.do_days()
grid5 = evmodel.Grid(ev_data5, ndays=ndays, step=step)
grid5.do_days()
grid6 = evmodel.Grid(ev_data6, ndays=ndays, step=step)
grid6.do_days()
f, (ax1, ax2, ax3) = plt.subplots(1,3)
ylim = 3.7*1000
grid1.plot_evload(ax=ax1, ylim=ylim, title='Batterie 24 kWh')
grid2.plot_evload(ax=ax2, ylim=ylim, title='Batterie 40 kWh')
grid3.plot_evload(ax=ax3, ylim=ylim, title='Batterie 60 kWh')
set_labels_fr(ax1)
set_labels_fr(ax2)
set_labels_fr(ax3)

f, (ax1, ax2, ax3) = plt.subplots(1,3)
ylim = 7.3*1000
grid4.plot_evload(ax=ax1, ylim=ylim, title='Batterie 24 kWh')
grid5.plot_evload(ax=ax2, ylim=ylim, title='Batterie 40 kWh')
grid6.plot_evload(ax=ax3, ylim=ylim, title='Batterie 60 kWh')
set_labels_fr(ax1)
set_labels_fr(ax2)
set_labels_fr(ax3)
#%% Histogram in %
#grid1 = evmodel.Grid(ev_data1, ndays=ndays-1, step=step)
#grid1.do_days()
#grid2 = evmodel.Grid(ev_data2, ndays=ndays-1, step=step)
#grid2.do_days()
#grid3 = evmodel.Grid(ev_data3, ndays=ndays-1, step=step)
#grid3.do_days()
grid4 = evmodel.Grid(ev_data1, ndays=ndays-1, step=step)
grid4.do_days()
grid5 = evmodel.Grid(ev_data2, ndays=ndays-1, step=step)
grid5.do_days()
grid6 = evmodel.Grid(ev_data3, ndays=ndays-1, step=step)
grid6.do_days()
#%% Hisogram of dist
f, axs = plt.subplots(1,3)
grids = [grid1, grid2, grid3]
grids = [grid4, grid5, grid6]
ds = [[]] * 3
dhs = [[]] * 3
hs = [[]] * 3
bins = [i for i in range(9)]
batts = [24, 40, 60]
#for i in range(3):
#    ds[i] = np.asarray([ev.ch_status.sum() for types in grids[i].evs
#                                for ev in grids[i].evs[types]])
#    hs[i] = np.histogram(ds[i], bins)
#    axs[i].bar(hs[i][1][:-1], hs[i][0]/sum(hs[i][0]))
#    axs[i].set_xlabel('# de sessions de recharge')    
#    axs[i].set_ylabel('Densité')
#    axs[i].set_title('Batterie %d kWh' %batts[i])
#    axs[i].set_xlim([0,8])
#    axs[i].set_xticks([i+1 for i in range(7)])
#    axs[i].set_ylim([0,0.700])
for i in range(3):
    ds[i] = np.asarray([ev.ch_status.sum() for types in grids[i].evs
                                for ev in grids[i].evs[types]])
    hs[i] = np.histogram(ds[i], bins)
    dhs[i] = hs[i][0]/sum(hs[i][0])
    axs[i].bar(hs[i][1][:-1], dhs[i])
    axs[i].set_xlabel('Charging sessions')    
    axs[i].set_ylabel('Density')
    axs[i].set_title('%d kWh' %batts[i])
    axs[i].set_xlim([0,8])
    axs[i].set_xticks([i+1 for i in range(7)])
    axs[i].set_ylim([0,0.700])
#%
ff, ax = plt.subplots(1,1)
w = 0.25
delta = [-w, 0, w]
for i in range(3):
    ax.bar(hs[i][1][:-1]+ delta[i], dhs[i], width=w, label='%d kWh' %batts[i])
ax.legend()
ax.set_xlim([0,8])
ax.set_xticks([i+1 for i in range(7)])
ax.set_xlabel('Charging sessions')    
ax.set_ylabel('Density')
#grid4.do_ncharging_hist(ax=ax4, title='#Charging Sessions, Bat=24 kWh, 7.2kW')
#grid5.do_ncharging_hist(ax=ax5, title='#Charging Sessions, Bat=40 kWh, 7.2kW')
#grid6.do_ncharging_hist(ax=ax6, title='#Charging Sessions, Bat=60 kWh, 7.2kW')
#%% Do histogram of distances
#import model as evmodel

#ev_data1 = {'dumb': {'type' : 'dumb',
#                 'n_ev' : 1000,
#                 'other': {
#                         'charging_type' : 'if_needed',
#                         'batt_size' : 24}}}
#grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step)
grid1.do_dist_hist()
#grid1.do_days()
#%% Sensitivity on range anxiety factor

ev_data1 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 40,
                         'range_anx_factor' : 1.5}}}

ev_data2 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 40,
                         'range_anx_factor' : 2}}}

ev_data3 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 40,
                         'range_anx_factor' : 2.5}}}

grid1 = evmodel.Grid(ev_data1, ndays=ndays-1, step=step)
grid1.do_days()
grid2 = evmodel.Grid(ev_data2, ndays=ndays-1, step=step)
grid2.do_days()
grid3 = evmodel.Grid(ev_data3, ndays=ndays-1, step=step)
grid3.do_days()


grids = [grid1, grid2, grid3]
ds = [[]] * 3
dhs = [[]] * 3
hs = [[]] * 3
bins = [i for i in range(9)]
ranges = [1.5, 2.0, 2.5]

for i in range(3):
    ds[i] = np.asarray([ev.ch_status.sum() for types in grids[i].evs
                                for ev in grids[i].evs[types]])
    hs[i] = np.histogram(ds[i], bins)
    dhs[i] = hs[i][0]/sum(hs[i][0])

ff, ax = plt.subplots(1,1)
w = 0.25
delta = [-w, 0, w]
for i in range(3):
    ax.bar(hs[i][1][:-1]+ delta[i], dhs[i], width=w, label='RAF %1.1f' %ranges[i])
ax.legend()
ax.set_xlim([0,8])
ax.set_xticks([i+1 for i in range(7)])
ax.set_xlabel('Charging sessions')    
ax.set_ylabel('Density')

#%% Load Data - real sims 
import model as evmodel
print('loading data')
load_by_comm, load_profiles, SS = evmodel.load_conso_ss_data()
hh, hw = evmodel.load_hist_data()

#%% Run Simulation for a given SS, Dumb charging -Vaudreuil
import model as evmodel
print('preprocessing data')
ss = 'VAUDREUIL'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=1), 15)[:-1]


# Computing pdf of home and work
print('computing pdfs')
hh_ss = evmodel.extract_hist(hh, SS.Communes[ss]).sum()
nh = hh_ss.sum()
pdf_hh = hh_ss.cumsum()/nh

hw_ss = evmodel.extract_hist(hw, SS.Communes[ss]).sum()
nw = hw_ss.sum()
pdf_hw = hw_ss.cumsum()/nw

# % work and home and penetration
voitures_actifs = 0.5
voitures_non_compta = 2
ev_penetration = 0.5
home_vs_work = 1 # 1 only home, 0 only work
nevh = int(nh * ev_penetration * home_vs_work * voitures_actifs * voitures_non_compta)
nevw = int(nw * ev_penetration * (1-home_vs_work) * voitures_actifs * voitures_non_compta)
ev_data1 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' :  3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
ev_data2 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep : 0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2,
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
                          
print('Simulating grids with %d EVs home charging, %d EVs work charging' 
      %(int(nh * ev_penetration * home_vs_work),int(nw * ev_penetration * (1-home_vs_work))))

grid1 = evmodel.Grid(ev_data1, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid2 = evmodel.Grid(ev_data2, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid1.do_days()
grid2.do_days()
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_tot_load(ax=ax1, title='%s Tot Load - All Days' %ss)
grid2.plot_tot_load(ax=ax2, title='%s Tot Load - If Needed' %ss)
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_evload(ax=ax1, title='%s EV Load - All Days' %ss)
grid2.plot_evload(ax=ax2, title='%s EV Load - If Needed' %ss)
f, ax1 = plt.subplots(1,1)
grid1.do_dist_hist(ax=ax1)

##%% Run Simulation for a given SS - ToU Charging
#import model as evmodel
#
## % work and home and penetration
#ev_penetration = 1
#home_vs_work = 1 # 1 only home, 0 only work
#
#ev_data1 = {'home': {'type' : 'tou', #dumb, tou
#                 'n_ev' : int(nh * ev_penetration * home_vs_work),
#                 'other': {
#                         'charging_type' : 'all_days',
#                         'batt_size' : 24, #,
##                         'tou_ini' : 1, 
##                         'tou_end' : 8,
#                          'pdf_dist' : pdf_hh, 
##                          'charging_power' :  3.6, 
##                          'charging_eff' : 0.95, 
##                          'driving_eff' : 0.2
##                          'range_anx_factor' : 1.5,
##                          'extra_trip_proba' : 0.2
#                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
#                                                      #pdf_dep  0, 
#                                                      'mu_arr' : 18, 
#                                                      'mu_dep' : 8, 
#                                                      'std_arr' : 1, 
#                                                      'std_dep' : 1/3}
#                          }},
#            'work': {'type' : 'tou',     #dumb, tou
#                 'n_ev' : int(nw * ev_penetration * (1-home_vs_work)),
#                 'other': {
#                         'charging_type' : 'all_days',
#                         'batt_size' : 24, #,
##                         'tou_ini' : 1, 
##                         'tou_end' : 8,
#                          'pdf_dist' : pdf_hw, 
##                          'charging_power' : 3.6, 
##                          'charging_eff' : 0.95, 
##                          'driving_eff' : 0.2
##                          'range_anx_factor' : 1.5,
##                          'extra_trip_proba' : 0.2
#                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
#                                                      #pdf_dep  0, 
#                                                      'mu_arr' : 9, 
#                                                      'mu_dep' : 17, 
#                                                      'std_arr' : 1, 
#                                                      'std_dep' : 1}
#                         }}}
#ev_data2 = {'home': {'type' : 'tou',     #dumb, tou
#                 'n_ev' : int(nh * ev_penetration * home_vs_work),
#                 'other': {
#                         'charging_type' : 'if_needed',
#                         'batt_size' : 24,
##                         'tou_ini' : 1, 
##                         'tou_end' : 8,
#                          'pdf_dist' : pdf_hh, 
##                          'charging_power' : 3.6, 
##                          'charging_eff' : 0.95, 
##                          'driving_eff' : 0.2
##                          'range_anx_factor' : 1.5,
##                          'extra_trip_proba' : 0.2
#                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
#                                                      #pdf_dep : 0, 
#                                                      'mu_arr' : 18, 
#                                                      'mu_dep' : 8, 
#                                                      'std_arr' : 1, 
#                                                      'std_dep' : 1/3}
#                          }},
#            'work': {'type' : 'tou',     #dumb, tou
#                 'n_ev' : int(nw * ev_penetration * (1-home_vs_work)),
#                 'other': {
#                         'charging_type' : 'if_needed',
#                         'batt_size' : 24,
##                         'tou_ini' : 1, 
##                         'tou_end' : 8,
#                          'pdf_dist' : pdf_hw, 
##                          'charging_power' : 3.6, 
##                          'charging_eff' : 0.95, 
##                          'driving_eff' : 0.2
##                          'range_anx_factor' : 1.5,
##                          'extra_trip_proba' : 0.2,
#                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
#                                                      #pdf_dep  0, 
#                                                      'mu_arr' : 9, 
#                                                      'mu_dep' : 17, 
#                                                      'std_arr' : 1, 
#                                                      'std_dep' : 1}
#                         }}}
#
#print('Simulating grids with %d EVs home charging, %d EVs work charging' 
#      %(int(nh * ev_penetration * home_vs_work),int(nw * ev_penetration * (1-home_vs_work))))
#
#grid1 = evmodel.Grid(ev_data1, 
#                     ndays=ndays, step=step, 
#                     load = np.asarray(load.sum(axis=1)), 
#                     ss_pmax=SS.Pmax[ss], 
#                     name=ss)
#grid2 = evmodel.Grid(ev_data2, 
#                     ndays=ndays, step=step, 
#                     load = np.asarray(load.sum(axis=1)), 
#                     ss_pmax=SS.Pmax[ss], 
#                     name=ss)
#grid1.do_days()
#grid2.do_days()
#f, (ax1, ax2) = plt.subplots(1,2)
#grid1.plot_tot_load(ax=ax1, title='%s Tot Load - All Days' %ss)
#grid2.plot_tot_load(ax=ax2, title='%s Tot Load - If Needed' %ss)
#f, (ax1, ax2) = plt.subplots(1,2)
#grid1.plot_evload(ax=ax1, title='%s EV Load - All Days' %ss)
#grid2.plot_evload(ax=ax2, title='%s EV Load - If Needed' %ss)
#f, ax1 = plt.subplots(1,1)
#grid1.do_dist_hist(ax=ax1)
#

#%% Load Data - real sims - Vanves
import model as evmodel
#print('loading data')
#load_by_comm, load_profiles, SS = evmodel.load_conso_ss_data()
#hh, hw = evmodel.load_hist_data()
##%
print('preprocessing data')
ss = 'VANVES'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=1), 15)[:-1]


# Computing pdf of home and work
print('computing pdfs')
hh_ss = evmodel.extract_hist(hh, SS.Communes[ss]).sum()
nh = hh_ss.sum()
pdf_hh = hh_ss.cumsum()/nh

hw_ss = evmodel.extract_hist(hw, SS.Communes[ss]).sum()
nw = hw_ss.sum()
pdf_hw = hw_ss.cumsum()/nw

#%% Run Simulation for a given SS, Dumb charging
import model as evmodel

voitures_actifs = 0.27
voitures_non_compta = 2
ev_penetration = 1
home_vs_work = 1 # 1 only home, 0 only work
nevh = int(nh * ev_penetration * home_vs_work * voitures_actifs * voitures_non_compta)
nevw = int(nw * ev_penetration * (1-home_vs_work) * voitures_actifs * voitures_non_compta)
ev_data1 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' :  3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
ev_data2 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep : 0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2,
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
                          
print('Simulating grids with %d EVs home charging, %d EVs work charging' 
      %(int(nh * ev_penetration * home_vs_work),int(nw * ev_penetration * (1-home_vs_work))))

grid1 = evmodel.Grid(ev_data1, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid2 = evmodel.Grid(ev_data2, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid1.do_days()
grid2.do_days()
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_tot_load(ax=ax1, title='%s Tot Load - All Days' %ss)
grid2.plot_tot_load(ax=ax2, title='%s Tot Load - If Needed' %ss)
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_evload(ax=ax1, title='%s EV Load - All Days' %ss)
grid2.plot_evload(ax=ax2, title='%s EV Load - If Needed' %ss)
f, ax1 = plt.subplots(1,1)
grid1.do_dist_hist(ax=ax1)

#%% Run Simulation for a given SS - ToU Charging

# % work and home and penetration
ev_penetration = 0.4
home_vs_work = 1 # 1 only home, 0 only work

ev_data1 = {'home': {'type' : 'tou', #dumb, tou
                 'n_ev' : int(nh * ev_penetration * home_vs_work),
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' :  3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'tou',     #dumb, tou
                 'n_ev' : int(nw * ev_penetration * (1-home_vs_work)),
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
ev_data2 = {'home': {'type' : 'tou',     #dumb, tou
                 'n_ev' : int(nh * ev_penetration * home_vs_work),
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep : 0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'tou',     #dumb, tou
                 'n_ev' : int(nw * ev_penetration * (1-home_vs_work)),
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2,
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
                          
print('Simulating grids with %d EVs home charging, %d EVs work charging' 
      %(int(nh * ev_penetration * home_vs_work),int(nw * ev_penetration * (1-home_vs_work))))

grid1 = evmodel.Grid(ev_data1, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid2 = evmodel.Grid(ev_data2, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid1.do_days()
grid2.do_days()
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_tot_load(ax=ax1, title='%s Tot Load - All Days' %ss)
grid2.plot_tot_load(ax=ax2, title='%s Tot Load - If Needed' %ss)
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_evload(ax=ax1, title='%s EV Load - All Days' %ss)
grid2.plot_evload(ax=ax2, title='%s EV Load - If Needed' %ss)
f, ax1 = plt.subplots(1,1)
grid1.do_dist_hist(ax=ax1)

#%% Load Data - real sims - Verfeil
import model as evmodel
#print('loading data')
#load_by_comm, load_profiles, SS = evmodel.load_conso_ss_data()
#hh, hw = evmodel.load_hist_data()
##%
print('preprocessing data')
ss = 'VERFEIL'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=1), 15)[:-1]

load.plot(kind='area')
plt.legend(loc=1)
# Computing pdf of home and work
print('computing pdfs')
hh_ss = evmodel.extract_hist(hh, SS.Communes[ss]).sum()
nh = hh_ss.sum()
pdf_hh = hh_ss.cumsum()/nh

hw_ss = evmodel.extract_hist(hw, SS.Communes[ss]).sum()
nw = hw_ss.sum()
pdf_hw = hw_ss.cumsum()/nw

#%% Run Simulation for a given SS, Dumb charging
import model as evmodel

# % work and home and penetration
voitures_actifs = 0.5
voitures_non_compta = 2
ev_penetration = 1
home_vs_work = 1 # 1 only home, 0 only work
nevh = int(nh * ev_penetration * home_vs_work * voitures_actifs * voitures_non_compta)
nevw = int(nw * ev_penetration * (1-home_vs_work) * voitures_actifs * voitures_non_compta)
ev_data1 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' :  3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
ev_data2 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep : 0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2,
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
                          
print('Simulating grids with %d EVs home charging, %d EVs work charging' 
      %(nevh,nevw))

grid1 = evmodel.Grid(ev_data1, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid2 = evmodel.Grid(ev_data2, 
                     ndays=ndays, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)
grid1.do_days()
grid2.do_days()
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_tot_load(ax=ax1, title='%s Tot Load - All Days' %ss)
grid2.plot_tot_load(ax=ax2, title='%s Tot Load - If Needed' %ss)
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_evload(ax=ax1, title='%s EV Load - All Days' %ss)
grid2.plot_evload(ax=ax2, title='%s EV Load - If Needed' %ss)
f, ax1 = plt.subplots(1,1)
grid1.do_dist_hist(ax=ax1)

#%% histos of charging sessions:
ss = 'VANVES'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=1), 15)[:-1]


# Computing pdf of home and work
print('computing pdfs')
hh_ss = evmodel.extract_hist(hh, SS.Communes[ss]).sum()
nh = hh_ss.sum()
pdf_hh = hh_ss.cumsum()/nh

hw_ss = evmodel.extract_hist(hw, SS.Communes[ss]).sum()
nw = hw_ss.sum()
pdf_hw = hw_ss.cumsum()/nw

voitures_actifs = 0.27
voitures_non_compta = 2
ev_penetration = 1
home_vs_work = 1 # 1 only home, 0 only work
nevh = int(nh * ev_penetration * home_vs_work * voitures_actifs * voitures_non_compta)
nevw = int(nw * ev_penetration * (1-home_vs_work) * voitures_actifs * voitures_non_compta)
ev_data1 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' :  3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
                                  
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}

gridvanv = evmodel.Grid(ev_data1, 
                     ndays=ndays-1, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)

print('preprocessing data')
ss = 'VERFEIL'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=1), 15)[:-1]

#load.plot(kind='area')
#plt.legend(loc=1)
# Computing pdf of home and work
print('computing pdfs')
hh_ss = evmodel.extract_hist(hh, SS.Communes[ss]).sum()
nh = hh_ss.sum()
pdf_hh = hh_ss.cumsum()/nh

hw_ss = evmodel.extract_hist(hw, SS.Communes[ss]).sum()
nw = hw_ss.sum()
pdf_hw = hw_ss.cumsum()/nw

voitures_actifs = 0.5
voitures_non_compta = 2
ev_penetration = 1
home_vs_work = 1 # 1 only home, 0 only work
nevh = int(nh * ev_penetration * home_vs_work * voitures_actifs * voitures_non_compta)
nevw = int(nw * ev_penetration * (1-home_vs_work) * voitures_actifs * voitures_non_compta)
ev_data2 = {'home': {'type' : 'dumb',
                 'n_ev' : nevh,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hh, 
#                          'charging_power' :  3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
                 'n_ev' : nevw,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'batt_size' : 24, #,
#                         'tou_ini' : 1, 
#                         'tou_end' : 8,
                          'pdf_dist' : pdf_hw, 
#                          'charging_power' : 3.6, 
#                          'charging_eff' : 0.95, 
#                          'driving_eff' : 0.2
#                          'range_anx_factor' : 1.5,
#                          'extra_trip_proba' : 0.2
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}

gridverf = evmodel.Grid(ev_data2, 
                     ndays=ndays-1, step=step, 
                     load = np.asarray(load.sum(axis=1)), 
                     ss_pmax=SS.Pmax[ss], 
                     name=ss)

gridvanv.do_days()
gridverf.do_days()

f, axs = plt.subplots(1,2)
grids = [gridvanv, gridverf]
ds = [[]] * 2
dhs = [[]] * 2
hs = [[]] * 2
bins = [i for i in range(9)]
gtypes = ['Urban', 'Rural']
#for i in range(3):
#    ds[i] = np.asarray([ev.ch_status.sum() for types in grids[i].evs
#                                for ev in grids[i].evs[types]])
#    hs[i] = np.histogram(ds[i], bins)
#    axs[i].bar(hs[i][1][:-1], hs[i][0]/sum(hs[i][0]))
#    axs[i].set_xlabel('# de sessions de recharge')    
#    axs[i].set_ylabel('Densité')
#    axs[i].set_title('Batterie %d kWh' %batts[i])
#    axs[i].set_xlim([0,8])
#    axs[i].set_xticks([i+1 for i in range(7)])
#    axs[i].set_ylim([0,0.700])
for i in range(2):
    ds[i] = np.asarray([ev.ch_status.sum() for types in grids[i].evs
                                for ev in grids[i].evs[types]])
    hs[i] = np.histogram(ds[i], bins)
    dhs[i] = hs[i][0]/sum(hs[i][0])
    axs[i].bar(hs[i][1][:-1], dhs[i])
    axs[i].set_xlabel('Charging sessions')    
    axs[i].set_ylabel('Density')
    axs[i].set_title(gtypes[i])
    axs[i].set_xlim([0,8])
    axs[i].set_xticks([i+1 for i in range(7)])
    axs[i].set_ylim([0,0.700])
#%
ff, ax = plt.subplots(1,1)
w = 0.4
delta = [-w/2, w/2]
for i in range(2):
    ax.bar(hs[i][1][:-1]+ delta[i], dhs[i], width=w, label=gtypes[i])
ax.legend()
ax.set_xlim([0,8])
ax.set_xticks([i+1 for i in range(7)])
ax.set_xlabel('Charging sessions')    
ax.set_ylabel('Density')