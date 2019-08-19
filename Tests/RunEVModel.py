# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:39:37 2019

Runs EV model over a week
Does some graphs in english

@author: U546416
"""
import numpy as np
from matplotlib import pyplot as plt
import EVmodel as evmodel

ndays = 8
step = 15

bins_dist = np.linspace(0, 100, num=51)
dist_function = np.sin(bins_dist[:-1]/ 100 * np.pi * 2) + 1
dist_function[10:15] = [0, 0, 0 , 0 , 0]
pdfunc = (dist_function/sum(dist_function)).cumsum()
dlabels =['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

#%% Computes 4 combinations of dumb charging 
ev_data2 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday'}}}
ev_data1 = {'EV Load': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days'}}}
ev_data4 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 7.2}}}
ev_data3 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type': 'all_days',
                         'charging_power' : 7.2}}}
grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step)
grid1.do_days()
grid2 = evmodel.Grid(ev_data2, ndays=ndays, step=step)
grid2.do_days()
grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step)
grid3.do_days()
grid4 = evmodel.Grid(ev_data4, ndays=ndays, step=step)
grid4.do_days()
#f, ([ax1, ax2],  [ax3, ax4]) = plt.subplots(2,2)
f, (ax1, ax2) = plt.subplots(1, 2)
ylim = 3.7*1000
grid1.plot_evload(ax=ax1, ylim=ylim, title='Dumb, 3.6kW, All days')
grid2.plot_evload(ax=ax2, ylim=ylim, title='Dumb, 3.6kW, If needed')
#grid4.plot_evload(ax=ax3, ylim=ylim, title='Dumb, 7.2kW, All days')
#grid3.plot_evload(ax=ax4, ylim=ylim, title='Dumb, 7.2kW, If needed')
f, (ax1, ax2) = plt.subplots(1, 2)
grid1.plot_flex_pot(ax=ax1,title='Dumb, 3.6kW, All days')
grid2.plot_flex_pot(ax=ax2, title='Dumb, 3.6kW, If needed')
# Plot up and dn flex
f, (ax1) = plt.subplots(1, 1)
ax1.plot(grid1.ev_up_flex['Total'] - grid1.ev_mean_flex['Total'], label='Systematic, Up flexibility', linestyle='-', color='b')
ax1.plot(grid1.ev_dn_flex['Total'] - grid1.ev_mean_flex['Total'], label='Systematic, Down flexibility', linestyle='-', color='c')
ax1.plot(grid2.ev_up_flex['Total'] - grid2.ev_mean_flex['Total'], label='Non-systematic, Up flexibility', linestyle='-', color='r')
ax1.plot(grid2.ev_dn_flex['Total'] - grid2.ev_mean_flex['Total'], label='Non-systematic, Down flexibility', linestyle='-', color='orange')
ax1.legend(loc=2)
ax1.set_ylabel('Flexible storage [kWh]')
ax1.set_xlabel('Days')
ax1.set_xlim([0, 24*4*7.5])
ax1.set_ylim([-25000, 20000])
ax1.set_xticks(np.arange(ndays) * 24 * 4)
ax1.set_xticklabels(dlabels)
ax1.grid(axis='x')
f, (ax11) = plt.subplots(1, 1)
ax11.plot(grid1.ev_up_flex['Total'] - grid1.ev_dn_flex['Total'], label='Systematic, Total flexibility')
ax11.plot(grid2.ev_up_flex['Total'] - grid2.ev_dn_flex['Total'], label='Non-systematic, Total flexibility')
ax11.legend(loc=2)
ax11.set_ylabel('Flexible storage [kWh]')
ax11.set_xlabel('Days')
ax11.set_xlim([0, 24*4*7.5])
ax11.set_xticks(np.arange(ndays) * 24 * 4)
ax11.set_ylim([0, 30000])
ax11.set_xticklabels(dlabels)
ax11.grid(axis='x')
#%% Plot average up and dn flex
nevs = np.zeros(ndays)
for ev in grid2.get_evs():
    nevs+= ev.ch_status
nnevs = np.zeros(int(ndays * 24 * 60/ step))+0.001
delta = int(12 * 60/step)
for d in np.arange(ndays):
    for i in np.arange(int(24 * 60/ step)):
        if int(d * 24 * 60/ step) + i + delta >= len(nnevs):
            break
        nnevs[int(d * 24 * 60/ step) + i + delta] = nevs[d]
f, ax2 = plt.subplots(1)
ax2.plot((grid1.ev_up_flex['Total'] - grid1.ev_mean_flex['Total'])/1000, label='Systematic, Up flexibility', linestyle='-', color='b')
ax2.plot((grid1.ev_dn_flex['Total'] - grid1.ev_mean_flex['Total'])/1000, label='Systematic, Down flexibility', linestyle='-', color='c')
ax2.plot((grid2.ev_up_flex['Total'] - grid2.ev_mean_flex['Total'])/nnevs, label='Non-systematic, Up flexibility', linestyle='-', color='r')
ax2.plot((grid2.ev_dn_flex['Total'] - grid2.ev_mean_flex['Total'])/nnevs, label='Non-systematic, Down flexibility', linestyle='-', color='orange')
ax2.legend(loc=2)
ax2.set_ylabel('Average flexible storage per EV [kWh]')
ax2.set_xlabel('Days')
ax2.set_xlim([0, 24*4*7.5])
ax2.set_ylim([-25, 20])
ax2.set_xticks(np.arange(ndays) * 24 * 4)
ax2.set_xticklabels(dlabels)
ax2.grid(axis='x')
f, (ax22) = plt.subplots(1, 1)
ax22.plot((grid1.ev_up_flex['Total'] - grid1.ev_dn_flex['Total'])/1000, label='Systematic, Total flexibility')
ax22.plot((grid2.ev_up_flex['Total'] - grid2.ev_dn_flex['Total'])/nnevs, label='Non-systematic, Total flexibility')
ax22.legend(loc=2)
ax22.set_ylabel('Average flexible storage per EV [kWh]')
ax22.set_xlabel('Days')
ax22.set_xlim([0, 24*4*7.5])
ax22.set_xticks(np.arange(ndays) * 24 * 4)
ax22.set_ylim([0, 30])
ax22.set_xticklabels(dlabels)
ax22.grid(axis='x')

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
ev_data5 = {'Dumb_Modulated': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 3.6}}}
ev_data6 = {'Dumb_Modulated': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type': 'all_days',
                         'charging_power' : 3.6}}}
ev_data7 = {'ToU_Modulated': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed_sunday',
                         'charging_power' : 3.6,
                         'tou_ini' : 23,
                         'tou_end' : 7}}}
ev_data8 = {'ToU_Modulated': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type': 'all_days',
                         'charging_power' : 3.6,
                         'tou_ini' : 23,
                         'tou_end' : 7}}}


grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step)
grid3.do_days()
grid4 = evmodel.Grid(ev_data4, ndays=ndays, step=step)
grid4.do_days()
grid5 = evmodel.Grid(ev_data5, ndays=ndays, step=step)
grid5.do_days()
grid6 = evmodel.Grid(ev_data6, ndays=ndays, step=step)
grid6.do_days()
grid7 = evmodel.Grid(ev_data7, ndays=ndays, step=step)
grid7.do_days()
grid8 = evmodel.Grid(ev_data8, ndays=ndays, step=step)
grid8.do_days()
#
#f, ([ax1, ax2],  [ax3, ax4]) = plt.subplots(2,2)
#ylim = 3.6*1000
#grid2.plot_evload(ax=ax1, ylim=ylim, title='Dumb, 3.6kW, All days')
#grid1.plot_evload(ax=ax3, ylim=ylim, title='Dumb, 3.6kW, If needed')
#grid4.plot_evload(ax=ax2, ylim=ylim, title='ToU, 3.6kW, All days')
#grid3.plot_evload(ax=ax4, ylim=ylim, title='ToU, 3.6kW, If needed')
#
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

f, (ax1, ax2) = plt.subplots(1, 2)
ylim = 3.7*1000
grid4.plot_evload(ax=ax1, ylim=ylim, title='ToU, 3.6kW, All days')
grid3.plot_evload(ax=ax2, ylim=ylim, title='ToU, 3.6kW, If needed')

#%%
evs = grid4.get_evs()
for ev in evs:
    ch = ev.charging.reshape([8, 24*4])
    ch_out = ch[:,8*4:22*4].sum()
    if ch_out > 0:
        break
#%% Computes 4 combinations of dumb charging different battery size
ev_data1 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 24}}}
ev_data2 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 40}}}
ev_data3 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 60}}}
ev_data4 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'charging_power' : 7.2,
                         'batt_size' : 24}}}
ev_data5 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'charging_power' : 7.2,
                         'batt_size' : 40}}}
ev_data6 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
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
f, ([ax1, ax2, ax3],  [ax4, ax5, ax6]) = plt.subplots(2,3)
ylim = 7.2*1000
grid1.plot_evload(ax=ax1, ylim=ylim, title='Dumb, 3.6kW, Bat=24 kWh')
grid2.plot_evload(ax=ax2, ylim=ylim, title='Dumb, 3.6kW, Bat=40 kWh')
grid3.plot_evload(ax=ax3, ylim=ylim, title='Dumb, 3.6kW, Bat=60 kWh')
grid4.plot_evload(ax=ax4, ylim=ylim, title='Dumb, 7.2kW, Bat=24 kWh')
grid5.plot_evload(ax=ax5, ylim=ylim, title='Dumb, 7.2kW, Bat=40 kWh')
grid6.plot_evload(ax=ax6, ylim=ylim, title='Dumb, 7.2kW, Bat=60 kWh')

f, ([ax1, ax2, ax3],  [ax4, ax5, ax6]) = plt.subplots(2,3)
grid1.do_ncharging_hist(ax=ax1, title='#Charging Sessions, Bat=24 kWh, 3.6kW')
grid2.do_ncharging_hist(ax=ax2, title='#Charging Sessions, Bat=40 kWh, 3.6kW')
grid3.do_ncharging_hist(ax=ax3, title='#Charging Sessions, Bat=60 kWh, 3.6kW')
grid4.do_ncharging_hist(ax=ax4, title='#Charging Sessions, Bat=24 kWh, 7.2kW')
grid5.do_ncharging_hist(ax=ax5, title='#Charging Sessions, Bat=40 kWh, 7.2kW')
grid6.do_ncharging_hist(ax=ax6, title='#Charging Sessions, Bat=60 kWh, 7.2kW')
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

#%% Peak / off Peak 1 off peak 2
ev_data1 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24}},
            'Off Peak 1': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24,
                         'tou_ini' : 23,
                         'tou_end' : 6}},
            'Off Peak 2': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'batt_size' : 24,
                         'tou_ini' : 1,
                         'tou_end' : 8}}}
ev_data2 = {'dumb': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 24}},
            'Off Peak 1': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 24,
                         'tou_ini' : 23,
                         'tou_end' : 6}},
            'Off Peak 2': {'type' : 'dumb',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'if_needed',
                         'batt_size' : 24,
                         'tou_ini' : 1,
                         'tou_end' : 8}}}            
grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step)
grid1.do_days()
grid2 = evmodel.Grid(ev_data2, ndays=ndays, step=step)
grid2.do_days()
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_evload(ax=ax1, ylim=ylim*1.5, title='Different EV charging, 3.6kW, Bat=24 kWh, all days')
grid2.plot_evload(ax=ax2, ylim=ylim*1.5, title='Different EV charging, 3.6kW, Bat=24 kWh, If needed')

#%% Modulated

ev_data00 = {'dumb': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'pmin_charger' : 1,
                         'batt_size' : 24}}}
ev_data10 = {'dumb': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'pmin_charger' : 0.6,
                         'batt_size' : 24}}}
ev_data20 = {'dumb': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'pmin_charger' : 0,
                         'batt_size' : 24}}}                           
                         
ev_data0 = {'dumb': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'pmin_charger' : 1,
                         'batt_size' : 24,
                         'tou_ini' : 23,
                         'tou_end' : 6}}}
ev_data1 = {'dumb': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'pmin_charger' : 0.6,
                         'batt_size' : 24,
                         'tou_ini' : 23,
                         'tou_end' : 6}}}
ev_data2 = {'dumb': {'type' : 'mod',
                 'n_ev' : 1000,
                 'other': {
                         'charging_type' : 'all_days',
                         'pmin_charger' : 0,
                         'batt_size' : 24,
                         'tou_ini' : 23,
                         'tou_end' : 6}}}     

grid00 = evmodel.Grid(ev_data00, ndays=ndays, step=step)
grid00.do_days()
grid10 = evmodel.Grid(ev_data10, ndays=ndays, step=step)
grid10.do_days()
grid20 = evmodel.Grid(ev_data20, ndays=ndays, step=step)
grid20.do_days()
grid0 = evmodel.Grid(ev_data0, ndays=ndays, step=step)
grid0.do_days()
grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step)
grid1.do_days()
grid2 = evmodel.Grid(ev_data2, ndays=ndays, step=step)
grid2.do_days()
f, ([ax00, ax10, ax20], [ax0, ax1, ax2]) = plt.subplots(2,3)
grid00.plot_evload(ax=ax00, ylim=3.7e3, title='Not Modulated')
grid10.plot_evload(ax=ax10, ylim=3.7e3, title='Modulated, Pmin 0.6')
grid20.plot_evload(ax=ax20, ylim=3.7e3, title='Modulated, Pmin 0.0')
grid0.plot_evload(ax=ax0, ylim=3.7e3, title='Not Modulated')
grid1.plot_evload(ax=ax1, ylim=3.7e3, title='Modulated, Pmin 0.6')
grid2.plot_evload(ax=ax2, ylim=3.7e3, title='Modulated, Pmin 0.0')


#%% Load Data - real sims
import model as evmodel
print('loading data')
load_by_comm, load_profiles, SS = evmodel.load_conso_ss_data()
hh, hw = evmodel.load_hist_data()

#%% Run Simulation for a given SS, Dumb charging
import model as evmodel
print('preprocessing data')
ss = 'VAUDREUIL'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=ndays-7), 15)[:-1]


# Computing pdf of home and work
print('computing pdfs')
hh_ss = evmodel.extract_hist(hh, SS.Communes[ss]).sum()
nh = hh_ss.sum()
pdf_hh = hh_ss.cumsum()/nh

hw_ss = evmodel.extract_hist(hw, SS.Communes[ss]).sum()
nw = hw_ss.sum()
pdf_hw = hw_ss.cumsum()/nw

# % work and home and penetration
ev_penetration = 1
home_vs_work = 1 # 1 only home, 0 only work

ev_data1 = {'home': {'type' : 'dumb',
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
#                          'extra_trip_proba' : 0.2,
#                          'tou_ini' : 11,
#                          'tou_end' : 7,                          
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
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
#                          'extra_trip_proba' : 0.2,
#                          'tou_ini' : 11,
#                          'tou_end' : 7,   
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep  0, 
                                                      'mu_arr' : 9, 
                                                      'mu_dep' : 17, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1}
                         }}}
ev_data2 = {'home': {'type' : 'dumb',
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
#                          'extra_trip_proba' : 0.2,
#                          'tou_ini' : 11,
#                          'tou_end' : 7,   
                          'arrival_departure_data_wd' : {#pdf_arr : 0, 
                                                      #pdf_dep : 0, 
                                                      'mu_arr' : 18, 
                                                      'mu_dep' : 8, 
                                                      'std_arr' : 1, 
                                                      'std_dep' : 1/3}
                          }},
            'work': {'type' : 'dumb',
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
#                          'tou_ini' : 11,
#                          'tou_end' : 7,   
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
grid1.plot_tot_load(ax=ax1, title='%s Total Load - All Days' %ss)
grid2.plot_tot_load(ax=ax2, title='%s Total Load - If Needed' %ss)
f, (ax1, ax2) = plt.subplots(1,2)
grid1.plot_evload(ax=ax1, title='%s EV Load - All Days' %ss)
grid2.plot_evload(ax=ax2, title='%s EV Load - If Needed' %ss)
f, ax1 = plt.subplots(1,1)
grid1.do_dist_hist(ax=ax1)

#%% Run Simulation for a given SS - ToU Charging
import model as evmodel

# % work and home and penetration
ev_penetration = 1
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


#%% Load Data - real sims
import model as evmodel
#print('loading data')
#load_by_comm, load_profiles, SS = evmodel.load_conso_ss_data()
#hh, hw = evmodel.load_hist_data()
##%
print('preprocessing data')
ss = 'VANVES'
load = evmodel.interpolate(
            evmodel.get_max_load_week(
                    evmodel.compute_load_from_ss(load_by_comm, load_profiles, SS, ss), extradays=ndays-7), 15)[:-1]


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
ev_penetration = 1
home_vs_work = 1 # 1 only home, 0 only work

ev_data1 = {'home': {'type' : 'dumb',
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
            'work': {'type' : 'dumb',
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
ev_data2 = {'home': {'type' : 'dumb',
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
            'work': {'type' : 'dumb',
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