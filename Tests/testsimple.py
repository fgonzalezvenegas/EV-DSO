# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:32:55 2019
testestest
@author: U546416
"""

import numpy as np
from matplotlib import pyplot as plt
import EVmodel as evmodel

ndays = 8
step = 30

bins_dist = np.linspace(0, 100, num=51)
pdf_dist = np.zeros(50)
for i in range(40):
    pdf_dist[-(i+1)] = 1
print(pdf_dist)
nev = 1
cp = 7.2

ev_data1 = {'dumb': {'type' : 'dumb',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'all_days',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 0}}}
ev_data2 = {'tou': {'type' : 'dumb',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'all_days',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 6}}}
ev_data3 = {'modulated': {'type' : 'mod',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'all_days',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 0}}}
ev_data4 = {'modulated': {'type' : 'mod',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'all_days',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 6,
                         'tou_we' : True}}}


grid1 = evmodel.Grid(ev_data1, ndays=ndays, step=step, name='dumb')
grid2 = evmodel.Grid(ev_data2, ndays=ndays, step=step, name='dumb ToU')
grid3 = evmodel.Grid(ev_data3, ndays=ndays, step=step, name='mod')
grid4 = evmodel.Grid(ev_data4, ndays=ndays, step=step, name='mod ToU')

grids = [grid1, grid2, grid3, grid4]
#set driving distance at 150
evs = [grid.get_evs()[0] for grid in grids]
for ev in evs:
    ev.dist_wd = 75
    ev.dist_we = 75
for grid in grids:
    grid.do_days()

f1, axs1 = plt.subplots(2,2)
f2, axs2 = plt.subplots(2,2)
ylim = nev*cp*1.1
for i in range(4):
    grids[i].plot_evload(ax=axs1[i//2][i%2], ylim=ylim, title=grids[i].name)
    grids[i].plot_flex_pot(ax=axs2[i//2][i%2], title=grids[i].name)

#%%
import EVmodel as evmodel
ev_data1_2 = {'dumb': {'type' : 'dumb',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'if_needed',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 0}}}
ev_data2_2 = {'tou': {'type' : 'dumb',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'if_needed',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 6}}}
ev_data3_2 = {'modulated': {'type' : 'mod',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'if_needed',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 0}}}
ev_data4_2 = {'modulated': {'type' : 'mod',
                 'n_ev' : nev,
                 'other': {
                         'cdf_dist_wd' : pdf_dist,
                         'charging_type' : 'if_needed',
                         'charging_power' : cp,
                         'tou_ini' : 0,
                         'tou_end': 6,
                         'tou_we' : True}}}


grid1_2 = evmodel.Grid(ev_data1_2, ndays=ndays, step=step, name='dumb, if needed')
grid2_2 = evmodel.Grid(ev_data2_2, ndays=ndays, step=step, name='tou, if needed')
grid3_2 = evmodel.Grid(ev_data3_2, ndays=ndays, step=step, name='modulated, if needed')
grid4_2 = evmodel.Grid(ev_data4_2, ndays=ndays, step=step, name='mod weekend, if needed')
grids_2 = [grid1_2, grid2_2, grid3_2, grid4_2]
for grid in grids_2:
    grid.do_days()

f1, axs1 = plt.subplots(2,2)
f2, axs2 = plt.subplots(2,2)
ylim = nev*cp*1.1
for i in range(4):
    grids_2[i].plot_evload(ax=axs1[i//2][i%2], ylim=ylim, title=grids[i].name)
    grids_2[i].plot_flex_pot(ax=axs2[i//2][i%2], title=grids[i].name)

evs = [grid.get_ev() for grid in grids_2]
#%% plot lognormal

bs = np.arange(0,100,1)
s = 0.736
m = 2.75
lognormale = 1/(np.sqrt(2*np.pi) * bs * s ) * np.exp(-((np.log(bs) - m)**2)/(2*s*s))
d = np.random.lognormal(mean=m, sigma=s, size=1000)
h = np.histogram(d, bins = bs)
h = h[0]/sum(h[0])

print((h*(bs[1:]-0.5)).sum())

f, ax = plt.subplots()
ax.plot(bs[1:], lognormale[1:], color='red')
ax.bar(bs[1:], h)
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Densité')
ax.set_xlim([0,100])

#%%
mln = [0]*100
for i in range(100):
    d = np.random.lognormal(mean=m, sigma=s, size=10000)
    h = np.histogram(d, bins = bs)
    h = h[0]/sum(h[0])
    mln[i] = (h*(bs[1:]-0.5)).sum()
print(np.mean(mln), np.max(mln), np.min(mln))

##%% plot and transform
#ff, (ax1)  = plt.subplots(1,1)
#bs = bins_dist[1:]
#xlim  = [0,100]
#ylim = [0,0.2]
#ax1.plot(bs, d_vanves, color='blue', label='Parisian Suburb')
#ax1.plot(bs, d_verf, color='red', label='Rural South')
#ax1.legend()
#ax1.set_xlim(xlim)
#ax1.set_ylim(ylim)
#ax1.set_ylabel('Density')
#ax1.set_xlabel('Distance [km]')
