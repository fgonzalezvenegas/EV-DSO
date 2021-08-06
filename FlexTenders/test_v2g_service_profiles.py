# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:28:02 2020

@author: U546416
"""

import EVmodel
import util
import matplotlib.pyplot as plt

#%% 1/2 service
util.self_reload(EVmodel)
import EVmodel

grid = EVmodel.Grid(ndays=3, step=1)
grid.add_evs('', 1, 'dumb', alpha=1000, up_dn_flex=True, target_soc = 1)

grid.evs[''][0].dist_wd = 20
grid.do_days()
grid.plot_flex_pot()

ev = grid.get_ev()       

plt.subplots()
plt.plot(ev.up_flex_kw_meantraj, label='Up_mt')
plt.plot(ev.up_flex_kw_immediate, label='Up_imm')
plt.plot(ev.up_flex_kw_delayed, label='Up_del')
plt.plot(ev.dn_flex_kw_meantraj, label='dn_mt')
plt.plot(ev.dn_flex_kw_immediate, label='dn_imm')
plt.plot(ev.dn_flex_kw_delayed, label='dn_del')
plt.legend()

#%% 3h Service
util.self_reload(EVmodel)
import EVmodel

grid = EVmodel.Grid(ndays=3, step=1)
grid.add_evs('', 1, 'dumb', alpha=1000, up_dn_flex=True, target_soc = 1, flex_time=180)

grid.evs[''][0].dist_wd = 20
grid.do_days()
grid.plot_flex_pot()

ev = grid.get_ev()       

plt.subplots()
plt.plot(ev.up_flex_kw_meantraj, label='Up_mt')
plt.plot(ev.up_flex_kw_immediate, label='Up_imm')
plt.plot(ev.up_flex_kw_delayed, label='Up_del')
plt.plot(ev.dn_flex_kw_meantraj, label='dn_mt')
plt.plot(ev.dn_flex_kw_immediate, label='dn_imm')
plt.plot(ev.dn_flex_kw_delayed, label='dn_del')
plt.legend()