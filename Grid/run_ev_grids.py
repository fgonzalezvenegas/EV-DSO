# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:57:22 2020
Run power flow on PandaPower using EV profiles
@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe
import util
import util_grid as ug
import pandapower as pp
import pandapower.topology as ppt
import pandapower.plotting as ppp
import pandapower.control as ppc
import pandapower.timeseries as ppts

import time 
import datetime as dt

from ppgrid import *
from grid import *
import VoltVar


# importing base grid


# load predefined grid
folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
print('\t loading grids from folder {}'.format(folder_grids))
#net_res = pp.from_json(folder_grids + 'res_grid.json')

# number of LV trafos per IRIS
folder_lv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
lv_iris = pd.read_csv(folder_lv+'Nb_BT_IRIS2016.csv',
                      engine='python', index_col=0)

# Base profiles
print('\tloading base load profiles')
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles_thesis\\'
profiles_load = pd.read_csv(folder_profiles  + r'profiles_iris.csv',
                       engine='python', index_col=0)
# PV profiles
print('\tloading PV load profiles')
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles_thesis\\'
pv_farm = pd.read_csv(folder_profiles  + r'PV_2018_localtime.csv',
                       engine='python', index_col=0)
pv_rooftop = pd.read_csv(folder_profiles  + r'PVrooftop_2018_localtime.csv',
                       engine='python', index_col=0)
# interpolating, setting it as 30min time series
#pv_farm.index = pd.to_datetime(pv_farm.index)
#pv_rooftop.index = pv_farm.index
#pv_farm = pv_farm.resample('30T').asfreq().interpolate().squeeze()
## adding one time stamp to the end
#dt = pv_farm.index[-1]-pv_farm.index[-2]
#t = pv_farm.index[-1]+dt
#pv_farm[t] = 0

pv_profiles = pd.concat([pv_farm, pv_rooftop], axis=1)
pv_profiles.columns = ['farm', 'rooftop']
#pv_profiles.rooftop = pv_profiles.rooftop.interpolate()

# Load IRIS polygons
#print('Loading IRIS polygons')
#folder_iris = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
#file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
#iris_poly = pd.read_csv(folder_iris+file_iris,
#                        engine='python', index_col=0)
## Plot supply zone
#iris = net.load.zone.astype(int).unique()
#polygons = util.do_polygons(iris_poly.loc[iris], plot=False)
#cmap = plt.get_cmap('plasma')
#nb_bt_b = net.load[net.load.type_load=='Base'].groupby('zone').type_load.count()[iris] # Number of supplied LV trafos per IRIS by SS
#nb_bt=lv_iris.Nb_BT[iris] # Total number of trafos per IRIS
#supply = 1-((nb_bt-nb_bt_b)/nb_bt) # Ratio of supply
#colors=cmap(supply)
#
#ax=util.plot_polygons(util.list_polygons(polygons,(supply).index), color=colors, edgecolor='darkgrey', linestyle='--')
#plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
#plt.plot(net.bus_geodata.x[0], net.bus_geodata.y[0], 'o', color='red')
#
#tranches = np.linspace(0,1,6)
#labels = ['{}%'.format(int(t*100)) for t in tranches]
#colorslab = cmap(tranches)
#util.do_labels(labels, colorslab, ax)
#
#plt.title('Supply of load from substation')
# EV profiles
#%%
folder_evs = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles_thesis\\'
sets = [#{'name': 'ev',
#         'gridn': 'base_grid.json',
#         'ev_profs': 'EV_p0.5_w0.3.csv'},
        #{'name': 'evtou',
#         'gridn': 'base_grid.json',
#         'ev_profs': 'EV_p0.5_w0.3_tou.csv'},
#        {'name': 'ev_pvcontinue',
#         'gridn': 'res_grid_equi_2050.json',
#         'ev_profs': 'EV_p0.5_w0.3.csv'},
#        {'name': 'evtou_pvcontinue',
#         'gridn': 'res_grid_equi_2050.json',
#         'ev_profs': 'EV_p0.5_w0.3_tou.csv'},
         {'name': 'evtou_pvrooftop',
         'gridn': 'res_grid_roof_2.json',
         'ev_profs': 'EV_p1_w0.5_tou.csv'},
#         {'name': 'base_noEV',
#          'gridn': 'base_grid.json',
#          'ev_profs': None},
         ]

for keys in sets:
    print('Doing case '  + keys['name'])
    # Reading grid
    net = pp.from_json(folder_grids + keys['gridn'])
    ev_prof_fn = keys['ev_profs']
       
    print('\t loading ev profile {}'.format(ev_prof_fn))
    if not (ev_prof_fn is None):
        ev_prof = pd.read_csv(folder_evs + ev_prof_fn,
                              engine='python', index_col=0)
    # Removing EV loads if already existing in grid
    net.load=net.load[~(net.load.type_load=='EV')]
    # merge Home & Work profiles and set up profiles and loads for pandapower
    if not (ev_prof_fn is None):
        homecols = [c for c in ev_prof.columns if 'Home' in c]
        workcols = [c for c in ev_prof.columns if 'Work' in c]
        idhome = [c.replace('Home_','EV_') for c in homecols]
        idwork = [c.replace('Work_','EV_') for c in workcols]
        ev_prof_h = ev_prof[homecols]
        ev_prof_w = ev_prof[workcols]
        ev_prof_h.columns = idhome
        ev_prof_w.columns = idwork
        ev_prof = ev_prof_h + ev_prof_w
        ev_prof_pu = ev_prof / ev_prof.max(axis=0) # profile in pu, max=1
        load_ev_lv = ev_prof.max(axis=0)  # Max load of EV per LV trafo per IRIS
        load_ev_lv.index = [int(c.replace('EV_','')) for c in load_ev_lv.index]
        load_ev_lv = load_ev_lv / lv_iris.Nb_BT[load_ev_lv.index]  
    
        # transform EV profile to yearly
        ev_prof_pu = ug.yearly_profile(ev_prof_pu, step=30, day_ini=0)
        ev_prof_pu.index = profiles_load.index
    
        # Adding pandapower loads
        print('Adding EV loads')
        idxs = []
        zones = []
        for j, t in net.load[net.load.type_load == 'Base'].iterrows():
            b = t.bus
            i = int(t.zone)
            zones.append(str(i))
            idxs.append(pp.create_load(net, bus=b,  p_mw=load_ev_lv[i], q_mvar=0, name='EV_' + str(i)+'_'+str(j)))
        net.load.type_load[idxs] = 'EV' 
        net.load.zone[idxs] = zones
    
    # Adding profiles
    # Adding profile database
    # Creating profiler
    profiler = Profiler(net=net, profiles_db=profiles_load)
    if not ev_prof_fn is None:
        # Adding database of EV profiles
        profiler.add_profile_db(ev_prof_pu)
    for n in net.load.zone.unique():
        if not ((n is None) or not (n==n)):
            # Assigning index of base load profiles
            profiler.add_profile_idx(element='load', 
                                     idx=net.load[(net.load.zone==n) & (net.load.type_load=='Base')].index, 
                                     variable='scaling', 
                                     profile=str(int(n)))
            if not ev_prof_fn is None:
                # Assigning index of EV load profiles
                profiler.add_profile_idx(element='load',
                                         idx=net.load[(net.load.zone==n) & (net.load.type_load=='EV')].index,
                                         variable='scaling',
                                         profile='EV_' + str(int(n)))
    # Adding PV profiles if any:
    if net.sgen.shape[0]>0:
        pv_profiles.index = ev_prof_pu.index
        profiler.add_profile_db(pv_profiles)
        idxfarms = net.sgen[net.sgen.type.isin(['Farm_PV'])].index
        idxrooftop = net.sgen[net.sgen.type.isin(['RES_PV', 'Comm_PV'])].index
        profiler.add_profile_idx(element='sgen',
                                 idx=idxfarms,
                                 variable='scaling',
                                 profile='farm')
        profiler.add_profile_idx(element='sgen',
                                 idx=idxrooftop,
                                 variable='scaling',
                                 profile='rooftop')
        # Adding voltvar controller
        # Adding VoltVar controllers to Farms PV
        voltvars = []
        farms = net.sgen[net.sgen.type=='Farm_PV'].index
        for i in farms:
            voltvars.append(VoltVar.VoltVar(net, i, level=1, tolerance=0.15))

    
    output_folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\\'
    # Setting iterator
    time_steps=profiler.profiles_db.index   
    iterator = Iterator(net=net, profiler=profiler,
                     line_loading=True, feeder_pq=False)
    of = output_folder + keys['name']
    iterator.iterate(time_steps=time_steps, save=True, 
                     outputfolder=of, ultraverbose=False)
#    
#    # Net load at transformer - May and February
#    f, ax = plt.subplots()
#    ax.plot(np.arange(0,2*24*7), iterator.ow.global_res.TrafoOut_MW[24*2*(31+19):24*2*(31+26)], label='February')
#    ax.plot(np.arange(0,2*24*7), iterator.ow.global_res.TrafoOut_MW[24*2*(31+28+31+30+6):24*2*(31+28+31+30+13)], label='May')
#    plt.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
#    plt.xlim(0,48*7)
#    plt.title('Net load at transformer for a given week')
#    plt.xticks(np.arange(0,48*7,48),[util.daysnames[(i)%7] for i in range(7)])
#    plt.legend()
#    plt.grid()
#    
#    # Set critical day as res
#    minv = iterator.ow.res_v.min(axis=1).idxmin()
#    net.res_bus.vm_pu = iterator.ow.res_v.loc[minv]
#    f, ax = plt.subplots()
#    plot_v_profile(net, ax=ax)
#    plt.title('Voltage profile at minimum Vpu, {}'.format(minv))
    
    