# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:57:22 2020

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

import time as time
import datetime as dt

from ppgrid import *
from grid import *
import VoltVar


# importing base grid


folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
print('\t loading grids from folder {}'.format(folder_grids))
# load predefined grid
folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
net = pp.from_json(folder_grids + 'base_grid.json')
#net_res = pp.from_json(folder_grids + 'res_grid.json')

# number of LV trafos per IRIS
folder_lv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
lv_iris = pd.read_csv(folder_lv+'Nb_BT_IRIS2016.csv',
                      engine='python', index_col=0)

# Base profiles
print('\tloading base profiles')
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles\\'
profiles_load = pd.read_csv(folder_profiles  + r'profiles_iris.csv',
                       engine='python', index_col=0)
# Load IRIS polygons
print('Loading IRIS polygons')
folder_iris = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
iris_poly = pd.read_csv(folder_iris+file_iris,
                        engine='python', index_col=0)
# Plot supply zone
iris = net.load.zone.astype(int).unique()
polygons = util.do_polygons(iris_poly.loc[iris], plot=False)
cmap = plt.get_cmap('plasma')
nb_bt_b = net.load[net.load.type_load=='Base'].groupby('zone').type_load.count()[iris] # Number of supplied LV trafos per IRIS by SS
nb_bt=lv_iris.Nb_BT[iris] # Total number of trafos per IRIS
supply = supply=1-((nb_bt-nb_bt_b)/nb_bt) # Ratio of supply
colors=cmap(supply)

ax=util.plot_polygons(util.list_polygons(polygons,(supply).index), color=colors, edgecolor='darkgrey', linestyle='--')
plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
plt.plot(net.bus_geodata.x[0], net.bus_geodata.y[0], 'o', color='red')

tranches = np.linspace(0,1,6)
labels = ['{}%'.format(int(t*100)) for t in tranches]
colorslab = cmap(tranches)
util.do_labels(labels, colorslab, ax)

plt.title('Supply of load from substation')
# EV profiles
#%%
folder_evs = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles\\'
fns = ['EV_p0.5_w0.csv', 'EV_p0.5_w0.5.csv', 'EV_p1_w0.csv', 'EV_p1_w0.5.csv']
for ev_prof_fn in fns:
       
    print('\t loading ev profile {}'.format(ev_prof_fn))
    ev_prof = pd.read_csv(folder_evs + ev_prof_fn,
                          engine='python', index_col=0)
    # Removing EV loads if already existing in grid
    net.load=net.load[~(net.load.type_load=='EV')]
    # merge Home & Work profiles and set up profiles and loads for pandapower
    homecols = [c for c in ev_prof.columns if 'Home' in c]
    workcols = [c for c in ev_prof.columns if 'Work' in c]
    idhome = [c.replace('Home_','EV_') for c in homecols]
    idwork = [c.replace('Work_','EV_') for c in workcols]
    ev_prof_h = ev_prof[homecols]
    ev_prof_w = ev_prof[workcols]
    ev_prof_h.columns = idhome
    ev_prof_w.columns = idwork
    ev_prof = ev_prof_h + ev_prof_w
    ev_prof_pu = ev_prof / ev_prof.max(axis=0) # profile in pu
    load_ev_lv = ev_prof.max(axis=0)  # Max load of EV per LV trafo per IRIS
    load_ev_lv.index = [int(c.replace('EV_','')) for c in load_ev_lv.index]
    load_ev_lv = load_ev_lv / lv_iris.Nb_BT[load_ev_lv.index]
    
    # plot EV load per IRIS
    iris = [int(i.replace('EV_', '')) for i in ev_prof.columns]
    polygons = util.do_polygons(iris_poly.loc[iris], plot=False)
    
    #plotting total MAX EV load per iris
    evload = ev_prof.max()
    polys = util.list_polygons(polygons, iris)
    
    cmap = plt.get_cmap('plasma')
    colors = cmap(evload/evload.max())
    #plot polygons
    ax=util.plot_polygons(polys, color=colors, edgecolor='darkgrey', linestyle='--')
    # add legend
    tranches = np.linspace(0,np.ceil(evload.max()),5)
    labels = ['{} MW'.format(i) for i in tranches]
    colorslab = cmap(tranches/evload.max())
    util.do_labels(labels, colorslab, ax)
    plt.title('Max EV load [MW]')
    
    # Plot ratio Home Work charging
    hwch = (ev_prof_h.sum()/ev_prof_w.sum())
    hwch[hwch<1] = 2 - (ev_prof_w.sum()[hwch<1]/ev_prof_h.sum()[hwch<1])
    ratiomax=4
    colors = cmap((hwch-(-(ratiomax-1)))/(ratiomax*2-1))
    #plot polygons
    ax=util.plot_polygons(polys, color=colors, edgecolor='darkgrey', linestyle='--')
    # add legend
    tranches = np.linspace(0,1,ratiomax*2-1)
    labels = ['Work:Home {}:1'.format(i) for i in np.arange(ratiomax,0,-1)] + ['Work:Home 1:{}'.format(i) for i in np.arange(2,ratiomax+1)]
    colorslab = cmap(tranches)
    util.do_labels(labels, colorslab, ax)
    plt.title('Ratio Work:Home of EV load')
    
    # Plot load of extreme IRIS
    plt.figure()
    ih = hwch.idxmax()
    iw = hwch.idxmin()
    plt.plot(ev_prof_pu[ih][0:24*2*7*3], label=iris_poly.IRIS_NAME[int(ih.replace('EV_',''))])
    plt.plot(ev_prof_pu[iw][0:24*2*7*3], label=iris_poly.IRIS_NAME[int(iw.replace('EV_',''))])
    plt.xticks(np.arange(0,24*2*7*3+1,24*2), np.tile(util.dsnms,3))
    ax = plt.gca()
    for t in ax.get_xticklabels(minor=False):
        xy = t.get_position()
        t.set_horizontalalignment('left')
        t.set_y(0)
    plt.grid()
    plt.legend()
    w = 2
    plt.xlim(7*24*2*w,7*24*2*(1+w))
    plt.ylim(0, 1.1)
    plt.title('EV loads for two extreme types')
    plt.ylabel('Load [pu]')
    
    
    
    # transform profile to yearly
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
    # Adding database of EV profiles
    profiler.add_profile_db(ev_prof_pu)
    for n in net.load.zone.unique():
        if not ((n is None) or not (n==n)):
            # Assigning index of base load profiles
            profiler.add_profile_idx(element='load', 
                                     idx=net.load[(net.load.zone==n) & (net.load.type_load=='Base')].index, 
                                     variable='scaling', 
                                     profile=str(int(n)))
            # Assigning index of EV load profiles
            profiler.add_profile_idx(element='load',
                                     idx=net.load[(net.load.zone==n) & (net.load.type_load=='EV')].index,
                                     variable='scaling',
                                     profile='EV_' + str(int(n)))
    
    output_folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_EVs\\'
    # Setting iterator
    time_steps=profiler.profiles_db.index   
    iterator = Iterator(net=net, profiler=profiler)
    of = output_folder + ev_prof_fn
    iterator.iterate(time_steps=time_steps, save=True, outputfolder=of, ultraverbose=False)
    
    # Net load at transformer - May and February
    f, ax = plt.subplots()
    ax.plot(np.arange(0,2*24*7), iterator.ow.global_res.TrafoOut_MW[24*2*(31+19):24*2*(31+26)], label='February')
    ax.plot(np.arange(0,2*24*7), iterator.ow.global_res.TrafoOut_MW[24*2*(31+28+31+30+6):24*2*(31+28+31+30+13)], label='May')
    plt.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
    plt.xlim(0,48*7)
    plt.title('Net load at transformer for a given week')
    plt.xticks(np.arange(0,48*7,48),[util.daysnames[(i)%7] for i in range(7)])
    plt.legend()
    plt.grid()
    
    # Set critical day as res
    minv = iterator.ow.res_v.min(axis=1).idxmin()
    net.res_bus.vm_pu = iterator.ow.res_v.loc[minv]
    f, ax = plt.subplots()
    plot_v_profile(net, ax=ax)
    plt.title('Voltage profile at minimum Vpu, {}'.format(minv))
    