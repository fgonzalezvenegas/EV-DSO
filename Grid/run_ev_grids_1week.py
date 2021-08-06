# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 01:09:46 2021
Analyzing load flow results
@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

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

font = {'size':8}
plt.rc('font', **font)

times = [time()]
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
iris_ss = net.load.zone.astype(int).unique()
polygons = util.do_polygons(iris_poly.loc[iris_ss], plot=False)
polys = util.list_polygons(polygons,iris_ss)
cmap = plt.get_cmap('plasma')
nb_bt_b = net.load[net.load.type_load=='Base'].groupby('zone').type_load.count()[iris_ss] # Number of supplied LV trafos per IRIS by SS
nb_bt=lv_iris.Nb_BT[iris_ss] # Total number of trafos per IRIS
supply = 1-((nb_bt-nb_bt_b)/nb_bt) # Ratio of supply
colors=cmap(supply[iris_ss])

ax=util.plot_polygons(polys, color=colors, edgecolor='darkgrey', linestyle='--')
plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
plt.plot(net.bus_geodata.x[0], net.bus_geodata.y[0], 'o', color='red')

tranches = np.linspace(0,1,6)
labels = ['{}%'.format(int(t*100)) for t in tranches]
colorslab = cmap(tranches)
util.do_labels(labels, colorslab, ax)


print('Loading Histograms of distance')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
                    engine='python', index_col=0)
if 'ZE' in hwork.columns:
    hwork = hwork.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
    hhome = hhome.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
times.append(time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))

# DEMOGRAPHIC DATA, NUMBER OF RESIDENTS & WORKERS PER IRIS
# IRIS & Commune info
print('Loading IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
times.append(time())

#%% Plotting number of EVs
case = 'syn' #'cont', 'syn'
if case=='syn':
    wr = 0.5
    p = 1
    scale_h = 1000 # EVs
    scale_w = 1000 # EVs
elif case=='cont':
    wr = 0.3
    p = 0.5
    scale_h = 500 # EVs
    scale_w = 500 # EVs
nevs_h = iris.N_VOIT[iris_ss] * supply * (1-wr) * p
nevs_w =  hwork.loc[iris.COMM_CODE[iris_ss]].sum(axis=1) * wr * p
nevs_w.index=iris_ss
nevs_w =  nevs_w * iris.Work_pu[iris_ss]*1.78 * supply


# Plot supply zone
cmap = plt.get_cmap('Blues')

# colors
colors_h=cmap(nevs_h/scale_h)
colors_w=cmap(nevs_w/scale_w)

f, axs = plt.subplots(1,2)
f.tight_layout()
ax = axs[0]
plt.sca(ax)
util.plot_polygons(polys, color=colors_h, edgecolor='darkgrey', linestyle='--', ax=ax)
plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.title('(a) Residential', y=-0.1)
tranches = np.linspace(0,1,5)
labels = ['{} EVs'.format(int(t*scale_h)) for t in tranches]
colorslab = cmap(tranches)
#util.do_labels(labels, colorslab, ax)
#plt.legend(loc=1)
pos = ax.get_position()
ax.set_position([0.025, pos.y0+0.02, 0.46, pos.height-0.07])


ax = axs[1]
plt.sca(ax)
util.plot_polygons(polys, color=colors_w, edgecolor='darkgrey', linestyle='--', ax=ax)
plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.title('(b) Work-place', y=-0.1)
tranches = np.linspace(0,1,5)
labels = ['{} EVs'.format(int(t*scale_w)) for t in tranches]
colorslab = cmap(tranches)
#util.do_labels(labels, colorslab, ax)
#plt.legend(loc=1)
pos = ax.get_position()
ax.set_position([0.515, pos.y0+0.02, 0.46, pos.height-0.07])


# Adding zoom to Brive area
az1 = plt.axes([.325, .58, 0.15, .3])
util.plot_polygons(polys, color=colors_h, edgecolor='darkgrey', linestyle='--', ax=az1)
plot_lines(net.line_geodata, col='coords', ax=az1, color='k', linewidth=0.3)
plt.xticks([])
plt.yticks([])
# setting zoom
x0, x1 = 1.5125, 1.5836
y0, y1 = 45.13865,45.186013
plt.xlim(x0,x1)
plt.ylim(y0,y1)

# Adding zoom to Brive area
az2 = plt.axes([.815, .58, 0.15, .3])
util.plot_polygons(polys, color=colors_w, edgecolor='darkgrey', linestyle='--', ax=az2)
plot_lines(net.line_geodata, col='coords', ax=az2, color='k', linewidth=0.3)
plt.xticks([])
plt.yticks([])
# setting zoom
x0, x1 = 1.5125, 1.5836
y0, y1 = 45.13865,45.186013
plt.xlim(x0,x1)
plt.ylim(y0,y1)



util.do_labels(labels, colorslab, ax)
ax.get_legend().remove()
f.legend(ncol=len(labels), loc=9)

f.set_size_inches(7.47,3.5)

plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\EVs_wh_p{}wr{}_th.pdf'.format(p,str(wr).replace('.','')))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\EVs_wh_p{}wr{}_th.jpg'.format(p,str(wr).replace('.','')), dpi=300)




#%% Plotting grid

# Plot type IRIS
c_t = {'H':'b', 'Z':'g'}
colors = [c_t[iris_poly.IRIS_TYPE[i]] for i in iris_ss]
#plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.5)

linetypes = ['UG_95', 'OH_34', 'OH_75', 'UG_150', 'OH_54', 'OH_148', 'UG_240']
colors_tech = {'Underground' : ['maroon', 'red', 'orangered', 'salmon', 'khaki'],
               'Overhead' : ['midnightblue', 'mediumblue', 'slateblue', 'cornflowerblue', 'skyblue']}
colortype = {'UG_95': ('orangered', 1),
             'UG_150': ('red', 1.3),
             'UG_240': ('maroon', 1.5),
             'OH_34': ('cornflowerblue', 1),
             'OH_54': ('slateblue', 1.2),
             'OH_75': ('mediumblue', 1.4),
             'OH_148': ('midnightblue', 1.6)}

fcenter  =['F00', 'F01','F02', 'F03', 'F04', 'F06', 'F07', 'F08','F09','F10']
# coords for Boriette
xySS = net.bus_geodata[['x','y']].loc[0]
# Farther buses
feeders = net.bus.Feeder.dropna().drop(0).unique()
busf = {f : net.bus[net.bus.Feeder == f].index[-1] 
            for f in feeders}
fdx = {'F00':(0,-0.0052),'F01':(-0.005,0)}
f, ax= plt.subplots()
plt.tight_layout()
util.plot_polygons(polys, color=colors, alpha=0.3, edgecolor='k', ax=ax)


plt.plot(xySS.x, xySS.y, marker='o', markersize=5, color='purple', label='HV/MV Substation')
for key, (v1,v2) in colortype.items():
    ls = net.line[net.line.std_type == key].index
    plot_lines(net.line_geodata.loc[ls], col='coords',ax=ax, color=v1, linewidth=v2*1.5, label=key)
# adding feeder names
for f in feeders:
    if not (f in fcenter):
        ax.text(x=net.bus_geodata.loc[busf[f]].x,y=net.bus_geodata.loc[busf[f]].y, s=f)
plt.xticks([])
plt.yticks([])

# adding zoom to brive area
a = plt.axes([.65, .6, .3, .3])
plt.plot(xySS.x, xySS.y, marker='o', markersize=5, color='purple')
util.plot_polygons(polys, color=colors, alpha=0.3, edgecolor='k', ax=a)
colortype = {'UG_95': ('orangered', 1),
             'UG_150': ('red', 1.3),
             'UG_240': ('maroon', 1.5),
             'OH_34': ('cornflowerblue', 1),
             'OH_54': ('slateblue', 1.2),
             'OH_75': ('mediumblue', 1.4),
             'OH_148': ('midnightblue', 1.6)}
for key, (v1,v2) in colortype.items():
    ls = net.line[net.line.std_type == key].index
    plot_lines(net.line_geodata.loc[ls], col='coords',ax=a, color=v1, linewidth=v2*1.5)
for f in fcenter:
    if f in fdx:
        dx, dy = fdx[f]
    else:
        dx, dy = 0,0
    a.text(x=net.bus_geodata.loc[busf[f]].x + dx,
           y=net.bus_geodata.loc[busf[f]].y + dy, 
           s=f)
# setting zoom
x0, x1 = 1.5125, 1.5836
y0, y1 = 45.13865,45.186013
plt.xlim(x0,x1)
plt.ylim(y0,y1)

r= ptc.Rectangle([x0,y0], (x1-x0), (y1-y0), facecolor='None', edgecolor='k')
ax.add_patch(r )
#plt.plot([x0,x1],[y0,y0])
#plt.plot([x0,x1],[y1,y1])

plt.xticks([])
plt.yticks([])
plt.gcf().legend(loc=3, framealpha=1)
plt.gcf().set_size_inches( 5.24,  4.06)
#util.do_labels(['Urban', 'Rural'], list(c_t.values()), ax=ax)
#plt.title('Type IRIS')

plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\grid.pdf')
plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\grid.jpg', dpi=300)


#%% Plotting  grid data: length & max load

length = net.line.groupby('Feeder').length_km.sum()
ruralf = ['F{}'.format(i) for i in range(14,20)]
x= np.arange(len(length))
xr = np.arange(14,20)

#f, ax = plt.subplots()
f,axs = plt.subplots(1,2)
ax = axs[0]
plt.sca(ax)

plt.bar(length.index, length, color='b', label='Urban')
plt.bar(length[ruralf].index, length[ruralf], color='g', label='Rural')
plt.legend()
plt.ylabel('Length [km]')
ax.set_xticklabels(length.index, rotation=90)
plt.title('(b) Length', y=-0.24)
#f.set_size_inches(3.5,2.7)
plt.tight_layout()

load = []
for feed in length.index:
    load.append(net.load[net.load.bus.isin(net.bus[net.bus.Feeder==feed].index)].p_mw.sum())
load = np.array(load)

#f2, ax = plt.subplots()
ax = axs[1]
plt.sca(ax)

plt.bar(length.index, load, color='b', label='Urban')
plt.bar(length[ruralf].index, load[xr], color='g', label='Rural')
plt.legend()
plt.ylabel('Max load [MW]')
ax.set_xticklabels(length.index, rotation=90)
#f2.set_size_inches(3.5,2.7)
f.set_size_inches(7.57,3.3)
plt.title('(b) Max load', y=-0.24)
plt.tight_layout()

f.savefig(r'c:\user\U546416\Pictures\Boriette\GridData\feeders.pdf')
f.savefig(r'c:\user\U546416\Pictures\Boriette\GridData\feeders.jpg', dpi=300)

#f.savefig(r'c:\user\U546416\Pictures\Boriette\GridData\feederslength.pdf')
#f.savefig(r'c:\user\U546416\Pictures\Boriette\GridData\feederslength.jpg', dpi=300)
#f2.savefig(r'c:\user\U546416\Pictures\Boriette\GridData\feedersdemand.pdf')
#f2.savefig(r'c:\user\U546416\Pictures\Boriette\GridData\feedersdemand.jpg', dpi=300)


#%% Reading grid and plotting PV integration
folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
gridname =  'res_grid_equi_2050.json'#'res_grid_equi_2050/res_grid_roof_2.json'
net = pp.from_json(folder_grids + gridname)

# Compute installed RES per IRIS
conso_prod_iris = {}
for i in net.bus.iris.unique():
    if not np.isnan(i):
        bss = net.bus[net.bus.iris == i].index
        conso_prod_iris[int(i)] = [net.load[net.load.bus.isin(bss)].p_mw.sum(),
                                    net.sgen[(net.sgen.type == 'RES_PV') & (net.sgen.bus.isin(bss))].p_mw.sum(),
                                    net.sgen[(net.sgen.type == 'Comm_PV') & (net.sgen.bus.isin(bss))].p_mw.sum(),
                                    net.sgen[(net.sgen.type == 'Farm_PV') & (net.sgen.bus.isin(bss))].p_mw.sum()] 
conso_prod_iris = pd.DataFrame(conso_prod_iris, index=['Conso_MWh', 'PV_Home_MW','PV_Commercial_MW', 'PV_farm_MW']).T      

# Plot installed PV per IRIS
cmap = plt.get_cmap('YlOrRd')
scale = 3
tranches = np.linspace(0,1,6)
labels = ['{} MW'.format(round(t*scale,1)) for t in tranches]
colorslab = cmap(tranches)

f, axs = plt.subplots(1,2)
f.tight_layout()

# plotting rooftop
ax = axs[0]
plt.sca(ax)
rt = conso_prod_iris.PV_Commercial_MW.loc[iris_ss] + conso_prod_iris.PV_Home_MW.loc[iris_ss]
colorsrt = cmap(rt/scale)
util.plot_polygons(polys, ax=ax, color=colorsrt, edgecolor='darkgrey', linestyle='--')
plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
#plt.title('Rooftop PV: {} MW'.format(round(rt.sum())))
plt.title('(a) Rooftop', y=-0.1)
#util.do_labels(labels, colorslab, ax)
#plt.legend(loc=1)
pos = ax.get_position()
ax.set_position([0.025, pos.y0+0.02, 0.46, pos.height-0.07])

# plotting farms

ax = axs[1]
plt.sca(ax)
colorsgm = cmap((conso_prod_iris.PV_farm_MW.loc[iris_ss])/scale)
util.plot_polygons(polys, ax=ax, color=colorsgm, edgecolor='darkgrey', linestyle='--')
plot_lines(net.line_geodata, col='coords', ax=ax, color='k', linewidth=0.3)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
#plt.title('Ground-mounted PV: {} MW'.format(round(conso_prod_iris.PV_farm_MW.loc[iris_ss].sum())))
plt.title('(b) Ground-mounted', y=-0.1)
#util.do_labels(labels, colorslab, ax)
#plt.legend(loc=1)
pos = ax.get_position()
ax.set_position([0.515, pos.y0+0.02, 0.46, pos.height-0.07])


# Adding zoom to Brive area
az1 = plt.axes([.325, .58, 0.15, .3])
util.plot_polygons(polys, color=colorsrt, edgecolor='darkgrey', linestyle='--', ax=az1)
plot_lines(net.line_geodata, col='coords', ax=az1, color='k', linewidth=0.3)
plt.xticks([])
plt.yticks([])
# setting zoom
x0, x1 = 1.5125, 1.5836
y0, y1 = 45.13865,45.186013
plt.xlim(x0,x1)
plt.ylim(y0,y1)

# Adding zoom to Brive area
az2 = plt.axes([.815, .58, 0.15, .3])
util.plot_polygons(polys, color=colorsgm, edgecolor='darkgrey', linestyle='--', ax=az2)
plot_lines(net.line_geodata, col='coords', ax=az2, color='k', linewidth=0.3)
plt.xticks([])
plt.yticks([])
# setting zoom
x0, x1 = 1.5125, 1.5836
y0, y1 = 45.13865,45.186013
plt.xlim(x0,x1)
plt.ylim(y0,y1)


util.do_labels(labels, colorslab, ax)
ax.get_legend().remove()
f.legend(ncol=len(labels), loc=9)

f.set_size_inches(7.47,3.3)

plt.savefig(r'c:\user\U546416\Pictures\Boriette\PV\Installed_cap_{}_th.pdf'.format(gridname[:-5]))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\PV\Installed_cap_{}_th.jpg'.format(gridname[:-5]), dpi=300)


#%% Do installed capacity per feeder & # EVs per feeder
case = 'synergies' #synergies, continuity
if case == 'synergies':
    gridname =  'res_grid_roof_2.json'#'res_grid_equi_2050/res_grid_roof_2.json'
    wr = 0.5
    p = 1
elif case == 'continuity':
    gridname =  'res_grid_equi_2050.json'#'res_grid_equi_2050/res_grid_roof_2.json'
    wr = 0.3
    p = 0.5
    
net = pp.from_json(folder_grids + gridname)
avg_chp_w = 9.6875 #kW
avg_chp_h = 6.595   #kW

# Computing installed capacity PV per feeder

PV_feeder = pd.DataFrame([net.bus.Feeder[net.sgen.bus].values, net.sgen.p_mw], index=['Feeder','PV']).T.groupby('Feeder').PV.sum()

# Computing number of EVs per feeder
# evs per trafo BT
nevs_h_bt = (iris.N_VOIT[iris_ss] * (1-wr) * p)/nb_bt
nevs_w_bt =  hwork.loc[iris.COMM_CODE[iris_ss]].sum(axis=1) * wr * p
nevs_w_bt.index=iris_ss
nevs_w_bt =  (nevs_w_bt * iris.Work_pu[iris_ss]*1.78)/ nb_bt
# evs_per_feeder
evs_feeder = pd.DataFrame([net.bus.Feeder[net.load.bus].values, nevs_h_bt[net.load.zone], nevs_w_bt[net.load.zone]], 
                          index=['Feeder','NEVs_h', 'NEVs_w']).T.groupby('Feeder').sum()
# P per feeder (in MW)
p_ev_feeder = evs_feeder * np.array((avg_chp_h, avg_chp_w))/1000

#Plotting
f, axs = plt.subplots(1,2, sharey=True)
plt.sca(axs[0])
plt.bar(p_ev_feeder.index, p_ev_feeder.NEVs_h, label='Residential')
plt.bar(p_ev_feeder.index, p_ev_feeder.NEVs_w, bottom=p_ev_feeder.NEVs_h, label='Work-place')
plt.xticks(rotation=90)
plt.ylabel('Power [MW]')
plt.legend()
plt.title('(a) EV capacity', y=-.25)
plt.yticks([0,5,10,15,20])
plt.sca(axs[1])
plt.bar(PV_feeder.index, PV_feeder.values)
plt.xticks(rotation=90)
plt.title('(b) PV capacity', y=-.25)

f.set_size_inches(7.57,3.3)
plt.tight_layout()

plt.savefig(r'c:\user\U546416\Pictures\Boriette\PV\EVsPV_{}_th.pdf'.format(case))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\PV\EVsPV_{}_th.jpg'.format(case), dpi=300)


##%% Reading results
#
#case = 'Conti' #
#
#if case == 'EV':
#    folder_res = r'ev\\'
#    folder_restou = r'evtou\\'
#    file_prof = 'EV_p0.5_w0.3.csv'
#    file_proftou = 'EV_p0.5_w0.3.csv'
#if case == 'Conti':
#    folder_res = r'ev_pvcontinue\\'
#    folder_restou = r'evtou_pvcontinue\\'
#    file_prof = 'EV_p0.5_w0.3csv'
#if case == 'Syn':
#    folder_res = r'evopt_pvrooftop\\'
#    file_prof = 'EV_p1_w0.5_peakweek_optimized.3csv'
#    
#globres = pd.read_csv(mainfolder + folder_res + 'global_res.csv', index_col=0, engine='python')
#globrestou = pd.read_csv(mainfolder + folder_restou + 'global_res.csv', index_col=0, engine='python')
#
#volts = pd.read_csv(mainfolder + folder_restou + 'vm_pu.csv', index_col=0, engine='python')
#res_base = pd.read_csv(mainfolder + r'base_noEV\global_results.csv', index_col=0, engine='python')

#print(globres.TrafoOut_MW.max())
#print(globres.TotLoad_MW.sum()/2)
#print(globres.TrafoOut_MW.sum()/2)
#
#print(globrestou.TrafoOut_MW.max())
#print(globrestou.TotLoad_MW.sum()/2)
#print(globrestou.TrafoOut_MW.sum()/2)
#
#print(res_base.TrafoOut_MW.max())
#print(res_base.TotLoad_MW.sum()/2)
#print(res_base.TrafoOut_MW.sum()/2)
#%% Plotting Peak winter week & august week winter
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles_thesis\\'
mainfolder  = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\\'


ww = 9 # Winter week
sw = 34 # Summer week
nstepsweek = 7*2*24

# Reading base case
res_base = pd.read_csv(mainfolder + r'base_noEV\global_res.csv', index_col=0, engine='python')

idxsww = res_base.index[(ww-1)*nstepsweek:ww*nstepsweek]
idxssw = res_base.index[(sw-1)*nstepsweek:sw*nstepsweek]

cases = {'EVonly': 'evtou',
         'Continuity': 'evtou_pvcontinue',
         'HighPenetration': 'evtou_pvrooftop',
         'Synergies': 'evopt_pvrooftop'}
baseload = res_base.TrafoOut_MW

x = np.arange(0,24*7,0.5)

globres = {}
for case, folder in cases.items():
    # reading results
    globres[case] = pd.read_csv(mainfolder + folder + r'\global_res.csv', index_col=0, engine='python')
#%%

titles = ['(a) Peak load week', '(b) Min load week']
for case, folder in cases.items():
    if 'Static_gen' in globres[case].columns:
        pv = globres[case].Static_gen
    else:
        pv = 0
#        pv = globres.Static_gen
#        ev = globres.TrafoOut_MW + pv - baseload
#    else:
#        pv = 0
#        ev = globres.TrafoOut_MW - baseload
    netload = baseload-pv
    ev  = globres[case].TrafoOut_MW - netload
    f, axs = plt.subplots(1,2, sharey=True)
    for i, idxs in enumerate([idxsww, idxssw]):
        plt.sca(axs[i])
#        if pv is None:
        stacks = [netload[idxs], ev[idxs]]
        labels = ['Net load', 'EV load']
#        else:
#            stacks = [baseload[idxs], -pv[idxs], ev[idxs]]
#            labels = ['Net load', 'EV load', 'PV generation']
        plt.stackplot(x, stacks, 
                      labels=labels)
        plt.axhline(108, color='r', linestyle='--', label='Substation capacity')
        plt.xlim(x[0],x[-1])
        locs, labels = plt.xticks(range(0,24*7, 24), util.dsnms, ha='left')
           
        if i==0:
            plt.ylabel('Power [MVA]')
        plt.legend(framealpha=1, loc=2)
        plt.title(titles[i], y=-0.17)
    f.set_size_inches(7.57,3.3)
    
    f.tight_layout()
#    plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\Netload_{}.pdf'.format(case))
#    plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\Netload_{}.jpg'.format(case), dpi=300)
#
#f = plt.figure()
#plt.stackplot(x, [res_base.TrafoOut_MW.iloc[ixweek:ixweek+dweek],
#                  (globrestou.TrafoOut_MW.iloc[ixweek:ixweek+dweek]-res_base.TrafoOut_MW.iloc[ixweek:ixweek+dweek])], 
#            labels=['Base load', 'EV load'])
#plt.axhline(108, color='r', linestyle='--', label='Substation capacity')
#plt.xlim(x[0],x[-1])
#locs, labels = plt.xticks(range(0,dweek, 24*2), util.dsnms, ha='left')
#   
#plt.ylabel('Power [MVA]')
#
#f.set_size_inches(3.8,3.3)
#
#f.tight_layout()
#plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\WinterWeek_{}_tou.pdf'.format(case))
#plt.savefig(r'c:\user\U546416\Pictures\Boriette\EVLoads\WinterWeek_{}_tou.jpg'.format(case), dpi=300)
#%% Do colorbars:
f, axs = plt.subplots(1,2)
sm = ScalarMappable(cmap=plt.get_cmap('jet'))
sm.set_array(np.arange(0,101,10))
cbar = plt.colorbar(mappable=sm, cax=axs[0])
cbar.set_label('Line loading [%]')
sm = ScalarMappable(cmap=plt.get_cmap('coolwarm'))
sm.set_array(np.arange(0.95,1.05,0.01))
cbar = plt.colorbar(mappable=sm, cax=axs[1])
cbar.set_label('Voltage [pu]')
cbar.set_ticks((0.95,0.975,1,1.025,1.05))
f.set_size_inches(2,3.3)
plt.tight_layout()

#plt.yticks([])


#%% Plotting lowest V
# Plot V profiles
# Set critical day as result
critday = volts.min(axis=1).idxmin()
net.res_bus.vm_pu = volts.loc[critday]

f, ax = plt.subplots()
plot_v_profile(net, ax=ax)
f.set_size_inches(3.8,3.3)
plt.tight_layout()
plt.legend(loc=1)

# feeders to rem text:
notfrem = ['F16', 'F18', 'F00']
ts = ax.texts
while len(ts)>len(notfrem):
    for t in ts:
        te = t.get_text()
        if not (te in notfrem):
#            print('removing', t.get_text())
            t.remove()
#        continue
#    t.remove()
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\low_{}.pdf'.format(case))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\low_{}.jpg'.format(case), dpi=300)

# Set critical day as res
critday = volts.max(axis=1).idxmax()
net.res_bus.vm_pu = volts.loc[critday]

f, ax = plt.subplots()
plot_v_profile(net, ax=ax)
f.set_size_inches(3.8,3.3)
plt.tight_layout()
plt.legend(loc=4)

# feeders to rem text:
notfrem = ['']
ts = ax.texts
while len(ts)>len(notfrem):
    for t in ts:
        te = t.get_text()
        if not (te in notfrem):
#            print('removing', t.get_text())
            t.remove()
#        continue
#    t.remove()
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\high_{}.pdf'.format(case))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\high_{}.jpg'.format(case), dpi=300)

#plt.title('Voltage at critical hour, {}'.format(critday))

#critpower = iterator.ow.global_res.TotLoad_MW.idxmax()
#net.res_bus.vm_pu = iterator.ow.res_v.loc[critpower]
#f, ax = plt.subplots()
#plot_v_profile(net, ax=ax)
#plt.title('Voltage at max load, {}'.format(critpower))

#%% Plotting highest and lowest voltages for each node

def alternate_labels(ax, dy=-0.02):
    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if i%2:
            l.set_y(l.get_position()[1]+dy)
    ylim = ax.get_ylim()
    ax.plot(0,0)
    ax.set_ylim(ylim)
    
def plot_vminmax(vpu, net, ax=None, vmin=0.95, vmax=1.05):
    firstbus = net.line.groupby('Feeder').to_bus.min()
    if ax is None:
        f, ax = plt.subplots()
    plt.plot(vpu.min(axis=0).values[:-1])
    plt.plot(vpu.max(axis=0).values[:-1])
    for b in firstbus:
        plt.axvline(b, color='grey', linestyle='--', linewidth=0.5)
    plt.xticks(firstbus, firstbus.index, ha='left')
    plt.axhline(vmin, color='r', linestyle='--')
    plt.axhline(vmax, color='r', linestyle='--')
    plt.xlim(0,vpu.shape[1])
    plt.tight_layout()
    return ax

volts = {}

cases = {'EV-only': 'evtou',
         'Continuity': 'evtou_pvcontinue',
#         'HighPenetration': 'evtou_pvrooftop',
         'Synergies': 'evopt_pvrooftop'}
for case, folder in cases.items():
    # reading results
    volts[case] = pd.read_csv(mainfolder + folder + r'\vm_pu.csv', index_col=0, engine='python')
    
#%% 
# getting right order
plt.plot()


#%% Reading results for 1-week Synergies scenario    
folder_respw = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop_peak\\'
folder_resmw = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop_low\\'
file_profpw = 'EV_p1_w0.5_peakweek_optimized.csv'
file_profmw = 'EV_p1_w0.5_minweek_optimized.csv'

ev_peak =  pd.read_csv(folder_profiles + file_profpw, index_col=0, engine='python')
ev_minw =  pd.read_csv(folder_profiles + file_profmw, index_col=0, engine='python')
gr_peak =  pd.read_csv(folder_respw + 'global_res.csv', index_col=0, engine='python')
gr_minw =  pd.read_csv(folder_resmw + 'global_res.csv', index_col=0, engine='python')
gr_syn = pd.concat([gr_peak, gr_minw])

vpu_peak =  pd.read_csv(folder_respw + 'vm_pu.csv', index_col=0, engine='python')
vpu_minw =  pd.read_csv(folder_resmw + 'vm_pu.csv', index_col=0, engine='python')
vpu_syn = pd.concat([vpu_peak, vpu_minw])

linel_peak =  pd.read_csv(folder_respw + 'line_loading.csv', index_col=0, engine='python')
linel_minw =  pd.read_csv(folder_resmw + 'line_loading.csv', index_col=0, engine='python')
linel_syn = pd.concat([linel_peak, linel_minw])

linel_syn.to_csv(r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop\line_loading.csv')
vpu_syn.to_csv(r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop\vm_pu.csv')
gr_syn.to_csv(r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop\global_res.csv')

#%% Plotting lowest V
# Plot V profiles
# Set critical day as result

case = 'mw'
if case=='pw':    
    critday = vpu_peak.min(axis=1).idxmin()
    net.res_bus.vm_pu = vpu_peak.loc[critday]
elif case=='mw':    
    critday = vpu_minw.min(axis=1).idxmin()
    net.res_bus.vm_pu = vpu_minw.loc[critday]

f, ax = plt.subplots()
plot_v_profile(net, ax=ax)
f.set_size_inches(3.8,3.3)
plt.tight_layout()
plt.legend(loc=1)

# feeders to rem text:
notfrem = ['F16', 'F18', 'F00']
ts = ax.texts
while len(ts)>len(notfrem):
    for t in ts:
        te = t.get_text()
        if not (te in notfrem):
#            print('removing', t.get_text())
            t.remove()
#        continue
#    t.remove()
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\Synergies_low_{}.pdf'.format(case))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\Synergies_low_{}.jpg'.format(case), dpi=300)

# Set critical day as res
if case=='pw':    
    critday = vpu_peak.min(axis=1).idxmax()
    net.res_bus.vm_pu = vpu_peak.loc[critday]
    notfrem = ['']
elif case=='mw':    
    critday = vpu_minw.min(axis=1).idxmax()
    net.res_bus.vm_pu = vpu_minw.loc[critday]
    notfrem = ['F16', 'F17', 'F18', 'F19']
    
f, ax = plt.subplots()
plot_v_profile(net, ax=ax)
f.set_size_inches(3.8,3.3)
plt.tight_layout()
plt.legend(loc=4)

# feeders to rem text:
ts = ax.texts
while len(ts)>len(notfrem):
    for t in ts:
        te = t.get_text()
        if not (te in notfrem):
#            print('removing', t.get_text())
            t.remove()
        continue
    t.remove()
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\Synergies_high_{}.pdf'.format(case))
plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\Synergies_high_{}.jpg'.format(case), dpi=300)

#%% Identifying key nodes
b0 = net.ext_grid.bus[0]
distance_bus = ppt.calc_distance_to_bus(net, b0)
fbus = pd.DataFrame([distance_bus, net.bus.Feeder]).T.groupby('Feeder').idxmax().squeeze()

feeders = ['F06', 'F17']

week = 9 
idxi = (week-1)*7*2*24
idxf = idxi + 7*2*24
x = np.arange(0,24*7,0.5)

f, ax = plt.subplots()
for f in feeders:
    plt.plot(x, volts[str(fbus[f])].iloc[idxi:idxf], label=f)
plt.axhline(1.05, color='r', linestyle='--')
plt.axhline(0.95, color='r', linestyle='--')

ax.set_xlim(0, 7*24)
ax.xaxis.set_ticks(range(12,7*24,24))
ax.xaxis.set_ticks_position('none') 
ax.xaxis.set_ticklabels(['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
ax.set_xlabel('Time')
ax.set_ylabel('Voltage [pu]')
xpoints = np.arange(0,8*24,24)
for p in xpoints:
    plt.axvline(p, color = 'grey', linewidth = 0.5, linestyle='--')
plt.legend()


#%% Plotting a grid based on line loading  Reading results
#ev_pvcontinue, ev, evtou_pvcontinue, evtou, evopt_pvrooftop_low, evopt_pvrooftop_peak
case = 'Conti'
if case == 'Conti':
    folder_res = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evtou_pvcontinue\\'
    netname = 'res_grid_equi_2050.json'#'res_grid_equi_2050/res_grid_roof_2.json'
if case=='base':
    folder_res = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\base_noEV\\'
    netname = 'base_grid.json'
if case=='EV':
    folder_res = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evtou\\'
    netname = 'base_grid.json'
if case=='Syn':
    folder_res = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop_peak\\'
    folder_resmw = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evopt_pvrooftop_low\\'
    netname = 'res_grid_roof_2.json'   
if case=='HighPenetration':
    folder_res = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Result_Thesis\evtou_pvrooftop\\'
    netname = 'res_grid_roof_2.json'
    
net = pp.from_json(folder_grids + netname)

line_p = pd.read_csv(folder_res + 'line_loading.csv', index_col=0, engine='python')
if case=='Syn':
    line_p2 = pd.read_csv(folder_resmw + 'line_loading.csv', index_col=0, engine='python')
    line_p = pd.concat([line_p, line_p2])

#%% Plotting a grid based on line loading - Plots
polys = util.list_polygons(polygons,iris_ss)

dcb = 0.13
cmap = plt.get_cmap('jet')
colors = cmap(line_p.max(axis=0)/100)

f,ax = plt.subplots()

xySS = net.bus_geodata[['x','y']].loc[0]

util.plot_polygons(polys, facecolor='none', alpha=0.3, edgecolor='k', ax=ax, linestyle='--')
plot_lines(net.line_geodata, col='coords',ax=ax, color=colors, linewidth=1.5, alpha=1)

plt.xticks([])
plt.yticks([])

bfarms = net.sgen[net.sgen.type=='Farm_PV'].bus
plt.plot(net.bus_geodata.x[bfarms], net.bus_geodata.y[bfarms], 'or', label='Ground-mounted PV')
plt.plot(xySS.x, xySS.y, marker='o', markersize=5, color='purple', label='HV/MV Substation')
plt.tight_layout()
plt.legend(loc=3, framealpha=1)
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width-dcb, pos.height])

# adding zoom to brive area
a = plt.axes([.66-dcb, .64, .3, .3])
#util.plot_polygons(polys, facecolor='none', alpha=0.3, edgecolor='k', ax=a, linestyle='--')
plot_lines(net.line_geodata, col='coords',ax=a, color=colors, linewidth=1.5, alpha=1)

plt.xticks([])
plt.yticks([])

bfarms = net.sgen[net.sgen.type=='Farm_PV'].bus
plt.plot(net.bus_geodata.x[bfarms], net.bus_geodata.y[bfarms], 'or')
plt.plot(xySS.x, xySS.y, marker='o', markersize=5, color='purple', label='HV/MV Substation')
# setting zoom
x0, x1 = 1.5125, 1.5836
y0, y1 = 45.13865,45.186013
plt.xlim(x0,x1)
plt.ylim(y0,y1)


#adding colorbar
sm = ScalarMappable(cmap=cmap)
sm.set_array(np.arange(0,101,10))
cax = f.add_axes([1-dcb, 0.05, 0.025, 0.9])
cbar = plt.colorbar(mappable=sm, cax=cax)
#plt.yticks([])
cbar.set_label('Line loading [%]')

f.set_size_inches( 5.24,  4.06)


#plt.savefig(r'c:\user\U546416\Pictures\Boriette\LineLoading\MaxLoading_{}fn.pdf'.format(case))
#plt.savefig(r'c:\user\U546416\Pictures\Boriette\LineLoading\MaxLoading_{}.jpg'.format(case), dpi=300)

fcenter  =['F00', 'F01','F02', 'F03', 'F04', 'F06', 'F07', 'F08','F09','F10']
# Farther buses
feeders = net.bus.Feeder.dropna().drop(0).unique()
busf = {f : net.bus[net.bus.Feeder == f].index[-1] 
            for f in feeders}
fdx = {'F00':(0,-0.0062),'F01':(-0.005,0)}


# adding feeder names
for f in feeders:
    if not (f in fcenter):
        ax.text(x=net.bus_geodata.loc[busf[f]].x,y=net.bus_geodata.loc[busf[f]].y, s=f)
# adding feeder to subplot
#for f in fcenter:
#    if f in fdx:
#        dx, dy = fdx[f]
#    else:
#        dx, dy = 0,0
#    a.text(x=net.bus_geodata.loc[busf[f]].x + dx,
#           y=net.bus_geodata.loc[busf[f]].y + dy, 
#           s=f)

#%% Plotting week-long line loading - reading data
# reading line loading
line_loadings = {}

cases = {'EV-only': 'evtou',
         'Continuity': 'evtou_pvcontinue',
#         'HighPenetration': 'evtou_pvrooftop',
         'Synergies': 'evopt_pvrooftop'}
for case, folder in cases.items():
    # reading results
    line_loadings[case] = pd.read_csv(mainfolder + folder + r'\line_loading.csv', index_col=0, engine='python')
#%% Plotting data
    
# Identifying key lines:
#lines = [84, 1032, 647]
lines = [84, 666]
labels= ['F06', 'F17']
#labels= ['F01', 'F16']
nstepsweek= 7*2*24
ww = 9
sw = 34
idxsww = line_loadings['EV-only'].index[(ww-1)*nstepsweek:ww*nstepsweek]
idxssw = line_loadings['EV-only'].index[(sw-1)*nstepsweek:sw*nstepsweek]
x = np.arange(0,24*7,0.5)

for j, idxs in enumerate([idxsww, idxssw]):
    f, axs=  plt.subplots(1,len(lines), sharey=True)
    for i,l in enumerate(lines):
        plt.sca(axs[i])
        for case in cases:
            plt.plot(x, line_loadings[case][str(l)][idxs], label=case)
            plt.title('({}) '.format('abc'[i]) + labels[i], y=-0.18)
        plt.xlim(x[0], x[-1])
        plt.ylim(0,110)
        plt.legend(loc=1)
        if i==0:
            plt.ylabel('Line loading [%]')
        plt.xticks(range(0,24*7, 24), util.dsnms, ha='left')

    plt.tight_layout()
    f.set_size_inches(7.57,3.3)
    plt.savefig(r'c:\user\U546416\Pictures\Boriette\LineLoading\LL{}_{}.pdf'.format(labels[0]+labels[1], 'summer' if j else 'winter'))
    
#%% Plotting week-long line loading - reading data
# reading line loading
volts = {}

cases = {'EV-only': 'evtou',
         'Continuity': 'evtou_pvcontinue',
#         'HighPenetration': 'evtou_pvrooftop',
         'Synergies': 'evopt_pvrooftop'}
for case, folder in cases.items():
    # reading results
    volts[case] = pd.read_csv(mainfolder + folder + r'\vm_pu.csv', index_col=0, engine='python')
    volts[case].columns = volts[case].columns.astype(int)
#%% Plotting data
nfs = [net.line[net.line.Feeder == f].index for f in feeders]
# Identifying key buses:

#buses = [102, 519, 691]
#labels= ['F06', 'F16', 'F17']
buses = [102, 691]
labels= ['F06', 'F17']
nstepsweek= 7*2*24
ww = 9
sw = 34
idxsww = volts['EV-only'].index[(ww-1)*nstepsweek:ww*nstepsweek]
idxssw = volts['EV-only'].index[(sw-1)*nstepsweek:sw*nstepsweek]
x = np.arange(0,24*7,0.5)

for j, idxs in enumerate([idxsww, idxssw]):
    f, axs=  plt.subplots(1,len(buses), sharey=True)
    for i,b in enumerate(buses):
        plt.sca(axs[i])
        for case in cases:
            plt.plot(x, volts[case][b][idxs], label=case)
            plt.title('({}) '.format('abc'[i]) + labels[i], y=-0.18)
        plt.xlim(x[0], x[-1])
        plt.ylim(0.93,1.08)
        plt.legend(loc=3, framealpha=1)
        if i==0:
            plt.ylabel('Voltage [pu]')
        plt.xticks(range(0,24*7, 24), util.dsnms, ha='left')
        plt.axhline(0.95, color='r', linestyle='--', linewidth=1)
        plt.axhline(1.05, color='r', linestyle='--', linewidth=1)

    plt.tight_layout()
    f.set_size_inches(7.57,3.3)
    plt.savefig(r'c:\user\U546416\Pictures\Boriette\Voltages\V{}_{}.pdf'.format(labels[0]+labels[1], 'summer' if j else 'winter'))    

#%% Compute % of nodes in constraints and % of time in constraints
    
thup = 1.05
thdn = 0.95

critvs = {}
nbusfs = np.array([net.bus[net.bus.Feeder==f].shape[0] for f in feeders])
for case in cases:
    overvolts_nodes = [(volts[case][net.bus[net.bus.Feeder==f].index].max(axis=0)>=thup).sum() for f in feeders]
    overvolts_time = [(volts[case][net.bus[net.bus.Feeder==f].index].max(axis=1)>=thup).sum() for f in feeders]
    undervolts_nodes = [(volts[case][net.bus[net.bus.Feeder==f].index].min(axis=0)<=thdn).sum() for f in feeders]
    undervolts_time = [(volts[case][net.bus[net.bus.Feeder==f].index].min(axis=1)<=thdn).sum() for f in feeders]
    critvs[case, 'ovn'] = np.array(overvolts_nodes)/nbusfs
    critvs[case, 'uvn'] = np.array(undervolts_nodes)/nbusfs
    critvs[case, 'ovt'] = np.array(overvolts_time)/8760/2
    critvs[case, 'uvt'] = np.array(undervolts_time)/8760/2
critvs = pd.DataFrame(critvs, index=feeders)
critvs.to_csv(mainfolder + 'voltage_dev_lim{}.csv'.format(thup))

#%% Computing and saving max loading per feeder:
feeders= ['F{:02d}'.format(i) for i in range(20)]
# reading line loading
line_loadings = {}

cases = {'base': 'base_noEV',
         'EV-only': 'evtou',
         'Continuity': 'evtou_pvcontinue',
         'HighPenetration': 'evtou_pvrooftop',
         'Synergies': 'evopt_pvrooftop'}
maxlineloadings = {}
for case, folder in cases.items():
    # reading results
    lls = pd.read_csv(mainfolder + folder + r'\line_loading.csv', index_col=0, engine='python')
    lls.columns = lls.columns.astype(int)
    maxlineloadings[case] = [lls[net.line[net.line.Feeder ==f].index].max().max() for f in feeders]
maxlineloadings = pd.DataFrame(maxlineloadings, index=feeders)
maxlineloadings.to_csv(mainfolder + 'maxfeederloading.csv')
