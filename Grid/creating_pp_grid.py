# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:02:09 2020
Pandapower !
@author: U546416
"""

#import mobility as mb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe
import util
import pandapower as pp
import pandapower.topology as ppt
import pandapower.plotting as ppp
import pandapower.control as ppc
import pandapower.timeseries as ppts

import time as time
import datetime as dt

from ppgrid import *
from grid import *

#%% OPTIONAL: Load processed data
print('Loading MV grid data')
folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\\'
subf = r'Data_Boriette_2020-06-28\\'
lines = pd.read_csv(folder + subf + 'MVLines.csv', engine='python', index_col=0)
ss = pd.read_csv(folder + subf + 'SS.csv', engine='python', index_col=0)
lv = pd.read_csv(folder + subf + 'MVLV.csv', engine='python', index_col=0)
nodes = pd.read_csv(folder + subf + 'Nodes.csv', engine='python', index_col=0)

lines.ShapeGPS = lines.ShapeGPS.apply(eval)
nodes.ShapeGPS = nodes.ShapeGPS.apply(eval)


folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles\\'
profiles_load = pd.read_csv(folder_profiles  + r'profiles_iris.csv',
                       engine='python', index_col=0)

print('Loading tech data')
folder_tech = r'c:\user\U546416\Documents\PhD\Data\MVGrids\\'
file_tech = 'line_data_France_MV_grids.xlsx'
tech = pd.read_excel(folder_tech + file_tech, index_col=0)


#%% OPTIONAL: Load IRIS polygon data
print('Loading IRIS polygons')
# TODO: Load IRIS polygons
folder_iris = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
iris_poly = pd.read_csv(folder_iris+file_iris,
                        engine='python', index_col=0)
iris_polygons = pd.Series(util.do_polygons(iris_poly, plot=False))
print('\tDone loading polygons')
#%% OPTIONAL: Load conso data
print('Loading Conso per IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
print('Loading profiles')
profiles_all = pd.read_csv(folder_consodata + r'\conso_all_pu.csv',
                       engine='python', index_col=0)
profiles_all.drop(['ENT', 'NonAffecte'], axis=1, inplace=True)

#%% Load pv data per departement
folder_pv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau'
dep_parc_prod = pd.read_csv(folder_pv + r'\parc-pv-departement.csv',
                            engine='python', sep=',')
pv_dep = dep_parc_prod.groupby(['DEP_CODE', 'TYPE_PV']).P_MW.sum()

#%% showing data
n0 = ss.node.iloc[0]
#dep = 19
polys = iris_poly[['IRIS_NAME', 'Polygon', 'Lon', 'Lat']]
polys.columns = ['Name', 'Polygon', 'xGPS', 'yGPS']
polys.Polygon = iris_polygons

off = on_off_lines(lines, n0, ss=ss, lv=lv, GPS=True, geo=polys, tech=tech, nodes=nodes)


#%% Transforming data in pandapower DataFrames
v = util.input_y_n('Do you want to create the grid from loaded data (Y) or load existing grid (N):')
    
if v in ['Y', 'y', True]:
    print('\tCreating grids')
    n0 = ss.node.iloc[0]
    rename_nodes(nodes, n0, lines, lv, ss)
    n0 = 0
    rename_lines(lines, n0)
    #%% Create grid
    print('\t\tCreating base grid')
    net = create_pp_grid(nodes, lines, tech, lv, n0=0, 
                        hv=True, ntrafos_hv=2, vn_kv=20,
                        tanphi=0.3, verbose=True)
    
    bext = net.ext_grid.bus[0]
    trafo_ss = net.trafo[net.trafo.hv_bus == bext].index[0]
    
    # Add tap changer controller at MV side of SS trafo
    ppc.DiscreteTapControl(net, trafo_ss, 0.99, 1.01, side='lv')
    
    # Check connectedness
    # Check unsupplied buses from External grid
    ub = ppt.unsupplied_buses(net)
    
    if len(ub) == 0:
        print('Connectedness ok')
    else:
        print('There are Non supplied buses!')
    
    print('Running')
    t = time.time()
    pp.runpp(net, run_control=True)
    print('Run! dt={:.2f}'.format(time.time()-t))
    plot_v_profile(net)
    plt.title('Voltage profile')
    
    # TODO: Find out why time series of Pandapower doesnt work
    
    #%% Create Net with RES
    print('\t\tCreating RES grid')
    net_res = create_pp_grid(nodes, lines, tech, lv, n0=0, 
                            hv=True, ntrafos_hv=2, vn_kv=20,
                            tanphi=0.3, verbose=True)
    
    bext = net_res.ext_grid.bus[0]
    trafo_ss = net_res.trafo[net_res.trafo.hv_bus == bext].index[0]
    
    # Add tap changer controller at MV side of SS trafo
    ppc.DiscreteTapControl(net_res, trafo_ss, 0.99, 1.01, side='lv')
    # IRIS in the net
    ies = net_res.load.name.unique()
    
    if not 'iris' in net_res.bus:
        if not 'Geo' in nodes:
            assign_polys(nodes, polys)
        net_res.bus['iris'] = nodes.Geo
        
    # ADDING PV
    # Region of net
    reg = int(iris.REGION_CODE[ies].unique()[0])
    ies_reg = iris[iris.REGION_CODE == reg].index
    # load # BT/iris:
    folder_lv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
    lv_iris = pd.read_csv(folder_lv+'Nb_BT_IRIS2016.csv',
                          engine='python', index_col=0)
    # PV growth factor
    national_target = 35000 # MW of PV installed
    current_pv = dep_parc_prod.P_MW.sum() * 1.2 # times 1.2 because i only have Enedis data
    growth = national_target / current_pv
    
    # Regional PV targets [MW]
    pv_reg = dep_parc_prod.groupby(['REG_CODE', 'TYPE_PV']).P_MW.sum() * growth
    
    # RESIDENTIAL
    # Computing penetration of RES PV
    pv_res_cap = 3 #kW
    nb_res_reg = (iris.Nb_RES[ies_reg] * (100 - iris.Taux_logements_collectifs[ies_reg])/100).sum()
    RES_penetration = pv_reg[reg, 'home_rooftop'] / (pv_res_cap/1000 * nb_res_reg)

    # Add home rooftop PV
    add_res_pv_rooftop(net_res, pv_penetration=RES_penetration, 
                       iris_data=iris, lv_per_geo=lv_iris.Nb_BT, pv_cap_kw=pv_res_cap)
    
    # COMMERCIAL
    # Add commercial rooftop PV
    pv_comm_cap = 0.120 #MW
    # Computing potential sites per region and in the SS
    # Parameters for proxy of commercial sites
    tertiary_threshold = 600 #MWh/year
    industry_threshold = 600 #MWh/year
    agri_multiplier = 2
    # Potential sites per iris
    ncomm = ((iris.Conso_Tertiaire // tertiary_threshold).round(decimals=0) + 
             (iris.Conso_Industrie // industry_threshold).round(decimals=0) + 
             iris.Nb_Agriculture * agri_multiplier)
    # target MW of SS based on Ratio of potential sites in SS vs region
    ncomm_ss = ncomm[ies] * net_res.load.name.value_counts() / lv_iris.Nb_BT[ies]
    target_comm = pv_reg[reg, 'commercial_rooftop'] * ncomm_ss.sum() / ncomm[ies_reg].sum()
    
    # Random buses to be selected to host commercial PV
    buses = np.concatenate([np.random.choice(net_res.load[net_res.load['name'] == i].bus, 
                                             size=int(round(ncomm_ss[i],0)))
                            for i in ncomm_ss.index])      
    add_random_pv_rooftop(net_res, mean_size=pv_comm_cap, std_dev=pv_comm_cap/6, 
                          total_mw=target_comm, buses=buses, replace=True,
                          name='Comm_PV_', sgtype='Comm_PV')
    # Add solar farms
    # Target MW of solar farms based on ratio of Rural communes / total communes in the region
    pv_farm_cap = 2 #MW
    ncommunes_region = len(iris.loc[ies_reg][iris.IRIS_TYPE == 'Z'].COMM_CODE.unique())
    # We'll put the solar farms far from the center of the city, only in rural IRIS
    ies_rural = iris.loc[ies][iris.IRIS_TYPE == 'Z'].index
    ncommunes_SS = len(iris.COMM_CODE[ies_rural].unique())
    target_farms = pv_reg[reg, 'solar_farm'] * ncommunes_SS / ncommunes_region
    # Selecting buses of rural communes        
    buses = net_res.bus[net_res.bus.iris.isin(ies_rural)].index
    
    add_random_pv_rooftop(net_res, mean_size=pv_farm_cap, std_dev=pv_farm_cap/6, total_mw=target_farms,
                          buses=buses, replace=True, name='Solar_PV_', sgtype='Farm_PV')

#%% Alternative Load existing grid:
else:
    folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
    print('\t loading grids from folder {}'.format(folder_grids))
    # load predefined grid
    folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
    net = pp.from_json(folder_grids + 'base_grid.json')
    net_res = pp.from_json(folder_grids + 'res_grid.json')
    #net_res_ev = pp.from_json(folder_grids + 'res_ev_grid.json')

#%% Compute PV inst cap per IRIS // energy produced

ies = net_res.load.name.unique()

# Compute installed RES per IRIS
conso_prod_iris = {}
for i in net_res.bus.iris.unique():
    if not np.isnan(i):
        bss = net_res.bus[net_res.bus.iris == i].index
        conso_prod_iris[int(i)] = [net_res.load[net_res.load.bus.isin(bss)].p_mw.sum(),
                                    net_res.sgen[(net_res.sgen.type == 'RES_PV') & (net_res.sgen.bus.isin(bss))].p_mw.sum(),
                                    net_res.sgen[(net_res.sgen.type == 'Comm_PV') & (net_res.sgen.bus.isin(bss))].p_mw.sum(),
                                    net_res.sgen[(net_res.sgen.type == 'Farm_PV') & (net_res.sgen.bus.isin(bss))].p_mw.sum()] 
conso_prod_iris = pd.DataFrame(conso_prod_iris, index=['Conso_MWh', 'PV_Home_MW','PV_Commercial_MW', 'PV_farm_MW']).T      

# Plot installed PV per IRIS
pls = util.list_polygons(iris_polygons, ies)
f, axs = plt.subplots(1,3)
cmap = plt.get_cmap('plasma')
for i, ax in enumerate(axs):
    maxpv = conso_prod_iris.iloc[:,1:].max().max()
    colors = cmap(conso_prod_iris.iloc[:,i+1].loc[ies]/maxpv)
    util.plot_polygons(pls, ax=ax, color=colors, alpha=0.8)
    plot_lines(lines, col='ShapeGPS', ax=ax, color='k', linewidth=0.5)
    ax.set_title('{}\nTotal {:.1f} MW'.format(conso_prod_iris.columns[i+1], conso_prod_iris.iloc[:,i+1].sum()))
# TODO: add legend
dv_pv = np.round(maxpv/0.5,0)*0.5
tranches = np.arange(0,maxpv+0.1, maxpv/7)
labels = ['{:.2f} MW'.format(i) for i in tranches]
colors = cmap(tranches/maxpv)
util.do_labels(labels, colors, ax, f=True)
#f.title('PV installed capacity per IRIS')

# Plot type IRIS
c_t = {'A':'r', 'D':'gray', 'Z':'g', 'H':'b'}
colors = [c_t[iris_poly.IRIS_TYPE[i]] for i in ies]
ax=util.plot_polygons(pls, color=colors)
plot_lines(lines, col='ShapeGPS', ax=ax, color='k', linewidth=0.5)
util.do_labels(['Activité', 'Divers', 'Rural', 'Habitation'], list(c_t.values()), ax=ax)
plt.title('Type IRIS')

# Plot share of PV per feeder
plt.subplots()
fs = [f for f in net_res.bus.zone.unique() if (f==f and (not (f is None)))]
fs = np.sort(fs)
load_pv_feeder = {}
for f in fs:
    bs = net_res.bus[net_res.bus.zone == f].index
    load_pv_feeder[f] = {'load': net_res.load[net_res.load.bus.isin(bs)].p_mw.sum(),
                  'PV_gen': net_res.sgen[net_res.sgen.bus.isin(bs)].p_mw.sum()}
load_pv_feeder = pd.DataFrame(load_pv_feeder).T

plt.pie(load_pv_feeder.PV_gen, labels=load_pv_feeder.index)
plt.title('Share of total PV installed capacity among feeders. \nTotal {:.2f} MW'.format(load_pv_feeder.PV_gen.sum()))
plt.subplots()
plt.bar(load_pv_feeder.PV_gen.index, load_pv_feeder.PV_gen)
plt.title('Installed PV capacity per feeder. \nTotal {:.2f} MW'.format(load_pv_feeder.PV_gen.sum()))
plt.xticks(rotation=90)
plt.ylabel('PV capacity [MW]')


#plt.legend()
colors = ['r', 'g', 'b', 'gray', 'c', 'y', 'm', 'darkblue', 'purple', 'brown', 
          'maroon', 'olive', 'deeppink', 'blueviolet', 'darkturquoise', 'darkorange']
f,ax = plt.subplots()
for i, f in enumerate(fs):
    if f[0]=='F':
        plot_lines(lines[lines.Feeder==f], color=colors[i%len(colors)], linewidth=load_pv_feeder.PV_gen[f]/1, 
                   label=f, col='ShapeGPS', ax=ax)
plt.legend()
## Saving grids
v = util.input_y_n('Do you want to save grids created grids?')
if v in ['Y', 'y', True]:
    pp.to_json(net, folder + r'PPGrid/base_grid.json')
    pp.to_json(net_res, folder + r'PPGrid/res_grid.json')

#%% Run base grid time series
v = util.input_y_n('Run base grid time series?')
if v in ['Y', 'y', True]:
    # Creating profiler
    profiler = Profiler(net=net, profiles_db=profiles_load)
    for n in net.load.name.unique():
        if not n is None:
            profiler.add_profile_idx('load', net.load[net.load.name==n].index, variable='scaling', profile=str(n))
    # Setting iterator
    time_steps=profiler.profiles_db.index   
    iterator = Iterator(net=net, profiler=profiler)
    of = folder + r'\Results_Base'
    iterator.iterate(time_steps=time_steps, save=True, outputfolder=of, ultraverbose=False)

    #%% Plot some results - base case
    
    # critical V:
    farther_bus = pd.Series(index=[i for i in net.bus.zone.unique() if not i is None])
    d = dist_to_node_nx(lines, n0)
    for i in farther_bus.index:
        farther_bus[i] = int(d[net.bus[net.bus.zone == i].index].idxmax())
    # Critical feeders
    fs = ['0SS', 'F06', 'F16', 'F17', 'F18', 'F19']
    critday = iterator.ow.res_v.min(axis=1).idxmin()
    
    # Net load at transformer
    f, ax = plt.subplots()
    ax.plot(iterator.ow.global_res.TrafoOut_MW[::2])
    plt.xticks(np.arange(0,8760,8761/6), rotation=45)
    ax.set_xticklabels(['jan', 'mar', 'may', 'jul', 'sep', 'nov'])
    plt.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
    plt.xlim(0,8760)
    plt.title('Net load at transformer')
    
    
    # plot V whole year
    f, ax = plt.subplots()
    for f in fs:
        plt.plot(iterator.ow.res_v[farther_bus[f]][::2], label=f)
    plt.legend()
    
    plt.xticks(np.arange(0,8760,8761/6), rotation=45)
    ax.set_xticklabels(['jan', 'mar', 'may', 'jul', 'sep', 'nov'])
    plt.xlim(0,8760)
    plt.title('Yearly voltage at selected buses')
    
    # Plot V profiles
    # Set critical day as res
    critday = iterator.ow.res_v.min(axis=1).idxmin()
    net.res_bus.vm_pu = iterator.ow.res_v.loc[critday]
    f, ax = plt.subplots()
    plot_v_profile(net, ax=ax)
    plt.title('Voltage at critical hour, {}'.format(critday))
    
    critpower = iterator.ow.global_res.TotLoad_MW.idxmax()
    net.res_bus.vm_pu = iterator.ow.res_v.loc[critpower]
    f, ax = plt.subplots()
    plot_v_profile(net, ax=ax)
    plt.title('Voltage at max load, {}'.format(critpower))

#%% Adding profiles RES

# loading solar profiles => Download from renewables.ninja
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles\\'
prof_solar_farm = pd.read_csv(folder_profiles + 'solar_farm.csv',
                              engine='python', skiprows=3)
prof_solar_roof = pd.read_csv(folder_profiles + 'solar_roof.csv',
                              engine='python', skiprows=3)
# set as 30min series
profiles_pv = pd.DataFrame(index=profiles_load.index, columns=['solar_roof', 'solar_farm'], dtype=float)
for i, t in prof_solar_roof.iterrows():
    profiles_pv.iloc[2*i, 0] = t.electricity
    profiles_pv.iloc[2*i, 1] = prof_solar_farm.electricity.iloc[i]
for col in profiles_pv.columns:
    profiles_pv[col] = profiles_pv[col].interpolate(method='linear')

# Creating profiler
profiler_res = Profiler(net=net_res, profiles_db=profiles_load)
for n in net_res.load.name.unique():
    if not n is None:
        profiler_res.add_profile_idx('load', net_res.load[net_res.load.name==n].index, variable='scaling', profile=str(n))

# Adding profile database
profiler_res.add_profile_db(profile=profiles_pv)
# Adding controller
sroof_idx = net_res.sgen[net_res.sgen.p_mw<0.5].index
sfarm_idx = net_res.sgen[net_res.sgen.p_mw>0.5].index
profiler_res.add_profile_idx(element='sgen', idx=sroof_idx, variable='scaling', profile='solar_roof')
profiler_res.add_profile_idx(element='sgen', idx=sfarm_idx, variable='scaling', profile='solar_farm')
    
    
#%% Setting iterator and run simulation for RES net
v = util.input_y_n('Run RES grid time series?')
if v in ['Y', 'y', True]:
    time_steps=profiler_res.profiles_db.index
    iterator = Iterator(net=net_res, profiler=profiler_res)
    folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\\'
    of = folder + r'\Results_PV'
    # Run
    iterator.iterate(time_steps=time_steps, save=True, outputfolder=of, ultraverbose=False)

    #%% Plotting some results
    
    # critical V:
    # Critical buses
    farther_bus = pd.Series(index=[i for i in net.bus.zone.unique() if not i is None])
    d = dist_to_node_nx(lines, n0)
    for i in farther_bus.index:
        farther_bus[i] = int(d[net.bus[net.bus.zone == i].index].idxmax())
    # Critical feeders
    fs = ['0SS', 'F06', 'F16', 'F17', 'F18', 'F19']
    critday = iterator.ow.res_v.min(axis=1).idxmin()
    
    # plot whole year
    f, ax = plt.subplots()
    for f in fs:
        plt.plot(iterator.ow.res_v[farther_bus[f]][::2], label=f)
    plt.legend()
    
    plt.axhline(y=1.05, linestyle='--', color='r')
    plt.axhline(y=0.95, linestyle='--', color='r')
    plt.xticks(np.arange(0,8760,8761/6), rotation=45)
    ax.set_xticklabels(['jan', 'mar', 'may', 'jul', 'sep', 'nov'])
    plt.xlim(0,8760)
    plt.title('Yearly voltage at selected buses')
    
    
    # Net load at transformer
    f, ax = plt.subplots()
    ax.plot(iterator.ow.global_res.TrafoOut_MW[::2])
    plt.xticks(np.arange(0,8760,8761/6), rotation=45)
    ax.set_xticklabels(['jan', 'mar', 'may', 'jul', 'sep', 'nov'])
    plt.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
    plt.xlim(0,8760)
    plt.title('Net load at transformer')
    
    # Plot reverse flow duration histogram
    if sum(iterator.ow.global_res.TrafoOut_MW<0) > 0:
        f, ax = plt.subplots()
        hist, bins = util.hist_ovl(-iterator.ow.global_res.TrafoOut_MW, 0, 24)
        plt.bar(bins[1:]/2, hist, width=0.5)
        plt.title('Histogram of reverse flow events duration')
        plt.xlabel('Reverse flow duration [h]')
        plt.ylabel('Number of events')
        
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
    
    maxv = iterator.ow.res_v.max(axis=1).idxmax()
    net.res_bus.vm_pu = iterator.ow.res_v.loc[maxv]
    f, ax = plt.subplots()
    plot_v_profile(net, ax=ax)
    plt.title('Voltage profile at minimum Vpu, {}'.format(maxv))

