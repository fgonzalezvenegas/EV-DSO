    # -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:32:55 2019
testestest
@author: U546416
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import EVmodel as EV
import util
import time

times = [time.time()]
# DATA DATA
print('Loading data')
# Load data
# Histograms of Distance
print('Loading Histograms of distance')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal_SS.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal_SS.csv', 
                    engine='python', index_col=0)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
# Substation info
print('Loading SS')
folder_ssdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau'
SS = pd.read_csv(folder_ssdata + r'\postes_source.csv',
                                  engine='python', index_col=0)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
#SS_polys = pd.read_csv(folder_ssdata + r'/postes_source_polygons.csv', 
#                 engine='python', index_col=0)

# IRIS & Commune info
print('Loading IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
#print('Loading conso profiles')
#conso_profiles = pd.read_csv(folder_consodata + r'\SS_profiles.csv', 
#                    engine='python', index_col=0)
#times.append(time.time())
#print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))

# Histograms of arrival/departures
print('Arrival departures')
folder_arrdep = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité'
arr_dep = pd.read_csv(folder_arrdep + r'\DepartureArrivals.csv', 
                    engine='python', index_col=0)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))


#%% Run for one SS
import EVmodel as EV

# SIMULATION DATA (TIME)
# Days, steps
ndays = 21
step = 15

# GENERAL EV DATA

times = [time.time()]
# EV penetration
ev_penetration = .5
# EV home/work charging
ev_work_ratio = 0.3
# EV charging params (charging power, batt size, etc)
charging_power_home = [[3.6, 7.2, 11], [0.5, 0.4, 0.1]]
charging_power_work = [[3.6, 7.2, 11, 22], [0.0, 0.2, 0.5, 0.3]]
batt_size = [[20, 40, 60, 80], [0.20, 0.30, 0.30, 0.20]]

ss = 'BORIETTE'
iris_ss = iris[iris.SS==ss]

# Number of Evs
nevs_h = int(iris_ss.N_VOIT.sum() * ev_penetration * (1-ev_work_ratio))
nevs_w = int(hwork.loc[ss].sum() * ev_penetration * ev_work_ratio * 1.78) # 1.78 is the ratio between nation-wide Work EVs and Total EVs  
print('EVs Overnight', nevs_h)
print('EVs Work', nevs_w)

# Distributions of work and home distances
params_h = util.compute_lognorm_cdf(hhome.loc[ss], params=True)
params_w = util.compute_lognorm_cdf(hwork.loc[ss], params=True)

# Arrival and departure hourly cdfs
arr_dep_data_h = {'cdf_arr': arr_dep.ArrHome.cumsum(), 'cdf_dep': arr_dep.DepHome.cumsum()}
arr_dep_data_w = {'cdf_arr': arr_dep.ArrWork.cumsum(), 'cdf_dep': arr_dep.DepWork.cumsum()}

# compute base load for worst week, adding 7 days of buffer on each side
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\SS_profiles\\'
conso_profile = pd.read_csv(folder_profiles + ss + '.csv', engine='python', index_col=0)
load = util.get_max_load_week(conso_profile.squeeze(), buffer_before=7, buffer_after=7)



grid = EV.Grid(name=ss, ndays=ndays, step=step, load=load, ss_pmax=SS.Pmax[ss])
grid.add_evs('Overnight', nevs_h, ev_type='dumb',
             dist_wd=params_h, 
             charging_power=charging_power_home,
             charging_type='if_needed',
             batt_size=batt_size,
             arrival_departure_data_wd=arr_dep_data_h)
grid.add_evs('Day', nevs_w, ev_type='dumb',
             dist_wd=params_h,
             dist_we={'s':0.8,'loc':0,'scale':2.75},
             charging_power=charging_power_work,
             charging_type='weekdays',
             batt_size=batt_size,
             arrival_departure_data_wd=arr_dep_data_w)
times.append(time.time())
print('Finished preprocessing, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
grid.do_days()
times.append(time.time())
print('Finished running, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
print(grid.get_global_data())
print(grid.get_ev_data())
grid.plot_total_load(day_ini=7, days=7)
grid.plot_ev_load(day_ini=7, days=7)


#%% Run for all SS

dir_results = r'c:\user\U546416\Documents\PhD\Data\Simulations\Results\RandStart_EV05_W01'

# SIMULATION DATA (TIME)
# Days, steps
ndays = 15
step = 15

# GENERAL EV DATA

times = [time.time()]
# EV penetration
ev_penetration = .5
# EV home/work charging
ev_work_ratio = 0.1
# EV charging params (charging power, batt size, etc)
charging_power_home = [[3.6, 7.2, 11], [0.5, 0.4, 0.1]]
charging_power_work = [[3.6, 7.2, 11, 22], [0.0, 0.2, 0.5, 0.3]]
batt_size = [[20, 40, 60, 80], [0.20, 0.30, 0.30, 0.20]]

# Arrival and departure hourly cdfs
arr_dep_data_h = {'cdf_arr': arr_dep.ArrHome.cumsum(), 'cdf_dep': arr_dep.DepHome.cumsum()}
arr_dep_data_w = {'cdf_arr': arr_dep.ArrWork.cumsum(), 'cdf_dep': arr_dep.DepWork.cumsum()}

ev_type = 'randstart'

#Results

global_data = {}
ev_data = {}
ev_load_day = {}
ev_load_night = {}

#%%
f1, ax1 = plt.subplots()
f1.set_size_inches(7,6)
f2, ax2 = plt.subplots()
f2.set_size_inches(7,6)
counter = 0
for ss in SS.index:
    counter += 1
    if ss in global_data:
        continue
    print('\n\n {}: {}\n'.format(counter, ss))
    iris_ss = iris[iris.SS==ss]
    if len(iris_ss) == 0:
        continue
    # Number of Evs
    nevs_h = int(iris_ss.N_VOIT.sum() * ev_penetration * (1-ev_work_ratio))
    nevs_w = int(hwork.loc[ss].sum() * ev_penetration * ev_work_ratio * 1.78) # 1.78 is the ratio between nation-wide Work EVs and Total EVs  
    print('EVs Overnight', nevs_h)
    print('EVs Work', nevs_w)
    if (nevs_h==0) or (nevs_w==0):
        continue
    # Distributions of work and home distances
    params_h = util.compute_lognorm_cdf(hhome.loc[ss], params=True)
    params_w = util.compute_lognorm_cdf(hwork.loc[ss], params=True)
    

    # compute base load for worst week, adding 7 days of buffer on each side
    folder_profiles = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\SS_profiles\\'
    conso_profile = pd.read_csv(folder_profiles + ss + '.csv', engine='python', index_col=0)
    load = util.interpolate(util.get_max_load_week(conso_profile.squeeze(), buffer_before=7, buffer_after=1, extra_t=1), step=step)
    load = load.iloc[0:-1]
    
    
    grid = EV.Grid(name=ss, ndays=ndays, step=step, load=load, ss_pmax=SS.Pmax[ss], verbose=False)
    grid.add_evs('Overnight', nevs_h, ev_type=ev_type,
                 dist_wd=params_h, 
                 charging_power=charging_power_home,
                 charging_type='if_needed',
                 batt_size=batt_size,
                 arrival_departure_data_wd=arr_dep_data_h)
    grid.add_evs('Day', nevs_w, ev_type=ev_type,
                 dist_wd=params_h,
                 dist_we={'s':0.8,'loc':0,'scale':2.75},
                 charging_power=charging_power_work,
                 charging_type='weekdays',
                 batt_size=batt_size,
                 arrival_departure_data_wd=arr_dep_data_w)
    times.append(time.time())
    print('Finished preprocessing, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
    grid.do_days()
    times.append(time.time())
    print('Finished running, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
    global_data[ss] = grid.get_global_data()
    ev_data[ss] = grid.get_ev_data()
    ev_load_day[ss] = grid.ev_load['Day'][int(60/step*24*7):int(60/step*24*14)]
    ev_load_night[ss] = grid.ev_load['Overnight'][int(60/step*24*7):int(60/step*24*14)]
    
    ax1.clear()
    ax2.clear()
    grid.plot_total_load(day_ini=7, days=7, ax=ax1, title='Total load at ' + ss)
    grid.plot_ev_load(day_ini=7, days=7, ax=ax2, title='EV load at ' + ss)
    f1.savefig(dir_results + r'\Images\Total_{}.png'.format(ss))
    f2.savefig(dir_results + r'\Images\EV_{}.png'.format(ss))
    print('Total SS time: {} s'.format(np.round(times[-1]-times[-3],1)))
    print('Total elapsed time: {} s'.format(np.round(times[-1]-times[0],1)))

# Transform outputs in DataFrames and save
global_data = pd.DataFrame(global_data).T
ev_data = {ss : {el + evtype : ev_data[ss][el] 
                    for evtype in ev_data[ss]['EV_sets'] 
                    for el in ev_data[ss] if el != 'EV_sets'} 
        for ss in ev_data}
ev_data = pd.DataFrame(ev_data).T

ev_load_day = pd.DataFrame(ev_load_day, index=load.index[0:int(7*24*60/step)])
ev_load_night = pd.DataFrame(ev_load_night, index=load.index[0:int(7*24*60/step)])
global_data.to_csv(dir_results + r'\global_data.csv')
ev_data.to_csv(dir_results + r'\ev_data.csv')
ev_load_day.to_csv(dir_results + r'\ev_load_day.csv')
ev_load_night.to_csv(dir_results + r'\ev_load_night.csv')