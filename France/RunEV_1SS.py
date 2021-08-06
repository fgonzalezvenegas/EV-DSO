    # -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:32:55 2019
testestest
@author: U546416
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import EVmodel
import util
import time

times = [time.time()]
# DATA DATA
print('Loading data')
# Load data
# Histograms of Distance
print('Loading Histograms of distance')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
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
arr_dep_pdf = pd.read_csv(folder_arrdep + r'\Arr_Dep_pdf.csv', 
                          engine='python', index_col=0)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))

if 'ZE' in hwork.columns:
    hwork = hwork.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
    hhome = hhome.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)

util.plot_arr_dep_hist
#%%
"""
Substations:
    SEVRES - Paris, low plugin
    TAILLIS - PeriUrban, high plugin, high driving

"""

#%% Run for one SS
util.self_reload(EVmodel)
import EVmodel

# SIMULATION DATA (TIME)
# Days, steps
ndays = 22
step = 15

# GENERAL EV DATA

times = [time.time()]
# EV penetration
ev_penetration = .30
# EV home/work charging
ev_work_ratio = 0.3
# EV charging params (charging power, batt size, etc)
#charging_power_home = [[3.6, 7.2, 11], [0.5, 0.4, 0.1]]
#charging_power_work = [[3.6, 7.2, 11, 22], [0.0, 0.2, 0.5, 0.3]]
#batt_size = [[20, 40, 60, 80], [0.20, 0.30, 0.30, 0.20]]
charging_power_home=3.6
charging_power_work=7.2
batt_size = 50

ss = 'BORIETTE'
iris_ss = iris[iris.SS==ss]

tou = False
h_tous = 10
start_tous = 21
end_tous = 3
delta_tous = (end_tous - start_tous) % 24



# Distributions of work and home distances
#params_h = util.compute_lognorm_cdf(hhome.loc[ss], params=True)
#params_w = util.compute_lognorm_cdf(hwork.loc[ss], params=True)

# Arrival and departure hourly cdfs
arr_dep_data_h = dict(cdf_arr = arr_dep.ArrHome.cumsum(),
                      cdf_dep = arr_dep.DepHome.cumsum())
#arr_dep_data_w = dict(cdf_arr = arr_dep.ArrWork.cumsum(), 
#                      cdf_dep = arr_dep.DepWork.cumsum())
arr_dep_data_w = dict(pdf_a_d = arr_dep_pdf.values)

# Parameters
general_params = dict(batt_size = batt_size)

# overnight params
overnight_params = dict(charging_power = charging_power_home,
                        arrival_departure_data_wd = arr_dep_data_h,
                        charging_type = 'if_needed',
#                        charging_type = 'all_days',
                        alpha = 0,
                        tou_we = False)

# day params
day_params = dict(charging_power = charging_power_work,
                        arrival_departure_data_wd = arr_dep_data_w,
                        charging_type = 'weekdays',
                        alpha = 1,
                        tou_we = False,
                        dist_we= dict(s=0.8, loc=0, scale=2.75),
                        pmin_charger= 0.8)


# compute base load for worst week, adding 7 days of buffer on each side
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\SS_profiles\\'
conso_profile = pd.read_csv(folder_profiles + ss + '.csv', engine='python', index_col=0)
load = util.interpolate(util.get_max_load_week(conso_profile.squeeze(), buffer_before=7, buffer_after=8, extra_t=1), step=step)[:-1]


#%% Create Grid
grid = EVmodel.Grid(name=ss, ndays=ndays, step=step, load=load, ss_pmax=SS.Pmax[ss])
# Add EVs for each IRIS
iriss = [190630000, 190720000] #Cosnac Donzenac
#for i in iris_ss.index:
for i in iriss:
    # Compute # of EVs 
    # Number of Evs
#    comm = iris_ss.COMM_CODE[i]
    comm  = int(i)//10000
    nevs_h = int(iris_ss.N_VOIT[i] * ev_penetration * (1-ev_work_ratio))
    nevs_w = int(hwork.loc[comm].sum() * iris_ss.Work_pu[i] * # First term is total work EVs in the commune, second is the ratio of Workers in the iris
                 ev_penetration * ev_work_ratio * 1.78) # 1.78 is the ratio between nation-wide Work EVs and Total EVs  
    print('EVs Overnight', nevs_h)
    print('EVs Work', nevs_w)
    
    
    # Add EVs
    grid.add_evs('Overnight_' + str(i) , nevs_h, ev_type='dumb',
                 #pmin_charger=0.1,
                 dist_wd=dict(cdf = hhome.loc[comm].cumsum()/hhome.loc[comm].sum()),
                 **general_params,
                 **overnight_params)
    if tou:
            print('\tToU-ing')
            t0 = time.time()
            for ev in grid.evs['Overnight_' + str(i)]:
                ev.tou_ini = np.round((start_tous + np.random.rand(1) * delta_tous) % 24, 2)
                ev.tou_end = (ev.tou_ini + h_tous) % 24
                ev.set_off_peak(grid)
            print('\tFinished ToU-ing, elapsed time: {} s'.format(np.round(time.time()-t0,1)))
    
    grid.add_evs('Day_' + str(i), nevs_w, ev_type='mod',
                 dist_wd= dict(cdf = hwork.loc[comm].cumsum()/hwork.loc[comm].sum()),
                 **general_params,
                 **day_params)

times.append(time.time())
print('Finished preprocessing, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
#%% Simulation
grid.do_days()
times.append(time.time())
print('Finished running, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
global_data = grid.get_global_data()
ev_data = grid.get_ev_data()
grid.plot_total_load(day_ini=7, days=7)
grid.plot_ev_load(day_ini=7, days=7)

print('EV mean dist')
for t in grid.ev_sets:
    print(t, 'Wd: ', np.mean([ev.dist_wd for ev in grid.evs[t]]), 
          ';  Weekend: ', np.mean([ev.dist_we for ev in grid.evs[t]]))

print('EV plug in')
for t in grid.ev_sets:
    print(t, 'Wd: ', np.mean([ev.ch_status.sum() for ev in grid.evs[t]]))


#%% Get EV data and save
init_idx = int(7 * 24 *60 / step)
end_idx =  int(1 * 24 *60 / step)
idx = load.iloc[init_idx:-end_idx].index
evdata = {}
for t in grid.evs:
    evdata[t] = grid.ev_load[t][init_idx:-end_idx]
evdata = pd.DataFrame(evdata, index=idx)
if tou:
    toustr = 'TOU'
else:
    toustr = 'Uncontrolled'
    
name = ss + '_EV' + str(int(ev_penetration*100)) + '_' + toustr + '.csv'
folder = r'C:\Users\u546416\Downloads\tosend\Cosnac_Donzenac\\'

iriss = ['190630000', '190720000']
cols = [d + i for i in iriss for d in ['Day_', 'Overnight_']]

evdata[cols].to_csv(folder + name)
