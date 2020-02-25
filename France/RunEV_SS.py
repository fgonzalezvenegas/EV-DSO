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


#%% Run for all SS

dir_results = r'c:\user\U546416\Documents\PhD\Data\Simulations\Results'



# SIMULATION DATA (TIME)
# Days, steps
ndays = 15
step = 15

# GENERAL EV DATA


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

ev_type = 'mod'
tou = False
# available: dumb, mod, randstart, reverse

# Results folder
outputfolder = '{}{}_EV{:02d}_W{:02d}'.format(ev_type, '_ToU' if tou else '',int(ev_penetration*10), int(ev_work_ratio*10))

# Check that results folder exists:
util.create_folder(dir_results, outputfolder, 'Images')

#%% Results

global_data = {}
ev_data = {}
ev_load_day = {}
ev_load_night = {}



#%% Run for multiple ToU overnights


f1, ax1 = plt.subplots()
f1.set_size_inches(7,6)
f2, ax2 = plt.subplots()
f2.set_size_inches(7,6)
counter = 0
# To set only one overnight ToU: Set start_tous = end_tous
# to set full off-peak day (no ToU): Set everything to 0
h_tous = 6
start_tous = 22
end_tous = 3
delta_tous = (end_tous - start_tous) % 24
times = [time.time()]
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
#    params_h = util.compute_lognorm_cdf(hhome.loc[ss], params=True)
#    params_w = util.compute_lognorm_cdf(hwork.loc[ss], params=True)

    # compute base load for worst week, adding 7 days of buffer on each side
    folder_profiles = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\SS_profiles\\'
    conso_profile = pd.read_csv(folder_profiles + ss + '.csv', engine='python', index_col=0)
    load = util.interpolate(util.get_max_load_week(conso_profile.squeeze(), buffer_before=7, buffer_after=1, extra_t=1), step=step)
    load = load.iloc[0:-1]
    
    
    grid = EVmodel.Grid(name=ss, ndays=ndays, step=step, load=load, ss_pmax=SS.Pmax[ss], verbose=False)
    grid.add_evs('Overnight', nevs_h, 
                 ev_type=ev_type,
                 dist_wd={'cdf': hhome.loc[ss].cumsum()/hhome.loc[ss].sum()}, 
                 charging_power=charging_power_home,
                 charging_type='if_needed',
                 batt_size=batt_size,
                 arrival_departure_data_wd=arr_dep_data_h,
                 tou_we=True)
    
    if tou:
        print('\tToU-ing')
        t0 = time.time()
        for ev in grid.evs['Overnight']:
            ev.tou_ini = np.round((start_tous + np.random.rand(1) * delta_tous) % 24, 2)
            ev.tou_end = (ev.tou_ini + h_tous) % 24
            ev.set_off_peak(grid)
        print('\tFinished ToU-ing, elapsed time: {} s'.format(np.round(time.time()-t0,1)))
    grid.add_evs('Day', nevs_w, ev_type=ev_type,
                 dist_wd={'cdf': hwork.loc[ss].cumsum()/hwork.loc[ss].sum()},
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
    f1.savefig(dir_results + r'\\' + outputfolder + r'\Images\Total_{}.png'.format(ss))
    f2.savefig(dir_results + r'\\' + outputfolder + r'\Images\EV_{}.png'.format(ss))
    print('Total SS time: {} s'.format(np.round(times[-1]-times[-3],1)))
    print('Total elapsed time: {:02d}h{:02d}:{:02.1f}'.format(*util.sec_to_time(np.round(times[-1]-times[0],1))))
    #break
# Transform outputs in DataFrames and save
global_data = pd.DataFrame(global_data).T
ev_data = {ss : {el + '_' + ev_data[ss]['EV_sets'][i] : ev_data[ss][el][i] 
                    for i in range(len(ev_data[ss]['EV_sets']))
                    for el in ev_data[ss] if el != 'EV_sets'} 
        for ss in ev_data}
ev_data = pd.DataFrame(ev_data).T

ev_load_day = pd.DataFrame(ev_load_day, index=load.index[0:int(7*24*60/step)])
ev_load_night = pd.DataFrame(ev_load_night, index=load.index[0:int(7*24*60/step)])
global_data.to_csv(dir_results  + r'\\' + outputfolder + r'\global_data.csv')
ev_data.to_csv(dir_results  + r'\\'+ outputfolder + r'\ev_data.csv')
ev_load_day.to_csv(dir_results + r'\\' + outputfolder + r'\ev_load_day.csv')
ev_load_night.to_csv(dir_results + r'\\' + outputfolder + r'\ev_load_night.csv')