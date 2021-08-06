# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:06:48 2020
Runs EVs for one SS, detailing each IRIS subset of EVs.
Uses aggregator for EV valley filling or price response
Home & Work 

@author: U546416
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import EVmodel
import time
import pandapower as pp

times = [time.time()]
# DATA DATA
print('Loading data')
# Load data
# DAILY DISTANCE 
# Histograms of Distance per commune
print('Loading Histograms of distance')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
                    engine='python', index_col=0)
if 'ZE' in hwork.columns:
    hwork = hwork.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
    hhome = hhome.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))

# DEMOGRAPHIC DATA, NUMBER OF RESIDENTS & WORKERS PER IRIS
# IRIS & Commune info
print('Loading IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
times.append(time.time())
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))

# DISTRIBUTION OF ARRIVAL AND DEPARTURES
# Histograms of arrival/departures
print('Arrival departures')
# Bi variate distributions for arrival/departures
folder_arrdep = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité'
res_arr_dep_wd = pd.read_csv(folder_arrdep + r'\EN_arrdep_wd_modifFR.csv', 
                             engine='python', index_col=0)
res_arr_dep_we = pd.read_csv(folder_arrdep + r'\EN_arrdep_we_modifFR.csv', 
                             engine='python', index_col=0)
# Bi-variate distribution
work_arr_dep_wd = pd.read_csv(folder_arrdep + r'\Arr_Dep_pdf2.csv', 
                          engine='python', index_col=0)
times.append(time.time())

folder_profiles = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles_thesis\\'
netload = pd.read_csv(folder_profiles + 'netload_highpv.csv',
                      engine='python', index_col=0)
print('Finished loading, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))



# PARAMETERS OF SIMULATION

#############################
# IRIS TO SIMULATE
# iris_ss should be a list/pandas Series with IRIS codes
# reading grid
folder_grids = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\PPGrid\\'
net = pp.from_json(folder_grids + 'base_grid.json')
# selecting iris
iris_ss = net.load.zone.astype(int).unique()
# Computing supply area
# reading number of trafos per area
folder_lv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
lv_iris = pd.read_csv(folder_lv+'Nb_BT_IRIS2016.csv',
                      engine='python', index_col=0)
nb_bt_b = net.load[net.load.type_load=='Base'].groupby('zone').type_load.count()[iris_ss] # Number of supplied LV trafos per IRIS by SS
nb_bt=lv_iris.Nb_BT[iris_ss] # Total number of trafos per IRIS
supply = 1-((nb_bt-nb_bt_b)/nb_bt) # Ratio of supply


############################

###########################
# SIMULATION PARAMETERS
# Days, steps
step = 30 # time step (minutes)

###########################################
# GENERAL EV DATA

# EV penetration (0 to 1)
ev_penetration = 0.5
# EV home/work charging (0 to 1). 0.3 means 30% of EVs will charge 'at work'
ev_work_ratio = 0.3
# EV charging parameters (charging power, battery size, etc)
charging_powers= [3.6, 7.2, 11, 22]
corr_b_ch_home = {25: [0.7,0.3,0,0],
                  50: [0.3,0.5,0.2,0],
                  75: [0,0.7,0.3,0]}
# correlation between battery size // charging power
corr_b_ch_work = {25 : [0, 0.75, 0.25, 0],
                  50 : [0, 0.50, 0.50, 0],
                  75 : [0, 0.25, 0.50, 0.25]}
batt_size = [[25, 50, 75], [0.20, 0.6, 0.20]]
def driving_eff(b):
    return 0.14+0.0009*b

# Tou is used if Off-peak hours are enforced for home charging
tou = False
#tou_ini = 23
#tou_end = 8
#h_tous = 10
#start_tous = 21
#end_tous = 3
#delta_tous = (end_tous - start_tous) % 24


#####################################################
# ARRIVAL AND DEPARTURE SCHEDULES
#####################################################

# Arrival and departure hourly CDFs
n = res_arr_dep_wd.shape[0]
bins = np.arange(0,24.5,0.5)
arr_dep_data_h = {'wd': dict(pdf_a_d=res_arr_dep_wd.values,
                         bins=bins),
                 'we': dict(pdf_a_d=res_arr_dep_we.values,
                      bins=bins)}
arr_dep_data_w = {'0123456': dict(pdf_a_d=work_arr_dep_wd.values)}

###############################################
# CREATING SET OF PARAMETERS TO CREATE EV TYPES
##############################################
# these are common for all types of evs
general_params = dict(batt_size = batt_size,
                      driving_eff = driving_eff)

# these are for each kind of EV type
# Home charging params
home_params = dict(#charging_power = charging_power_home,
                   arrival_departure_data = arr_dep_data_h,
                   charging_type = 'if_needed',
                   n_if_needed = 1,   # This gives a mean of 2.5 plugs per week, similar to seen in demo projects
                   tou_we=False) 
# Day charging params
day_params = dict(#charging_power = charging_power_work,
                  arrival_departure_data = arr_dep_data_w,
                  charging_type = 'weekdays+1',
                  tou_we = False,
                                            # To have 0 distance during weekends
                  pmin_charger=0.0)




#%% Create Grid
ev_p = 1 #[0.25,0.5,0.75,1]
ev_wr = 0.5 #[0, 0.1, 0.2, 0.3, 0.4, 0.5]
times.append(time.time())
output_folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles_thesis\\'
totdays = 365
nweeks_rh = 2
print('Saving in ', output_folder)

# doing a rolling horizon
print('Creating grid for EV penetration {} and work ratio {}'.format(ev_p, ev_wr))
grid = EVmodel.Grid(ndays=nweeks_rh*7+1, step=step, verbose=True)
# Add aggregator
grid.add_prices(np.array(netload.values).squeeze(), step_p=0.5)
agg = grid.add_aggregator('agg', price_elasticity=0.001, param_update='prices')

# Add EVs for each IRIS
#for i in iris_ss.index:
for i in iris_ss:
    # Compute # of EVs 
    # Number of Evs
#    comm = iris_ss.COMM_CODE[i]
    comm  = int(i)//10000
    nevs_h = int(iris.N_VOIT[i] * ev_p * (1-ev_wr) * supply[i])
    nevs_w = int(hwork.loc[comm].sum() * iris.Work_pu[i] * # First term is total work EVs in the commune, second is the ratio of Workers in the iris
                 ev_p * ev_wr * 1.78 * supply[i])  # 1.78 is the ratio between nation-wide Work EVs and Total EVs  
#            print('EVs Overnight', nevs_h)
#            print('EVs Work', nevs_w)
    
    
    # Add EVs
    grid.add_evs('Home_' + str(i) , nevs_h, ev_type='optch',
                 #pmin_charger=0.1,
                 dist_wd=dict(cdf = hhome.loc[comm].cumsum()/hhome.loc[comm].sum()),
                 dist_we=dict(cdf = hhome.loc[comm].cumsum()/hhome.loc[comm].sum()),
                 aggregator=agg,
                 **general_params,
                 **home_params)  
    grid.add_evs('Work_' + str(i), nevs_w, ev_type='optch',
                 dist_wd= dict(cdf = hwork.loc[comm].cumsum()/hwork.loc[comm].sum()),
                 aggregator=agg,
                 **general_params,
                 **day_params)
    
# Correlating Battery sizes to Charging power
#    print('\tCorrelating Batt/Ch')
t0 = time.time()
for setev, evs in grid.evs_sets.items():
    for ev in evs:
        if 'Home' in setev:
            ev.charging_power = np.random.choice(a=charging_powers, p=corr_b_ch_home[ev.batt_size])
        elif 'Work' in setev:
            ev.charging_power = np.random.choice(a=charging_powers, p=corr_b_ch_work[ev.batt_size])
        ev.driving_eff = driving_eff(ev.batt_size)
                   
times.append(time.time())
print('Finished preprocessing, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
print('Total EVs: {}'.format(len(grid.evs)))


#%% Do simulations
print('Running EV simulation')
idw = 0
output_ev_data = []
while True:
    grid.prices = netload.values[idw*7*2*24:((idw+nweeks_rh)*7 + 1)*2*24].squeeze()
    agg.prices = netload.values[idw*7*2*24:((idw+nweeks_rh)*7 + 1)*2*24].squeeze()
    times.append(time.time())
    grid.do_days()
    times.append(time.time())
    print('Finished running, elapsed time: {} s'.format(np.round(times[-1]-times[-2],1)))
#    global_data = grid.get_global_data()
#    ev_data = grid.get_ev_data()
    idw += nweeks_rh
    if idw == 52:
        output_ev_data.append(pd.DataFrame(grid.ev_load).drop('Total', axis=1))
        break
    else:
        output_ev_data.append(pd.DataFrame(grid.ev_load).drop('Total', axis=1).iloc[:-2*24])
        grid.reset()

#        grid.plot_total_load(day_ini=7, days=7)
#        grid.plot_ev_load(day_ini=7, days=7)
output_ev_data = pd.concat(output_ev_data, axis=0, ignore_index=True)

#% Get EV data and save
print('Saving data')
of_name = 'EV_p' + str(ev_p) + '_w' + str(ev_wr) + '_optimized.csv'

output_ev_data.to_csv(output_folder + of_name)
print('End simulations')