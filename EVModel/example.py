# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:29:40 2021

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EVmodel

# Creating a base case
# number of days and time step of simulation
ndays = 7
timestep = 30 #minutes
grid = EVmodel.Grid(ndays, timestep)

# Creating a set of EVs doing uncontrolled charging
# number of EVs
nevs = 100
evs = grid.add_evs('uncontrolled', nevs, 'dumb')
# Doing simulation
grid.do_days()
# Plotting
grid.plot_ev_load()

#%% Adding an extra set of EVs doing 'average' charging
# Including extra parameters:
# Charger power, battery size and plug in behavior
grid.reset()
evsmod = grid.add_evs('average',nevs, 'mod', batt_size=25, charging_power=11, charging_type='all_days')
grid.do_days()
grid.plot_ev_load()

#%% Adding custom arrival and departure times
# Arrival and departure times are sampled every day from given probability distribution functions, 
# which can be different for each day of the week
# Options are:
#1- Bivariate probability distribution
#   {'pdf_a_d' : Matrix of joint probability distribution of arrival departure,
#   'bins' : bins in range [0,24]
#2- CDF of not correlated arrival and departure
#   {'cdf_arr': Array of cumulative distribution function of arrival,
#    'cdf_dep': Array of cdf of departure}
#3- Gaussian (normal) distributions
#    {'mu_arr': ma, 'std_dep': sa 
#    'mu_dep': md, 'std_dep': sd }

# Reading data from Electric Nation trial, it is a bivariate probability distribution of arr/dep during weekdays
res_arr_dep_wd = pd.read_csv('data/EN_arrdep_wd.csv', 
                             engine='python', index_col=0)
# Creating Arrival/departure dataset
# During weekdays we will use the data from Electric Nation
# During weekends, we will provide parameters for normal distribution
n = res_arr_dep_wd.shape[0]
bins = np.arange(0,24.5,24/n)
arr_dep_data = {'wd': dict(pdf_a_d=res_arr_dep_wd.values,
                         bins=bins), # weekdays
                  '5': dict(mu_arr=14, std_arr=5,
                           mu_dep=8, std_dep=2), # saturday
                  '6': dict(mu_arr=15, std_arr=1,
                            mu_dep=7, std_dep=2)} # sunday
                 

grid.reset()
evsmod = grid.add_evs('average',nevs, 'mod', batt_size=25, 
                      charging_power=11, charging_type='all_days',
                      arrival_departure_data=arr_dep_data)
grid.do_days()
grid.plot_ev_load()

#%% Doing optimized charging based on electricity prices
prices = -np.cos(np.arange(0,7*24)*2*np.pi/24) + 1

grid = EVmodel.Grid(ndays, timestep)
# Adding prices
# step_p represents the time step of the price vector (in hours), 
# which can be different from the simulation time step
grid.add_prices(prices, step_p=1)

# Adding EVs doing uncontrolled charging
grid.add_evs('uncontrolled', nevs, 'dumb')

# Adding EVs doing optimized charging
grid.add_evs('optimized', nevs, 'optch', charging_power=7, charging_type='all_days')
grid.do_days()

f, axs = plt.subplots(2)
grid.plot_ev_load(ax=axs[0])
axs[1].set_title('Electricity prices')
axs[1].plot(prices, 'k--', label='Electricity prices')
axs[1].set_ylabel('Price')
axs[1].set_xticks(axs[0].get_xticks())
axs[1].set_xticklabels(axs[0].get_xticklabels())
axs[1].set_xlim(0,len(prices))
plt.tight_layout()


#%% Doing decentralized valley filling charging based on substation load
base_load = -np.cos(np.arange(0,7*24*2)*2*np.pi/24/2) + 1

grid = EVmodel.Grid(ndays, timestep, ss_pmax=3)
# Adding base load
# step_p represents the time step of the price vector (in hours), 
# which can be different from the simulation time step
grid.add_base_load(base_load, step_p=0.5)

# Adding aggregator agent that will control the EV charging
agg = grid.add_aggregator('agg1', param_update='SS_load')

# Adding EVs doing optimized charging, they should be attached to the aggregator
nevs = 500
grid.add_evs('optimized', nevs, 'optch', charging_power=7, 
             charging_type='all_days', aggregator=agg)
grid.do_days()

grid.plot_total_load()
