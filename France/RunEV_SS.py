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


# DATA DATA
print('Loading data')
# Load data
# Histograms of Distance
print('Histograms of distance')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
                    engine='python', index_col=0) 
# Substation info
print('Loading SS')
folder_ssdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau'
SS = pd.read_csv(folder_ssdata + r'\postes_source.csv',
                                  engine='python', index_col=0)
#SS_polys = pd.read_csv(folder_ssdata + r'/postes_source_polygons.csv', 
#                 engine='python', index_col=0)

# IRIS & Commune info
print('Loading IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
print('Loading conso profiles')
conso_profiles = pd.read_csv(folder_consodata + r'\conso_all_pu.csv', 
                    engine='python', index_col=0)

# Histograms of arrival/departures
print('Arrival departures')
folder_arrdep = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Mobilité'
arr_dep = pd.read_csv(folder_arrdep + r'\DepartureArrivals.csv', 
                    engine='python', index_col=0)
# SIMULATION DATA (TIME)
# Days, steps
ndays = 21
step = 30

#%% Habitants per comm
habcomm = iris[['COMM_CODE', 'Habitants']].groupby('COMM_CODE').sum().squeeze()
habcomm  = habcomm[iris.COMM_CODE]
habcomm.index = iris.index
irishabpu = iris.Habitants / habcomm

#actives per iris, proportional to PRO, TERTIAIRE and INDUSTRIE conso 
act

irisactpu
#%% Run for one SS
# GENERAL EV DATA

# EV penetration
ev_penetration = .5
# EV home/work charging
ev_work_ratio = .3
# EV charging params (charging power, batt size, etc)
charging_power_home = [[3.6, 7.2, 11], [0.5, 0.4, 0.1]]
charging_power_work = [[3.6, 7.2, 11], [0.1, 0.4, 0.5]]
batt_size = [[20, 40, 60, 80], [20, 30, 30, 20]]

ss = 'BORIETTE'
iris_ss = iris[iris.SS==ss]

# Number of Evs
nevs_h = 
nevs_w = 

# compute base load
load = 1



grid = EV.Grid(ndays=ndays, step=step, load=load, ss_pmax=SS.Pmax[ss]
grid.add_evs('Overnight', n_evs)
