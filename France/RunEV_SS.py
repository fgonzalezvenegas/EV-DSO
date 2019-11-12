# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:32:55 2019
testestest
@author: U546416
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import EVmodel as evmodel


# DATA DATA
# Load data
# Histograms of Distance
folder_hdata = 'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_data + r'\HistWorkModal.csv', 
                    engine='python', index_col=0) 
# Substation info
folder_ssdata = 'c:\user\U546416\Documents\PhD\Data\Mobilité\Réseau'
SS = folder_geodata = 'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_geodata + r'\IRIS.csv', 
                    engine='python', index_col=0)
conso = iris = pd.read_csv(folder_geodata + r'\'conso_all_pu.csv', 
                    engine='python', index_col=0)
# IRIS & Commune info
folder_geodata = 'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_geodata + r'\IRIS.csv', 
                    engine='python', index_col=0)
conso = iris = pd.read_csv(folder_geodata + r'\'conso_all_pu.csv', 
                    engine='python', index_col=0)

# Histograms of arrival/departures


# GENERAL EV DATA

# EV penetration
# EV home/work charging
# EV charging params (charging power, batt size, etc)


# SIMULATION DATA (TIME)
# Days, steps
ndays = 21
step = 30

#%%