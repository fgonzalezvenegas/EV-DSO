# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:05:58 2019
Traitement de données de courbe de charge ENEDIS:
Inputs: Raw file de courbe de charge 
"Agrégats segmentés de consommation électrique au pas 1/2 h des points de soutirage <= 36kVA"
Sans répartition par puissance (i.e. seulement données PTotal par profil)
"Agrégats segmentés de consommation électrique au pas 1/2 h des points de soutirage > 36kVA"
@author: U546416
"""

import pandas as pd
import numpy as np
#from dateutil.parser import parse
import datetime as dt
import mobility as mb
from matplotlib import pyplot as plt


folder_raw = 'c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Base/Conso/'
folder_modif = 'c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Conso/'
fninf36 = 'conso-inf36.csv'
fnsup36 = 'conso-sup36.csv'

#%%
print('loading data <= 36kVA')
consoinf36 = pd.read_csv(folder_raw + fninf36, engine='python', delimiter=';')

#%%
# Rename columns and parse first to Date type
print('Parsing dates')
consoinf36 = consoinf36.rename(columns={consoinf36.columns[0]: 'Date', 
                                        consoinf36.columns[4]: 'TotalE',
                                        consoinf36.columns[11]: 'JMaxMois',
                                        consoinf36.columns[12]: 'SMaxMois'})
fmtdt = '%Y-%m-%dT%H:%M:%S%z'
consoinf36.Date = consoinf36.Date.apply(
        lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt)) #pas top mais ca va

#%% Do pivot table with total energy and pivot table with indicator of max jour/week of month
print('Creating pivot tables')
pvtable = pd.pivot_table(consoinf36, columns='Profil', 
                         index='Date', values='TotalE', 
                         aggfunc=np.sum).sort_index()

pt = pd.pivot_table(consoinf36,index='Date', values=['JMaxMois','SMaxMois'])
#% Computing aggregated load curves
print('Computing aggregated tables')
profils = {'ENT': ['ENT1 (+ ENT2)', 'ENNbT3 (+ ENT4 + ENT5)'], 
           'PRO': ['PRO1 (+ PRO1WE)', 'PRO2 (+ PRO2WE + PRO6)', 
                    'PRO3', 'PRO4', 'PRO5'],
           'RES': ['RES1 (+ RES1WE)', 'RES11 (+ RES11WE)', 
                    'RES2 (+ RES5)', 'RES2WE', 'RES3', 'RES4']}
# Co,puting, *2 is to convert it to power, /1e6 to convert it to MW
pvtable_inf = pd.DataFrame({key: sum([pvtable[pr] * 2/1e6 for pr in profils[key]]) for key in profils})

pvtable_inf_pu = pd.DataFrame({key: pvtable_inf[key]/sum(pvtable_inf[key])*8760*2 for key in pvtable_inf})

#%%
print('loading data > 36kVA')
consosup36 = pd.read_csv(folder_raw + fnsup36, engine='python', delimiter=';')
consosup36 = consosup36.rename(columns={consosup36.columns[0]: 'Date',
                                        consosup36.columns[2]: 'PSouscrite',
                                        consosup36.columns[3]: 'Secteur', 
                                        consosup36.columns[5]: 'TotalE',
                                        consosup36.columns[12]: 'JMaxMois',
                                        consosup36.columns[13]: 'SMaxMois'})
 
#%% Process and parse datetimes
print('Processing data and parsing')
# Deleting unuseful columns:
filtervalues = ['P3: Total ]36-250] kVA', 'P7: Total > 250 kVA']

consosup36 = pd.concat([consosup36[consosup36.PSouscrite == fval] for fval in filtervalues])
#Before parsing dates, reduce size of data, only leaving values for [36-250kVA] and [>250kVA]
print('Parsing Dates')
consosup36.Date = consosup36.Date.apply(
        lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt)) #ca marche et pas trop lent

#%% Do pivottable
print('Creating pivot tables')
pvtable_sup = pd.pivot_table(consosup36, columns='Secteur', 
                         index='Date', values='TotalE', 
                         aggfunc=np.sum).sort_index()
# convert to Power in MW
pvtable_sup = pd.DataFrame({key: pvtable_sup[key] * 2/1e6 for key in pvtable_sup})

Cols = ['Agriculture', 'Industrie', 'Tertiaire', 'NonAffecte']
pvtable_sup = pvtable_sup.rename(columns={pvtable_sup.columns[i]: Cols[i] for  i in range(4)})

# Create table in p.u.
pvtable_sup_pu = pd.DataFrame({key: pvtable_sup[key]/sum(pvtable_sup[key])*8760*2 for key in pvtable_sup})

#%%  Saving data

pvtable_all = pd.concat([pvtable_inf, pvtable_sup], axis=1)
#pvtable_all.to_csv(folder_modif + 'conso_all.csv')
pvtable_all_pu = pd.concat([pvtable_inf_pu, pvtable_sup_pu], axis=1)
#pvtable_all_pu.to_csv(folder_modif + 'conso_all_pu.csv')

#
#pvtable_sup_pu.to_csv(folder_modif + 'conso_sup36_pu.csv')
#pvtable_inf_pu.to_csv(folder_modif + 'conso_inf36_pu.csv')
#pvtable_sup.to_csv(folder_modif + 'conso_sup36.csv')
#pvtable_inf.to_csv(folder_modif + 'conso_inf36.csv')

