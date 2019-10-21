# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:26:51 2019
Analysis of Sub-station peak load:
    Inputs: 
        Base load in pu (per type of user)
        Communes with share of each type of load
        Assigment of commune - Substation
@author: U546416
"""
from matplotlib import pyplot as plt
import pandas as pd
#import mobility as mb
import numpy as np


#%% Load data
folder = 'c:/user/U546416/Documents/PhD/Data/Mobilité/'
data_file = 'data-flux-mob-dreal.txt'
geo_file = 'geoRefs.csv'
input_Thome = 'HistHome.csv'
input_Twork = 'HistWork.csv'

folder_load = 'Data_Traitee/Conso/'
folder_grid = 'Data_Traitee/Reseau/'

file_load_comm = 'consommation-electrique-par-secteur-dactivite-commune-red.csv'
file_load_profile = 'conso_all_pu.csv'
file_ss = 'postes_source.csv'
file_iris = 'IRIS.csv'

# Load conso by commune data
load_by_comm = pd.read_csv(folder + folder_load + file_load_comm, 
                           engine='python', delimiter=';', index_col=0, dtype = {'CODE':str})
load_by_comm.index = load_by_comm.index.astype(str)
load_by_comm.index = load_by_comm.index.map(lambda x: x if len(x) == 5 else '0' + x)
# Load conso by IRIS
iris = pd.read_csv(folder + folder_load + file_iris, engine='python', index_col=0)


# Load conso profiles data (in pu (power, not energy))
load_profiles = pd.read_csv(folder + folder_load + file_load_profile, 
                           engine='python', delimiter=',', index_col=0)
# drop ENT profile that's not useful
load_profiles = load_profiles.drop('ENT', axis=1)
# Load Trafo data
SS = pd.read_csv(folder + folder_grid + file_ss, 
                           engine='python', delimiter=',', index_col=0)
# parse communes
SS.Communes = SS.Communes.apply(lambda x: eval(x) if x==x else [])
# parse IRIS
SS.IRIS = SS.IRIS.apply(lambda x: [int(i) for i in eval(x)] if x==x else [])
#%% Compute load for given Substation
ss = input('Select Substation (ex. %s) : ' % np.random.choice(SS.index) )
while not ss in SS.index:
    ss = input('Invalid choice, select Substation (ex. %s) : ' % np.random.choice(SS.index) )
comms = SS.Communes[ss]
print('Youve selected %s, with %d associated communes %s' % (ss, len(comms), comms))

try: 
    factors_by_comm = load_by_comm.loc[comms, load_profiles.columns]
except:
    factors_by_comm = pd.DataFrame({'PRO': 0, 'RES': 0, 'Agriculture': 0, 'Industrie': 0, 
                                    'Tertiaire': 0, 'NonAffecte': 0}, index=['NoComm'])    
factors = factors_by_comm.sum() / (8760)
load = load_profiles * factors
print('Computed base load [MWh]')
print(load.sum())
print('Max load %f MW, over SS capacity of %d MW' % (load.sum(axis=1).max(), SS.Pmax[ss]))
maxweek = 9
plt.figure()
load.iloc[(maxweek-1)*7*2*24:maxweek*7*2*24+1].plot(kind='area')
plt.axhline(y=SS.Pmax[ss], color='red', linestyle='--', label='SS Pmax')
plt.ylim([0, max([SS.Pmax[ss], load.sum(axis=1).max()])*1.1])
plt.title('Max load week, SS %s' %ss)
plt.ylabel('MW')

#%% Pre analysis of load, using COMMUNES
print('Computing max load analysis')
SS_load = {}
fs = {}
for ss in SS.index:
    comms = SS.Communes[ss]
    try: 
        factors_by_comm = load_by_comm.loc[comms, load_profiles.columns]
    except:
        factors_by_comm = pd.DataFrame({'PRO': 0, 'RES': 0, 'Agriculture': 0, 'Industrie': 0, 
                                    'Tertiaire': 0, 'NonAffecte': 0}, index=['NoComm'])  
    factors = factors_by_comm.sum(axis=0) / (8760)
    load = load_profiles * factors
    fs[ss] = factors
    if SS.Pmax[ss] == 0:
        SS_load[ss] = [SS.Pmax[ss], factors.sum(), load.sum(axis=1).max(), 
                9999, load.sum(axis=1).idxmax()]
    else:
        SS_load[ss] = [SS.Pmax[ss], factors.sum(), load.sum(axis=1).max(), 
                load.sum(axis=1).max()/SS.Pmax[ss], load.sum(axis=1).idxmax()]

SS_load = pd.DataFrame(SS_load, index=['Pmax_SS', 'AnnualLoad', 'MaxLoad','SSCharge', 'idMaxLoad']).transpose()
fs = pd.DataFrame(fs).transpose()
SS_load = pd.concat([SS_load, fs], axis=1)
SS_loadred = SS_load.loc[(SS_load.SSCharge > 0) & (SS_load.SSCharge < 9998)]
print('Finished computing')
#%% Pre analysis of load, using IRIS
print('Computing max load analysis')
SS_load = {}
fs = {}
i = 0
for ss in SS[SS.GRD == 'Enedis'].index:
    i +=1
    if i%100 == 0:
        print(i)
    iriss = SS.IRIS[ss]
    
    factors_by_comm = iris.loc[iriss, load_profiles.columns]
    factors = factors_by_comm.sum(axis=0) / (8760)
    load = load_profiles * factors
    fs[ss] = factors
    if SS.Pmax[ss] == 0:
        SS_load[ss] = [SS.Pmax[ss], factors.sum(), load.sum(axis=1).max(), 
                9999, load.sum(axis=1).idxmax()]
    else:
        SS_load[ss] = [SS.Pmax[ss], factors.sum(), load.sum(axis=1).max(), 
                load.sum(axis=1).max()/SS.Pmax[ss], load.sum(axis=1).idxmax()]

SS_load = pd.DataFrame(SS_load, index=['Pmax_SS', 'AnnualLoad', 'MaxLoad','SSCharge', 'idMaxLoad']).transpose()
fs = pd.DataFrame(fs).transpose()
SS_load = pd.concat([SS_load, fs], axis=1)
SS_loadred = SS_load.loc[(SS_load.SSCharge > 0) & (SS_load.SSCharge < 9998)]
print('Finished computing')

SS_load.to_excel('SSload.xlsx')

#%%
plt.figure()
plt.hist(SS_loadred.SSCharge, bins=np.arange(0,3,0.2))
plt.title('Distribution of max load charge of substations')
plt.ylabel('Frequency')
plt.xlabel('Load charge [p.u.]')
plt.axvline(x=0.8, color='red')
p08 = (SS_loadred.SSCharge > 0.8).sum()/len(SS_loadred.index)*100
plt.text(0.9, plt.ylim()[1]*0.8, 'SS with max load > 0.8 pu = %1.2f%%' %p08)