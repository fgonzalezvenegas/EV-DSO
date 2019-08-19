# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:05:58 2019
Traitement de données de courbe de charge ENEDIS:
Inputs: Raw file de courbe de charge 
"Agrégats segmentés de consommation électrique au pas 1/2 h des points de soutirage <= 36kVA"
Sans répartition par puissance (i.e. seulement données PTotal par profil)
"Agrégats segmentés de consommation électrique au pas 1/2 h des points de soutirage > 36kVA"
Assignation des communes aux postes sources
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
profils = {'ENT': ['ENT1 (+ ENT2)', 'ENT3 (+ ENT4 + ENT5)'], 
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
#Before parsing dates, reduce size of data, removing all
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

#%% Reading comm & trafo data
print('loading data conso per commune')
fncomm = 'consommation-electrique-par-secteur-dactivite-commune-red.csv'

conso_comm = pd.read_csv(folder_modif + fncomm, engine='python', delimiter=';', 
                         index_col=0)
Tgeo = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/geoRefs.csv', 
                 engine='python', delimiter=';', index_col=0)
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', delimiter=';', decimal = ',', index_col=0)


#%% Assign commune to substation, given closer substation

comm_to_ss = {}
i = 0
print('Starting to compute closer trafo')
for comm in Tgeo.index:
    i += 1
    if i % 500 == 0:
        print(i)
    if (Tgeo.ZE[comm] > 9900 or #This means it is outside france
        comm[0:2] == '97' or        #This means its guyanne
        comm[0:2] == '2A' or    #This means its corse
        comm[0:2] == '2B')  :     #This means its corse
        comm_to_ss[comm] = 'NoSS'
    else:
        latlon = np.asarray([Tgeo.GeoLat[comm], Tgeo.GeoLong[comm]])
        d = 1000
        degree = 0.3
        for s in SS.loc[(SS.Lat-latlon[0] < degree) & (SS.Lon-latlon[1] < degree) 
                & (SS.Lat-latlon[0] > -degree) & (SS.Lon-latlon[1] > -degree)].index:
            dd = mb.computeDist(latlon, np.asarray([SS.Lat[s], SS.Lon[s]]))
            if dd < d:
                d = dd
                comm_to_ss[comm] = s
print('finished computing closer trafo, assigned comms %d' %len(comm_to_ss))

#%% 
c_no_ss = []
comm_in_ss = {}
for comm in Tgeo.index:
    if not comm in comm_to_ss:
        c_no_ss.append(comm)
    elif comm_to_ss[comm] in comm_in_ss:
        comm_in_ss[comm_to_ss[comm]].append(comm)
    else:
        comm_in_ss[comm_to_ss[comm]] = [comm]

ss_no_c = []
for ss in SS.index:
    if not ss in comm_in_ss:
        ss_no_c.append(ss)
print(ss_no_c)
print(c_no_ss)
SS['Communes']= pd.Series(comm_in_ss)
SS.to_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv')
#%%           
out =  ['2A', '2B', '97', 'SU', 'BE', 'LU', 'AL']
Tfr = Tgeo
for odep in out:
    Tfr = Tfr[Tfr.Dep != odep]

plt.plot(Tfr.GeoLong, Tfr.GeoLat,'*', label='Communes')
plt.plot(Tfr.GeoLong[c_no_ss], Tfr.GeoLat[c_no_ss],'*', label='Communes w/o SS')
plt.plot(SS.Lon, SS.Lat,'*', label='Substations')
plt.plot(SS.Lon[ss_no_c], SS.Lat[ss_no_c],'*', label='SS w/o Commune')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Communes and Substations')
plt.xlim([-6,10])
plt.legend()