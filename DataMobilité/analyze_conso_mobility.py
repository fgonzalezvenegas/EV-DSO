    # -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:56:04 2019
trying to do machine learnign but failed
@author: U546416
"""
import pandas as pd
import numpy as np
#import mobility as mb
import matplotlib.pyplot as plt
import sklearn.decomposition as sd

#%%
folder_data_conso = r'c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Conso/'
file_tgeo_xls = 'flux-DataGeo.xlsx'
file_tgeo_csv = 'geoRefs.csv'
folder_tgeo = r'c:/user/U546416/Documents/PhD/Data/Mobilité/'

print('loading data conso per commune')
fncomm = 'consommation-electrique-par-secteur-dactivite-commune-red.csv'

# Read conso per commune, drop NaN values and correct index
conso_comm = pd.read_csv(folder_data_conso + fncomm, engine='python', delimiter=';', 
                         index_col=0)
conso_comm= conso_comm.dropna()

print('parsing')
idx = conso_comm.index.astype(str)
idx = idx.map(lambda x: '00000'[0:5-len(x)] + x)
conso_comm.index = idx

Tgeo = pd.read_excel(folder_tgeo + file_tgeo_xls, sheetname='GeoDonnées', 
                      index_col=0)

Tgeored = Tgeo.dropna()
Tgeored = Tgeored.drop(Tgeored.index[Tgeored[r'#Actifs INSEE'] < 50])
#Parse Lat and Long
Tgeored.GeoLat = Tgeored.GeoLat.apply(lambda x: float(x.replace(',','.')))
Tgeored.GeoLong = Tgeored.GeoLong.apply(lambda x: float(x.replace(',','.')))
#%%

f, axs = plt.subplots(2,2)

Status = Tgeored.Status.unique()
colors = {'R' : 'green',
          'C' : 'yellow',
          'B' : 'orange', 
          'I' : 'blue' }
km_h = 'kmMoyen-Hab'
km_t = 'kmMoyen-Trav'

for i in range(4):
    s = Status[i]
    idx = Tgeored.Status == s
    ax = axs[i%2][i//2]
    ax.plot(Tgeored[km_h][idx], Tgeored.UU[idx], '*', color=colors[s], label=s)
    ax.legend()
    ax.set_ylim([-1,9])
    ax.set_xlim([0,100])

f2, ax2 = plt.subplots()
for i in range(4):
    s = Status[i]
    idx = Tgeored.Status == s
    ax2.plot(Tgeored.GeoLong[idx], Tgeored.GeoLat[idx], '*', color=colors[s], label=s)
ax2.legend()

#%%
profs = ['RES', 'PRO', 'Agriculture', 'Industrie', 'Tertiaire', 'NonAffecte']


T = conso_comm[profs].sum(axis=1)
ppu = {}
name = 'Nom Commune'
for p in profs:
    ppu[p] = conso_comm[p]  / T
ppu = pd.DataFrame(ppu)
for p in profs:
    idx = str(ppu[p].idxmax())
    while len(idx) < 5:
       idx = '0' + idx 
    print(p, idx, Tgeo[name][idx], ppu[p].max())

#%%

data = pd.concat([Tgeored.Status, ppu[profs[:-1]],Tgeored[km_h]/100, conso_comm['Habitants'], conso_comm.RES/conso_comm.NbRES], axis=1)
data = data.dropna()

data_nbres = pd.concat([Tgeored[km_h]/100, conso_comm[profs]/conso_comm.NbRES], axis=1)
#%
#pp = [profs[1]] + profs[3:5]
#for i in range(3):
#    y = pp[i]
#    f, axs = plt.subplots(2,2)
#    for i in range(4):
#        x = 'RES'
#        s = Status[i]
#        idx = data.Status == s
#        ax = axs[i%2][i//2]
#        ax.plot(data[x][idx], data[y][idx], '.', color=colors[s], label=s)
#        print(s, '\n', data[:][idx].mean())
#        ax.legend()
#        ax.set_ylim([0,1])
#        ax.set_xlim([0,1])
#        ax.set_xlabel(x)
#        ax.set_ylabel(y)
#%% Do PCA

pca = sd.PCA(n_components=4)
data_pu = {}
for p in data.columns:
    if p == 'Status':
        continue
    data_pu[p] = data[p]/data[p].max()
data_pu = pd.DataFrame(data_pu)


