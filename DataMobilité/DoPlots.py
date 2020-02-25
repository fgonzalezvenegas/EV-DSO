# -*- coding: utf-8 -*-
"""
Do some plots to have some fun
Created on Tue Feb 26 11:48:38 2019

@author: U546416
"""
#import time
#import csv
#import mobility as mb
#import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import mobility as mb
import numpy as np



folder = 'c:/user/U546416/Documents/PhD/Data/Mobilité/'
data_file = 'data-flux-mob-dreal.txt'
geo_file = 'geoRefs.csv'
input_Tdreal = 'HistDrealModal.csv'
input_Thome = 'HistHomeModal.csv'
input_Twork = 'HistWorkModal.csv'

f_data = 'Data_Traitee/Conso/'
file_conso = 'conso_all.csv'
file_conso_pu = 'conso_all_pu.csv'

# Init Tgeo
Tgeo = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/geoRefs.csv', 
                 engine='python', delimiter=',', index_col=0)
# Remove Fr hors metropolitaine
Tgeo = Tgeo[(Tgeo.Status != 'X') & (Tgeo.Dep != '97') & (Tgeo.Dep != '2A') & (Tgeo.Dep != '2B')]
#%% Init histograms of distribution of distances per commune
indexes = ['CODE', 'ZE', 'Dep', 'UU', 'Status']
Tdreal = pd.read_csv(folder + input_Tdreal, engine='python', index_col=indexes)
Thome = pd.read_csv(folder + input_Thome, engine='python', index_col='CODE')
Twork = pd.read_csv(folder + input_Twork, engine='python', index_col=indexes)

## Init conso data
#conso_data = pd.read_csv(folder + f_data + file_conso, 
#                         engine='python', index_col=0)
#conso_data_pu = pd.read_csv(folder + f_data + file_conso_pu, 
#                         engine='python', index_col=0)

#%% Add CAT_R to Thome index
Thome['CATAU_R'] = Tgeo.CATAU_R.loc[Thome.index]
Thome = Thome.set_index([Thome.index, 'CATAU_R'] + indexes[1:])
#%% Plot distribution of overall distances
step = 2
nbins = 50
x = [i*step+1 for i in range(nbins)]

f1, (ax1, ax2) = plt.subplots(1,2,sharey=True)
mb.plotDist(x, Tdreal.sum(), ax1)
ax1.set_title('Distribution distance \n Centre-Centre Voiture')
mb.plotDist(x, Thome.sum(), ax2)
ax2.set_title('Distribution distance \n Perturbée Voiture')    

#f1, ax1 = plt.subplots(1,1,sharey=True)
#ax1.plot(x, Thome.sum(), label='Habitant en France')
#ax1.plot(x, Twork.sum(), label='Travaillant en France')
#ax1.legend()

#%% Plot for one ZE - Paris 1101
ZE = 1101

f2, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
mb.plotDist(x,Thome.sum(level='ZE').loc[ZE],ax1)
ax1.set_title('Habitant à ZE Paris \n n=%8.0f' % Thome.sum(level='ZE').loc[ZE].sum())
mb.plotDist(x,Twork.sum(level='ZE').loc[ZE],ax2)
ax2.set_title('Travaillant à ZE Paris \n n=%8.0f' % Twork.sum(level='ZE').loc[ZE].sum())

#%%
# Plot for the UU
Thuu = Thome.sum(level='UU')
nuu, nvar = Thuu.shape
t = pd.Series(['Rural','UU<100k','UU<100k','UU<100k',
               'UU<100k','UU<100k','UU<2M','UU<2M','Paris'], 
              index=Thuu.index, name='UUtype')
Thuu_red = Thuu.set_index(t, append=True).sum(level='UUtype')
nuu_red = 4
muu_red = {}
# computing histograms of UU in per unit
for i in range(nuu_red):
    Thuu_red.iloc[i,:] = Thuu_red.iloc[i,:]/sum(Thuu_red.iloc[i,:])
    muu_red[Thuu_red.index[i]] = sum(Thuu_red.iloc[i,:] * x) 
#
f3, ax1 = plt.subplots(1, 1, sharey=True)
uutypes = ['Rural', 'UU<100k', 'UU<2M', 'Paris']
sp = ['       ', '', '   ', '       ']
for i in range(4):
    ax1.plot(x,Thuu_red.loc[uutypes[i]], label=uutypes[i] + ', ' + sp[i] + 'Moyenne=%2.1f km' %muu_red[uutypes[i]])
plt.legend()
ax1.set_title('Distribution de distances par type de UU (trajet aller)\npar commune de résidence')
ax1.set_xlim([0,100])
ax1.set_ylim([0,0.17])
ax1.set_ylabel('Densité')
ax1.set_xlabel('Distance [km]')

#%%
# Plot for the UU
Twuu = Twork.sum(level='UU')
nuu, nvar = Twuu.shape
t = pd.Series(['Rural','UU<100k','UU<100k','UU<100k',
               'UU<100k','UU<100k','UU<2M','UU<2M','Paris'], 
              index=Twuu.index, name='UUtype')
Twuu_red = Twuu.set_index(t, append=True).sum(level='UUtype')
nuu_red = 4
muu_red = {}
# computing histograms of UU in per unit
for i in range(nuu_red):
    Twuu_red.iloc[i,:] = Twuu_red.iloc[i,:]/sum(Twuu_red.iloc[i,:])
    muu_red[Twuu_red.index[i]] = sum(Twuu_red.iloc[i,:] * x) 
#
f3, ax1 = plt.subplots(1, 1, sharey=True)
uutypes = ['Rural', 'UU<100k', 'UU<2M', 'Paris']
sp = ['       ', '', '   ', '       ']
for i in range(4):
    ax1.plot(x,Twuu_red.loc[uutypes[i]], label=uutypes[i] + ', ' + sp[i] + 'Moyenne=%2.1f km' %muu_red[uutypes[i]])
plt.legend()
ax1.set_title('Distribution de distances par type de UU (trajet aller)\npar commune de travail')
ax1.set_xlim([0,100])
ax1.set_ylim([0,0.17])
ax1.set_ylabel('Densité')
ax1.set_xlabel('Distance [km]')

#%% Plot for the CAT Aire Urbaine:
Thau = Thome.sum(level='CATAU_R')
CatsAU = ['Urban', 'Periurban', 'Rural'] # 'Small_Pole' is out of this graph
# computing histograms of AU in per unit
Thau = (Thau.T / Thau.sum(axis=1)).T
# Plot
f3, ax1 = plt.subplots(1, 1, sharey=True)
sp = ['       ', ' ', '        ']
means_au = (Thau * x).sum(axis=1) 
for i in range(len(CatsAU)):
    ax1.plot(x,Thau.loc[CatsAU[i]], label=CatsAU[i] + ', ' + sp[i] + 'Mean=%2.1f km' %means_au.loc[CatsAU[i]])
plt.legend()
ax1.set_title('Commuting distance distribution')
ax1.set_xlim([0,100])
ax1.set_ylim([0,0.17])
ax1.set_ylabel('Density')
ax1.set_xlabel('Distance [km]')

#%% Plot mean distance
# Obsolete : Use Polygons!
step = 2
nbins = 50
x = [i*step+1 for i in range(nbins)]
# Drop small communes & Corse
Tth = Thome.sum(axis=1)
Ttw = Twork.sum(axis=1)
Tth = Tth[Tth>50]
Tth = Tth.drop(Tth.xs("2A", level='Dep', drop_level=False).index).drop(Tth.xs("2B", level='Dep', drop_level=False).index)
Ttw = Ttw[Ttw>50]
Ttw = Ttw.drop(Ttw.xs("2A", level='Dep', drop_level=False).index).drop(Ttw.xs("2B", level='Dep', drop_level=False).index)
# Compute mean distance
Th = ((Thome * x).sum(axis=1) / Tth).dropna()
Tw = ((Twork * x).sum(axis=1) / Ttw).dropna()

f4, (ax1, ax2) = plt.subplots(1,2)
# Do plot of mean distance for communes
colors = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
ranges = [i * 7.5 for i in range(len(colors))]
ranges.append(100)
for k in range(len(colors)):
    th = Th[(Th>ranges[k]) & (Th<=ranges[k+1])]
    ax1.plot(Tgeo.GeoLong[th.index.get_level_values('CODE')], Tgeo.GeoLat[th.index.get_level_values('CODE')], '1', color=colors[k], label='{}<d<{}'.format(ranges[k], ranges[k+1]))
ax1.legend()
ax1.grid()
ax1.set_title('Distance moyenne, par commune de résidence')

for k in range(len(colors)):
    tw = Tw[(Tw>ranges[k]) & (Tw<=ranges[k+1])]
    ax2.plot(Tgeo.GeoLong[tw.index.get_level_values('CODE')], Tgeo.GeoLat[tw.index.get_level_values('CODE')], '1', color=colors[k], label='{}<d<{}'.format(ranges[k], ranges[k+1]))
ax2.legend()
ax2.grid()
ax2.set_title('Distance moyenne, par commune de travail')


#%% Plot UU
Tg = Tgeo[(Tgeo.UU < 9) & (Tgeo.Dep != '97') & (Tgeo.Dep != '2A') & (Tgeo.Dep != '2B')]
uu = Tgeo.UU.unique().sort()
f, ax = plt.subplots()
# 0 Rural, 1-5 AU<100k, 6-7 AU>100k, 8 Paris
uutypes = ['Rural', 'UU<100k', 'UU<2M', 'Paris']
ranges = [0,1,6,8,9]
c = ['g','b','orange','pink']
for i in range(len(ranges)-1):
    coms =  Tg[(Tg.UU >= ranges[i]) & (Tg.UU <ranges[i+1])]
    print(uutypes[i], coms.shape[0])
    ax.plot(coms.GeoLong, coms.GeoLat, '.', c=c[i], markersize=1.5, label=uutypes[i])
ax.grid()
plt.legend()
plt.show()


#%% Plot demand

week = 9
days = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
hini = (week-1) * 2 * 24 * 7
hend = hini + 2 * 24 * 7
f3, (ax1, ax2) = plt.subplots(2, 1)
mb.plotWeek(conso_data.iloc[hini:hend,:], ax1, 'area', 2)
ax1.set_title('Demande Semaine %d' % week)
ax1.set_ylabel('[MW]')

cols = ['RES', 'PRO', 'Industrie', 'Tertiaire']
mb.plotWeek(conso_data_pu.loc[:,cols].iloc[hini:hend,:], ax2, 'line', 2)
ax2.set_title('Demande pu par type, Semaine %d' % week)
ax2.set_ylabel('[pu]')

#%

idxd = []
for x in days:
    idxd += [x] * 24 * 2 
idxh = np.repeat((np.arange(2*24)/2),7)
avg = pd.DataFrame(0, index=pd.DataFrame([idxd, idxh.tolist()]).transpose(), columns=conso_data.columns)
#%
i = 0
for t in np.arange(365):
   avg.iloc[i*24*2:(i+1)*24*2,:] += conso_data.iloc[t*2*24:(t+1)*2*24,:].values
   i += 1
   if i == 7:
       i = 0
       
avg = avg/52
avg.iloc[0:24,:] = avg.iloc[0:24,:] * 52/53 
#%
f4, (ax1) = plt.subplots(1, 1)
mb.plotWeek(avg, ax1, 'line', 2)
ax1.set_title('Profil moyen par type de demande, par jour de semaine')
ax1.set_ylabel('Demande [MW]')