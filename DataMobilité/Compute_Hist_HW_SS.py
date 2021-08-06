# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:40:37 2019
Compute 
@author: U546416
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe
#import assign_ss_modif as util
import util

    
#%% Reading commune, IRIS, & trafo data
print('loading data conso per commune')
fniris = 'IRIS_enedis_2017.csv'
print('IRIS Conso')
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\\' + fniris, 
                   engine='python', index_col=0)
#print('GeoData')
#Tgeo = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/geoRefs.csv', 
#                 engine='python', delimiter=';', index_col=0)
print('SS Data')
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', index_col=0)

SS_polys = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source_polygons.csv', 
                 engine='python', index_col=0)

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\IRIS_all_geo_2016.csv',
                        engine='python', index_col=0)
polygons = util.do_polygons(iris_poly)
print('Polygons SS')
polygons_ss = util.load_polygons_SS()

print('Load Profiles')
# Load conso profiles data (in pu (power, not energy))
load_profiles = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\conso_all_pu.csv', 
                           engine='python', delimiter=',', index_col=0)
# Load histograms of distances
print('Load Histograms')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
                    engine='python', index_col=0)

hhsc = pd.read_csv(folder_hdata + r'\hhome_samecom_modal.csv', 
                    engine='python', index_col=0)
hhdc = pd.read_csv(folder_hdata + r'\hhome_diffcom_modal.csv', 
                    engine='python', index_col=0)
hwdc = pd.read_csv(folder_hdata + r'\hwork_diffcom_modal.csv', 
                    engine='python', index_col=0)


#%% Plot type Iris
polyA = [p for i in iris_poly[iris_poly.IRIS_TYPE == 'A'].index for p in polygons[i]]
polyH = [p for i in iris_poly[iris_poly.IRIS_TYPE == 'H'].index for p in polygons[i]]
polyD = [p for i in iris_poly[iris_poly.IRIS_TYPE == 'D'].index for p in polygons[i]]
polyZ = [p for i in iris_poly[iris_poly.IRIS_TYPE == 'Z'].index for p in polygons[i]]
collectionA = PatchCollection(polyA, facecolors='r', label='Activité')
collectionH = PatchCollection(polyH, facecolors='b', label='Habitation')
collectionD = PatchCollection(polyD, facecolors='silver', label='Divers')
collectionZ = PatchCollection(polyZ, facecolors='g', label='Rural')

f, ax = plt.subplots()
ax.add_collection(collectionA)
ax.add_collection(collectionZ)
ax.add_collection(collectionD)
ax.add_collection(collectionH)
ax.autoscale()
plt.legend()

util.aspect_carte_france(ax=ax, labels=['Habitation', 'Activité', 'Rural', 'Divers'], palette=['b','r', 'g','silver'])

#%% Compute histograms per SS
print('Computing histograms per SS')
SS_hhome = {}
SS_hwork = {}
i = 0

hh = hhome.drop(['ZE','Status', 'UU', 'Dep'], axis=1)
hw = hwork.drop(['ZE','Status', 'UU', 'Dep'], axis=1)

#
print('Starting')
for ss in SS[SS.GRD == 'Enedis'].index:
    i +=1
    if i%100 == 0:
        print(i)
    irises = iris[iris.SS == ss]
    coms = irises[['COMM_CODE', 'Hab_pu', 'Work_pu']].groupby('COMM_CODE').sum()
    ss_hh = (hh.loc[coms.index,:].transpose() * (coms.Hab_pu)).transpose().sum(axis=0)
    ss_hw = (hw.loc[coms.index,:].transpose() * (coms.Work_pu)).transpose().sum(axis=0)
    SS_hhome[ss] = ss_hh
    SS_hwork[ss] = ss_hw


SS_hhome = pd.DataFrame(SS_hhome).T
SS_hwork = pd.DataFrame(SS_hwork).T
#SS_hhome.transpose().to_csv(folder_hdata + r'\HistHomeModal_SS.csv')

SS_hhome.to_csv(folder_hdata +  r'\HistHomeModal_SS.csv')
SS_hwork.to_csv(folder_hdata +  r'\HistWorkModal_SS.csv')

#%% Compute histograms per SS, for base histograms separated by workers in the same and different communes
print('Computing histograms per SS')
SS_hhdc = {}
SS_hhsc = {}
SS_hwdc = {}
i = 0

#
print('Starting')
for ss in SS[SS.GRD == 'Enedis'].index:
    i +=1
    if i%100 == 0:
        print(i)
    irises = iris[iris.SS == ss]
    coms = irises[['COMM_CODE', 'Hab_pu', 'Work_pu']].groupby('COMM_CODE').sum()
    ss_hhsc = (hhsc.loc[coms[coms.index.isin(hhsc.index)].index,:].transpose() * (coms.Hab_pu)).transpose().sum(axis=0)
    ss_hhdc = (hhdc.loc[coms[coms.index.isin(hhdc.index)].index,:].transpose() * (coms.Hab_pu)).transpose().sum(axis=0)
    ss_hwdc = (hwdc.loc[coms[coms.index.isin(hwdc.index)].index,:].transpose() * (coms.Work_pu)).transpose().sum(axis=0)
    SS_hhsc[ss] = ss_hhsc
    SS_hhdc[ss] = ss_hhdc
    SS_hwdc[ss] = ss_hwdc


SS_hhsc = pd.DataFrame(SS_hhsc).T
SS_hhdc = pd.DataFrame(SS_hhdc).T
SS_hwdc = pd.DataFrame(SS_hwdc).T
#SS_hhome.transpose().to_csv(folder_hdata + r'\HistHomeModal_SS.csv')

SS_hhsc.to_csv(folder_hdata +  r'\HistHomesamecomm_Modal_SS.csv')
SS_hhsc.to_csv(folder_hdata +  r'\HistHomediffcomm_Modal_SS.csv')
SS_hwdc.to_csv(folder_hdata +  r'\HistWorkdiffcomm_Modal_SS.csv')

#%% compute and plot Avg daily distance and KmVehicle per SS
bins = pd.Series(data=[i*2+1 for i in range(50)], index=SS_hhome.columns)
means = (SS_hhome * bins).transpose().sum(axis=0) / SS_hhome.sum(axis=1)
kmvoit = (SS_hhome * bins).transpose().sum(axis=0)
# Plot Average distance

km_base = 7.5
kmvoit_base = 150000


f, ax1 = plt.subplots()
f, ax2 = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
labels_km = [str(i*2*km_base) + ' km<d< ' + str((i+1)*2*km_base) + ' km' for i in range(len(palette))]
labels_kmv = [str(i*2*kmvoit_base/1000) + ' <k.km.veh< ' + str((i+1)*2*kmvoit_base/1000)  for i in range(len(palette))]

polys = [p for i in iris.index for p in polygons[i]]
kms = [palette[int(means[iris.SS[i]] // km_base)] for i in iris.index for p in polygons[i]]
kmsv = [palette[int(kmvoit[iris.SS[i]] // kmvoit_base)] for i in iris.index for p in polygons[i]]

util.plot_polygons(polys, ax1, facecolors=kms) #range every 15km of mean daily distance
util.plot_polygons(polys, ax2, facecolors=kmsv) #range every 150k kmvoit

util.aspect_carte_france(ax1, title='Average daily distance per Substation [km]\nFor Residents',
                    palette=palette, labels=labels_km)
util.aspect_carte_france(ax2, title='km' +r'$\bullet$' +'Vehicles\nFor Residents',
                    palette=palette, labels=labels_kmv)
         


#%% Plot for work trips
meansw = (SS_hwork * bins).transpose().sum(axis=0) / SS_hwork.sum(axis=1)
kmvoitw = (SS_hwork * bins).transpose().sum(axis=0)
    
f, ax1 = plt.subplots()
f, ax2 = plt.subplots()
# Plot avg km per car
#%Plot km*voiture by batch of 150.000 km*Veh/day (+- 30 MWh/day for 50% = EVs)

kms_w = [palette[min(int(meansw[iris.SS[i]] // km_base), len(palette)-1)] for i in iris.index for p in polygons[i]]
kmsv_w = [palette[min(int(kmvoitw[iris.SS[i]] // kmvoit_base), len(palette)-1)] for i in iris.index for p in polygons[i]]

    
util.plot_polygons(polys, ax1, facecolors=kms_w) #range every 15km of mean daily distance
util.plot_polygons(polys, ax2, facecolors=kmsv_w) #range every 15km of mean daily distance

util.aspect_carte_france(ax1, title='Average daily distance per Substation [km]\nFor Workers',
                         palette=palette, labels=labels_km)
util.aspect_carte_france(ax2, title='km' +r'$\bullet$' +'Vehicles\nFor Workers',
                         palette=palette, labels=labels_kmv)

#%% plot using SS
#Avg daily distance
#kmVehicle 
f, ax2 = plt.subplots()
f, ax1 = plt.subplots()
#palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
SSs = iris.SS.unique()

polyss = util.list_polygons(polygons_ss, SSs)

ss_km_res = [palette[int(means[ss] // km_base)] for ss in SSs for p in polygons_ss[ss]]
ss_kmv_res = [palette[int(kmvoit[ss] // kmvoit_base)] for ss in SSs for p in polygons_ss[ss]]


util.plot_polygons(polyss, ax1, facecolors=ss_km_res, edgecolors='k', linestyle='--', linewidth=0.15)
util.plot_polygons(polyss, ax2, facecolors=ss_kmv_res, edgecolors='k', linestyle='--', linewidth=0.15)

util.aspect_carte_france(ax1, title='Average daily distance per Substation [km]\nFor Residents',
                             palette=palette, labels=labels_km)
util.aspect_carte_france(ax2, title='km' +r'$\bullet$' +'Vehicles\nFor Residents',
                         palette=palette, labels=labels_kmv)

#%% Plot for workers
f, ax2 = plt.subplots()
f, ax1 = plt.subplots()

ss_km_w = [palette[min(int(meansw[ss] // km_base), len(palette)-1)] for ss in SSs for p in polygons_ss[ss]]
ss_kmv_w = [palette[min(int(kmvoitw[ss] // kmvoit_base), len(palette)-1)] for ss in SSs for p in polygons_ss[ss]]


util.plot_polygons(polyss, ax1, facecolors=ss_km_w, edgecolors='k', linestyle='--', linewidth=0.15)
util.plot_polygons(polyss, ax2, facecolors=ss_kmv_w, edgecolors='k', linestyle='--', linewidth=0.15)

util.aspect_carte_france(ax1, title='Average daily distance per Substation [km]\nFor Workers',
                             palette=palette, labels=labels_km)
util.aspect_carte_france(ax2, title='km' +r'$\bullet$' +'Vehicles\nFor Workers',
                         palette=palette, labels=labels_kmv)  

#%% Plot for IdF
deps_idf = [75, 77, 78, 91, 92, 93,  94, 95]
ssidf = SS[(SS.GRD == 'Enedis') & (SS.Departement.isin(deps_idf))].index

# Avg distance &
# kmVoitures
f, ax1 = plt.subplots()
f, ax2 = plt.subplots()
f, ax3 = plt.subplots()
f, ax4 = plt.subplots()

polys_idf = util.list_polygons(polygons_ss, ssidf)
ss_res_km = [palette[int(means[ss] // km_base)] for ss in ssidf for p in polygons_ss[ss]]
ss_res_kmv = [palette[int(kmvoit[ss] // kmvoit_base)] for ss in ssidf for p in polygons_ss[ss]]
ss_w_km = [palette[min(int(meansw[ss] // km_base), len(palette)-1)] for ss in ssidf for p in polygons_ss[ss]]
ss_w_kmv = [palette[min(int(kmvoitw[ss] // kmvoit_base), len(palette)-1)] for ss in ssidf for p in polygons_ss[ss]]

util.plot_polygons(polys_idf, ax1, facecolors=ss_res_km, edgecolors='k', linestyle='--', linewidth=0.15)
util.plot_polygons(polys_idf, ax2, facecolors=ss_res_kmv, edgecolors='k', linestyle='--', linewidth=0.15)
util.plot_polygons(polys_idf, ax3, facecolors=ss_w_km, edgecolors='k', linestyle='--', linewidth=0.15)
util.plot_polygons(polys_idf, ax4, facecolors=ss_w_kmv, edgecolors='k', linestyle='--', linewidth=0.15)

util.aspect_carte_france(ax1, title='Average daily distance per Substation [km]\nFor Residents', 
                    palette=palette, labels=labels_km, cns='idf')
util.aspect_carte_france(ax2, title='km'+r'$\bullet$'+ 'Vehicle'+ '\nFor Residents',
                    palette=palette, labels=labels_kmv, cns='idf')  
util.aspect_carte_france(ax3, title='Average daily distance per Substation [km]\nFor Workers', 
                    palette=palette, labels=labels_km, cns='idf')
util.aspect_carte_france(ax4, title='km'+r'$\bullet$'+ 'Vehicle'+ '\nFor Workers',
                    palette=palette, labels=labels_kmv, cns='idf')  

    
#%%
ratio_hw_ss = kmvoitw/kmvoit
ratio_hw_ss = ratio_hw_ss.replace(np.inf, 100)
ratio_hw_ss = ratio_hw_ss.replace(np.nan, 0)
ratio_hw_ss = ratio_hw_ss.apply(lambda x: 2 - 1/x if x > 1 else x)

cmap = plt.get_cmap('viridis')
ss_colors = cmap([ratio_hw_ss[ss]/2 for ss in SSs for p in polygons_ss[ss]])
ss_colors_idf = cmap([ratio_hw_ss[ss]/2 for ss in ssidf for p in polygons_ss[ss]])
ax = util.plot_polygons(polyss, facecolors=ss_colors, edgecolors='k', linestyle='--', linewidth=0.15)
ax2 = util.plot_polygons(polys_idf, facecolors=ss_colors, edgecolors='k', linestyle='--', linewidth=0.15)

palette = cmap(np.linspace(0,1,7))
labels=['High Residents','.', '.', 'Equivalent', '.','.', 'High Workers']

util.aspect_carte_france(ax, title='Ratio km.veh Residents over Workers', 
                    palette=palette, labels=labels, cns='France')
util.aspect_carte_france(ax2, title='Ratio km.veh Residents over Workers', 
                    palette=palette, labels=labels, cns='idf')