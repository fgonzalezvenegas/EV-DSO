# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:04:09 2019
Reads Commune json file and plots Avg daily distance
@author: U546416
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polygons as pg
import matplotlib.patches as ptc
from matplotlib.collections import PatchCollection

#%% Read files
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', delimiter=',', decimal = '.', index_col=0)
iris = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
                   engine='python', index_col=2, delimiter=';')

iris_red = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\IRIS.csv', 
                   engine='python', index_col=0)
     
#%% Read json from OpenStreetMap
cosm = pd.read_json(r'C:\Users\u546416\Downloads\communes-20190101.json')
#%% Constructs set of Communes polygons      
osmpolygons = {}
j = 0
for i in cosm.index:
    if j%1000 == 0:
        print('CommOSM # {}'.format(j))
    j += 1
    try:
        poly = cosm.features[i]['geometry']
        code = cosm.features[i]['properties']['insee']
        if poly['type'] == 'Polygon':
            osmpolygons[code] = pg.Polygons(poly['coordinates'][0])
        if poly['type'] == 'MultiPolygon':
            osmpolygons[code] = pg.multiPolygons(poly['coordinates'])
    except:
        pass
# Creates pd.Series lat and lon for osmpolygons
latlon = np.zeros((len(osmpolygons),2))
idx = []
i = 0
for p in osmpolygons:
    latlon[i] = osmpolygons[p].get_center()
    idx.append(p)
    i += 1
osmLon = pd.Series(latlon[:,0], index=idx)
osmLat = pd.Series(latlon[:,1], index=idx)
    

#%% Read km data and plot polygons
comkm = pd.read_excel(r'c:\user\U546416\Documents\PhD\Data\Mobilité\flux-DataGeo.xlsx', sheetname='GeoDonnées', index_col=0)
#%% Draw all polygons

#Comunnes in Metro France (w/o Corse)
cs = osmLon[(osmLat < 51.2) & (osmLat > 42) & 
            (osmLon <  8.3) & (osmLon > -5.05)].index
pypolys = []
colors = []
ckm = comkm[(comkm['kmMoyen-Hab'] == comkm['kmMoyen-Hab']) & (comkm['#VoituresEnedis'] > 50)]
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
pypolysout = []

for c in cs:
    if c in ckm.index:
        try:
            pypolys.append(ptc.Polygon(osmpolygons[c].corners))
            colors.append(palette[int(min(ckm.loc[c, 'kmMoyen-Hab']//7.5, 6))])
        except:
            for pp in osmpolygons[c].polygons:
                pypolys.append(ptc.Polygon(pp.corners))
                colors.append(palette[int(min(ckm.loc[c, 'kmMoyen-Hab']//7.5, 6))])
    else:
        try:
            pypolysout.append(ptc.Polygon(osmpolygons[c].corners))
        except:
            for pp in osmpolygons[c].polygons:
                pypolysout.append(ptc.Polygon(pp.corners))
    
#%%
f, ax = plt.subplots()                
p = PatchCollection(pypolys, facecolors=colors)
ax.add_collection(p)
ax.autoscale()       

# write names of some comms
import matplotlib.patheffects as pe
# names of communes in comkm DataFrame
cnames = ['Paris', 'Marseille', 'Lyon', 'TOULOUSE', 'BORDEAUX', 'NANTES', 'LILLE', 'RENNES', 'STRASBOURG']
# Names to print in map
cns = ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Bordeaux', 'Nantes', 'Lille', 'Rennes', 'Strasbourg']
idxs = [comkm[comkm['Nom Commune'] == c].index for c in cnames]
for i in range(len(cns)):
    ax.text(osmLon[idxs[i]],osmLat[idxs[i]]+0.2, cns[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])
    
# add SS 
ssnames = ['VANVES', 'VERFEIL']
ss = ax.plot(SS.Lon[ssnames], SS.Lat[ssnames], '^', markersize=8, color='m')
