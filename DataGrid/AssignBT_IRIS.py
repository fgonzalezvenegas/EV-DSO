# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:34:14 2019
Assign HTA/BT substations to IRIS
@author: U546416
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches as ptc 
import time

# Load data
print('Loading Data')
print('IRIS')
#folder_geodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\GeoData'
#iris = pd.read_csv(folder_geodata + r'\IRIS.csv', 
#                    engine='python', index_col=0)

# BT SS
print('Load Secondary SS')
# data base, not treated
#BT = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\PosteSource\poste-electrique.csv',
#                 engine='python', sep=';')
#read BT already assigned
BT =  pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau\postes_BT.csv',
                 engine='python', index_col=0)

print('Polygons')
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\GeoData\IRIS_all_geo.csv',
                        engine='python', index_col=0)
# constructing iris polygons
print('Constructing IRIS polygons')
iris_polygons = {i: [ptc.Polygon(p)] for i in iris.index for p in eval(iris.Polygon[i])}
print('Ready!')
#%% Formatting BT SS

#Lat = BT['Geo Point'].apply(lambda x: eval(x)[0])
#Lon = BT['Geo Point'].apply(lambda x: eval(x)[1])
#
#BT = pd.DataFrame([Lat, Lon], index=['Lat', 'Lon']).transpose()

#%% Assigning BT to IRIS, for all BTs
bt_iris = {}
bt_noiris = []
i=0
start = time.time()
t0 = start
print('Starting assignment')
for bt in BT.index:
    i += 1
    if i%10000 == 0:
        t1 = time.time()
        lap = t1-t0
        t0 = t1
        print(i, ';\tAssigned BTSS:', len(bt_iris), ';\tLap (s):', np.round(lap,1), ';\tTotal (s):', np.round(t1-start,1))
    lat, lon = BT[['Lat', 'Lon']].loc[bt]
    y = True
    for ies in iris[(iris.Lat > lat-0.1) & (iris.Lat < lat + 0.1) &
                    (iris.Lon > lon-0.1) & (iris.Lon < lon + 0.1)].index:
        for p in iris_polygons[ies]:
            if p.contains_point((lon, lat)):
                bt_iris[bt] = [ies, 'CHECKSS', iris.Code_Comm[ies]]
                y = False
                break
        if not y:
            break
    if y:
        bt_noiris.append(bt)
print('Finished')
print(i, ';\tAssigned BTSS:', len(bt_iris), ';\tLap (s):', np.round(lap,1), ';\tTotal (s):', np.round(t1-start,1))
bt_iris = pd.DataFrame(bt_iris, index=['Code_IRIS', 'SS', 'Code_commune']).transpose()



#%% Redo with wider range
#bt_nan = BT[BT.Code_IRIS.isnull()].index
bt_nan = bt_noiris
i=0
start = time.time()
t0 = start
bt_iris = {}
bt_noiris = []
print('Start')
for bt in bt_nan:
    i += 1
    if i%100 == 0:
        t1 = time.time()
        lap = t1-t0
        t0 = t1
        print(i, ';\tAssigned BTSS:', len(bt_iris), ';\tLap (s):', np.round(lap,1), ';\tTotal (s):', np.round(t1-start,1))
    lat, lon = BT[['Lat', 'Lon']].loc[bt]
    y = True
    for ies in iris[(iris.Lat > lat-1) & (iris.Lat < lat + 1) &
                    (iris.Lon > lon-1) & (iris.Lon < lon + 1)].index:
        for p in iris_polygons[ies]:
            if p.contains_point((lon, lat)):
                bt_iris[bt] = [ies, 'CHECK', iris.Code_Comm[ies]]
                y = False
                break
        if not y:
            break 
            
    if y:
        ies = iris[(iris.Lat > lat-1) & (iris.Lat < lat + 1) &
                    (iris.Lon > lon-1) & (iris.Lon < lon + 1)]
        d = pd.Series({j: (lat-ies.Lat[j])**2 + (lon-ies.Lon[j])**2 for j in ies.index}).idxmin()
        bt_iris[bt] = [d, 'CHECK', iris.Code_Comm[d]]
bt_iris = pd.DataFrame(bt_iris, index=['Code_IRIS', 'SS', 'Code_commune']).transpose()
BT.loc[bt_iris.index, ['Code_IRIS', 'SS', 'Code_commune']] = bt_iris
print('Finished')
print(i, ';\tAssigned BTSS:', len(bt_iris), ';\tLap (s):', np.round(lap,1), ';\tTotal (s):', np.round(t1-start,1))


#%% Save
BT.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau\postes_BT.csv')

#%% Read all iris
#iris_all = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\GeoData\iris.csv', 
#            engine='python', sep=';')
#Lat = iris_all['Geo Point'].apply(lambda x: eval(x)[0])
#Lon = iris_all['Geo Point'].apply(lambda x: eval(x)[1])
#iris_all['Lat'] = Lat
#iris_all['Lon'] = Lon
##%% do polygons
#polys2 = {}
#for i in iris_all.index:
#    code, geo = iris_all.CODE_IRIS[i], eval(iris_all['Geo Shape'][i])
#    if geo['type'] == "Polygon":
#        polys2[code] = [ptc.Polygon(geo['coordinates'][0])]
#    else:
#        polys2[code] = [ptc.Polygon(geo['coordinates'][i][0]) for i in range(len(geo['coordinates']))]
##        
#polys2 = {iris_all.CODE_IRIS[i] : ptc.Polygon(eval(iris_all['Geo Shape'][i])["coordinates"][0])
#                for i in iris_all.index
#                if eval(iris_all['Geo Shape'][i])["type"] == "Polygon"}
