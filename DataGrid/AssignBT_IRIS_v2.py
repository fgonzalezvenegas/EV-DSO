# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:16:28 2020
Assigning BTs to IRIS

Using IRIS 2016 on data from cartographie reseaux Enedis (06-2020)

@author: U546416
"""
import pandas as pd
from grid import *
import util
import coord_transform as ct

folder = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
postesbt = pd.read_csv(folder + 'poste-electrique.csv',
                       engine='python', sep=';')
# formatting
postesbt = postesbt['Geo Shape'].apply(lambda x: eval(x)['coordinates'])

#%% Load IRIS polygon data
print('Loading IRIS polygons')
# TODO: Load IRIS polygons
folder_iris = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
iris_poly = pd.read_csv(folder_iris+file_iris,
                        engine='python', index_col=0)
iris_poly.Polygon = pd.Series(util.do_polygons(iris_poly, plot=False))

iris_polys = iris_poly[['Polygon', 'IRIS_NAME', 'Lon', 'Lat']]
iris_polys.columns = ['Polygon', 'Name', 'xGPS', 'yGPS']
print('\tDone loading polygons')

#%% Transform to WGPS

xyGPS = postesbt.apply(lambda x: ct.point_LAMB93CC_WGS84((x[0], x[1]), cc=8))
postesbt = pd.DataFrame([xyGPS.apply(lambda x: x[0]), xyGPS.apply(lambda x: x[1])], index=['xGPS', 'yGPS']).T

#%% Searching polygon
assign_polys(postesbt, iris_polys, 0.05)
# Searching polygons in a larger area for those that didnt get one
nonass = postesbt[postesbt.Geo.isnull()]
print(len(nonass))
assign_polys(nonass, iris_polys, dt=0.05, notdt=0.5)
postesbt.Geo[nonass.index] = nonass.Geo
postesbt.Geo_Name[nonass.index] = nonass.Geo_Name
nonass = postesbt[postesbt.Geo.isnull()]
print(len(nonass))
# Assigning remaining BT to closer IRIS (LV that are located in the water!)
for n, t in nonass.iterrows():
    d = (t.xGPS-iris_polys.xGPS)**2 + (t.yGPS-iris_polys.yGPS)**2
    nonass.Geo[n] = d.idxmin()
postesbt.Geo[nonass.index] = nonass.Geo
postesbt.Geo_Name[nonass.index] = iris_polys.Name[nonass.Geo]


#%% Saving and Counting polygons

postesbt.to_csv(folder + '/poste-electrique-assignedIRIS2016.csv')
nb_bt = postesbt.groupby(['Geo', 'Geo_Name']).xGPS.count()
nb_bt.name = 'Nb_BT'
nb_bt.index.names = ['IRIS_CODE', 'IRIS_NAME']
nb_bt.to_csv(folder + '/Nb_BT_IRIS2016.csv')
