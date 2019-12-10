# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:46:51 2019
IRIS ShapeFile from IGN-INSEE in LAMB93 projection to WGS4 (geocoords system) system.
Saved to a Pandas DF
@author: U546416
"""

import shapefile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
import time
import util

def lat_iso_lat(lat_iso, e):
    tol = 1e-11
    lats = []
    lat0 = 2 * np.arctan(np.exp(lat_iso)) - np.pi/2
    lats.append(lat0)
    tol_nok = True
    i=0
    while tol_nok:
        i=i+1
        lat_i = 2 * np.arctan(
                (((1 + e * np.sin(lats[-1]))/(1 - e * np.sin(lats[-1]))) ** (e/2)) *
                np.exp(lat_iso)) - np.pi/2
        lats.append(lat_i)
        if abs(lats[-1]-lats[-2]) < tol:
            tol_nok = False
    #print(lats)
    return lats[-1]

def point_LAMB938_WGS84(xy):
    """ Returns geo coordinates from LAMBERT 93 french projection system
    """
    n = 0.7256077650
    C = 11754255.426
    xs = 700000.00
    ys = 12655612.050
    e = 0.08181919112
    #e = 0.08248325676
    lon0 = 3*np.pi/180  # IERS Meridian #(2 + 20/60 + 14.025/3600) * np.pi / 180 # degrees
    
    R = np.sqrt((xy[0]-xs)**2+(xy[1]-ys)**2)
    y = np.arctan((xy[0] - xs)/(ys - xy[1]))
    
    lat_iso = -1/n * np.log(abs(R/C))
    
    lon = lon0 + y/n 
    lat = lat_iso_lat(lat_iso, e)
    
    return lon*180/np.pi, lat*180/np.pi

def polygon_LAMB93_WGS84(points):
    """ Returns geo coordinates from LAMBERT 93 french projection system
    """
    return [point_LAMB938_WGS84(xy) for xy in points]

def coords(polygon):
    """ Returns a representative point for the polygon
    """
    return np.mean(np.array(polygon), axis=0)
    #%% Read shapefile and transform iris contours from LAMBERT 93 to WGS84

# c:\user\U546416\Documents\PhD\Data\DataGeo\CONTOURS-IRIS_2-1_SHP_LAMB93_FE-2016
shape = shapefile.Reader(r'c:\user\U546416\Documents\PhD\Data\DataGeo\CONTOURS-IRIS_2-1_SHP_LAMB93_FE-2016\CONTOURS-IRIS.shp',
                         encoding="ISO-8859-1")

#fields = [f[0] for f in shape.fields[1:]]
fields = ['INSEE_COM', 'NOM_COM', 'IRIS','CODE_IRIS','NOM_IRIS','TYP_IRIS']

iris = {}
iris_poly = {}
outs={}
i = 0
mp = 0
ts = [time.time()]
while i<len(shape):
    if i%10000 == 0:
        ts.append(time.time())
        print(i, ';\tAssigned IRIS:', len(iris), ';\tMultiPolygons:', mp, ';\tLap (s):', np.round(ts[-1]-ts[-2],1), ';\tTotal (s):', np.round(ts[-1]-ts[0],1))
    s = shape.shapeRecord(i)
    try:
        code = int(s.record.CODE_IRIS)
    except:
        i+=1
        continue
    rec = s.record
    iris[code] = [int(rec[0])] + [rec[1]] + rec[4:]
    if len(s.shape.parts)==1:
        #Normal Polygon
        iris[code].append('Polygon')
        iris_poly[code] = [polygon_LAMB93_WGS84(s.shape.points)]
    else:
        iris[code].append('MultiPolygon')
        parts = list(s.shape.parts) + [len(s.shape.points)]
        poly = polygon_LAMB93_WGS84(s.shape.points)
        iris_poly[code] = [poly[parts[i]:parts[i+1]] for i in range(len(parts)-1)]
        mp +=1
    i+=1
ts.append(time.time())
print('FINISHED')
print(i, ';\tAssigned IRIS:', len(iris), ';\tMultiPolygons:', mp, ';\tLap (s):', np.round(ts[-1]-ts[-2],1), ';\tTotal (s):', np.round(ts[-1]-ts[0],1))



#%%
polygons = {c: [ptc.Polygon(p) for p in iris_poly[c]] for c in iris_poly.keys()}
util.plot_polygons([p for pp in polygons.values() for p in pp])
#%%saving
iris = pd.DataFrame(iris, index=['Code_Comm', 'Nom_Comm', 'Nom_IRIS', 'IRIS_Type', 'PolygonType']).transpose()
#iris.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\GeoData\IRIS_all_geo.csv')
iris['Polygon'] = pd.Series(iris_poly)
iris['Lon'] = iris.Polygon.apply(lambda x: np.mean(np.asarray(x[0]), axis=0)[0])
iris['Lat'] = iris.Polygon.apply(lambda x: np.mean(np.asarray(x[0]), axis=0)[1])
iris.index.name = 'CODE_IRIS'
iris.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\GeoData\IRIS_all_geo_2016.csv')
