# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:57:54 2020

@author: felip_001
"""
import pandas as pd
import numpy as np


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

def point_LAMB93_WGS84(xy):
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

def L_LAM93_transform(lat, e):
    return 1/2*np.log((1+np.sin(lat))/(1-np.sin(lat))) - e/2 * np.log((1+e*np.sin(lat))/(1-e*np.sin(lat)))

def n_LAM93_transform(lat1, lat2, a, e):
    
    c_lat1 = a * np.cos(lat1) / np.sqrt(1 - e**2 * np.sin(lat1)**2)
    c_lat2 = a * np.cos(lat2) / np.sqrt(1 - e**2 * np.sin(lat2)**2)
    dL =  L_LAM93_transform(lat1,e) - L_LAM93_transform(lat2,e)
    
    return np.log(c_lat2/c_lat1) / dL

def c_LAM93_transform(lat, a, e, n):
    c0 = a * np.cos(lat) / np.sqrt(1 - e**2 * np.sin(lat)**2)
    ex = np.exp(n * L_LAM93_transform(lat, e))
    
    return c0 /n * ex
    
def point_LAMB93CC_WGS84(xy, cc=8):
    """ Returns geo coordinates from LAMBERT 93 Conique Conforme french projection system,
    Needs the number of the Conique Conforme zone (1-9)
    By default we use zone 9, corresponding to Paris area
    """
    a = 6378137                                 # Demi grand axe
    e = 0.08181919112                           # First excentricity
    
    lat0 = (41+cc)  * np.pi/180                   # Latitude of origine in rads
    lat1 = lat0 - 0.75 * np.pi/180              # Automecoique parallel
    lat2 = lat0 + 0.75 * np.pi/180              # Automecoique parallel

    lon0 = 3*np.pi/180  # IERS Meridian #(2 + 20/60 + 14.025/3600) * np.pi / 180 # degrees

    Eo = 1700000
    No = 1000000 * cc + 200000

    
    n = n_LAM93_transform(lat1, lat2, a, e)
    C = c_LAM93_transform(lat1, a, e, n)
    xs = Eo
    ys = No + C * np.exp(-n * L_LAM93_transform(lat0, e))
    #e = 0.08248325676
    
    R = np.sqrt((xy[0]-xs)**2+(xy[1]-ys)**2)
    y = np.arctan((xy[0] - xs)/(ys - xy[1]))
    
    lat_iso = -1/n * np.log(abs(R/C))
    
    lon = lon0 + y/n 
    lat = lat_iso_lat(lat_iso, e)
    
    return lon*180/np.pi, lat*180/np.pi

def polygon_LAMB93_WGS84(points):
    """ Returns geo coordinates from LAMBERT 93 french projection system
    """
    return [point_LAMB93_WGS84(xy) for xy in points]

def polygon_LAMB93CC_WGS84(points, cc=8):
    """ Returns geo coordinates from LAMBERT 93 Conic Conformal french projection system
    """
    return [point_LAMB93CC_WGS84(xy) for xy in points]

def coords(polygon):
    """ Returns a representative point for the polygon
    """
    return np.mean(np.array(polygon), axis=0)

