# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:20:42 2019
Some useful functions
@author: U546416
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
#import pandas as pd

def distance2cities(npoints, size1, size2, d):
    """ Returns npoints points of distance, given a random distr of initial/final
    points within a given cities of size (t1/t2), at distance d
    """
    p1 = (np.random.rand(npoints, 2) -0.5) * np.sqrt(size1) 
    p2 = (np.random.rand(npoints, 2) -0.5) * np.sqrt(size2)    
    
    return d + np.sqrt(np.sum(p1*p1, axis=1)) + np.sqrt(np.sum(p2*p2, axis=1))
         
              
def hist(points, start, step, nbins):
    out = [0] * nbins
    for p in points:
        out[int(min((p-start) // step, nbins-1))] += 1
    return out

        
def readTgeo(fold, file):
    # Create dict of geoRefs
    Tgeo = {}
    with open(fold + file, 'r') as datafile:
        geo_reader = csv.reader(datafile, delimiter=';')
        headers_tgeo = next(geo_reader)
        #print(headers_tgeo)
        for geo_rows in geo_reader:
            Tgeo[geo_rows[0]] = geo_rows[1:3] +  [float(geo_rows[3])] + geo_rows[4:]
    print('Finish reading Tgeo')
    return Tgeo

def readModal(fold, file):
    # Create dict of geoRefs for Modal transport
    Modal = {}
    with open(fold + file, 'r') as datafile:
        reader = csv.reader(datafile, delimiter=';')
        headers = next(reader)
        #CODGEO;LIBGEO;ZE;Zone d'Emploi;Type;Status;UU;Dep;MC-AUCUN;MC-PIED;MC-2ROUES;MC-VOITURE;MC-TC;DC-AUCUN;DC-PIED;DC-2ROUES;DC-VOITURE;DC-TC
        idx_same_comm = headers.index('MC-AUCUN')
        idx_car_sc =  headers.index('MC-VOITURE')
        idx_diff_comm = headers.index('DC-AUCUN')
        idx_car_dc =  headers.index('DC-VOITURE')
        #print(headers_tgeo)
        for rows in reader:
            ts = sum(int(i) for i in rows[idx_same_comm:idx_same_comm+5])
            td = sum(int(i) for i in rows[idx_diff_comm:idx_diff_comm+5])
            if ts == 0:
                ratio_sc = 0
            else:
                ratio_sc = int(rows[idx_car_sc]) / ts
            if td == 0:
                ratio_dc = 0
            else:
                ratio_dc = int(rows[idx_car_dc]) / td
            Modal[rows[0]] = [ratio_sc, ratio_dc]
    print('Finish reading Modal')
    return Modal


def initFrCommDict(Tgeorefs, nzeros):
    """ Creates a dict of key: Code for french commune, 
    and initialization of a numpy array of nzeros
    Input, GeoRefs dict where GeoRefs[key][5] 
    is the UU of the commune, 99 for outside france"""
    Tout = {}
    for key in Tgeorefs:
        if not(Tgeorefs[key][5] == '99'):
            # UU==99 means its outside France
            Tout[key] = np.zeros(nzeros)
    return Tout


def plotWeek(data, axs, gtype, nperhour):
    """ Plots a data for a week and puts ticks marks at beginning of days
    """
    data.plot(ax=axs, kind=gtype)
    days = ['lundi', 'mardi', 'mercredi', 
            'jeudi', 'vendredi', 'samedi', 'dimanche']
    axs.grid(axis='x')
    axs.set_xticks(np.arange(7)*24*nperhour)
    axs.set_xticklabels(days)
    

def plotDist(bins, dist, ax):
    """ Plots a distribution with the mean as a red vertical line
    distribution given by vector dist, with bins, in a graphic ax
    """
    md = sum(dist*bins)/sum(dist)
    ax.bar(bins, dist)
    ax.axvline(x=md, color='red')
    ax.text(md+1, plt.ylim()[1]*0.7, 'Mean= %2.2f' %md)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('km')
    return ax


def computeDist(latlon1, latlon2):
    """Computes pythagorean distance between 2 points (need to be np.arrays)
    """
    radius=6371
    latlon1 = latlon1 * np.pi/180
    latlon2 = latlon2 * np.pi/180
    deltaLatLon = (latlon2-latlon1)
    x = deltaLatLon[1] * np.cos((latlon1[0]+latlon2[0])/2)
    y = deltaLatLon[0]
    return radius*np.sqrt(x*x + y*y)
    




    
