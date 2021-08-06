# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:49:29 2019
Define outside shapes of a set of polygons:
    In this case, used for defining borders of SS areas, based on individual IRIS
@author: U546416
"""
#import mobility as mb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe
#import assign_ss_modif as ass_ss
import util
import time

def check_edge(point, polygons, d=0.00001):
    """ Checks if a point is in the interior of a set of polygons
    Returns True if point is in the Edge, False if it is in the Interior
    """
    plus = sum(ps.contains_point(point+d) for ps in polygons)
    minus = sum(ps.contains_point(point-d) for ps in polygons)
    return (plus + minus) < 2

def check_edge4(point, polygons, d=0.00001):
    """ Checks if a point is in the interior of a set of polygons
    Returns True if point is in the Edge, False if it is in the Interior
    """
    ds = [d, -d, [d,-d], [-d,d]]
    for di in ds:
        check = sum(ps.contains_point(point+di) for ps in polygons)
        if check == 0:
            return True
    return False

def get_edge_segments(polygons, use4=False):
    """ Get edge segments for a group of polygons
    """
    if use4:
        checkfx = check_edge4
    else:
        checkfx = check_edge
    segments = []
    for p in polygons:
        # iterate over polygons, and over vertices, to define if they are exterior or interior points
        # It creates segments of silhouette points
        vs = p.get_verts()
        e = []
        skip=False
        for v in vs:
            if skip:
                skip=False
                continue
            exterior = checkfx(v, polygons)
            v=list(v)
            unique = (sum([v in s for s in segments]) == 0)
            if exterior and unique:
                e.append(v)
            else:
                if len(e)>1:
                    segments.append(e)
                    e = []
                    skip = True
                else:
                    e = []
                    skip = True
        if len(e)>1:
            segments.append(e)
    return segments
    
def order_segments(segments):
    """Orders the segments of silhouette points to create a polygon. Assumes all segments are clockwise (or counterclockwise)
    """
    polygons = []
    while len(segments) > 1:
        ins = [e[0] for e in segments] # Extreme (begining) of each segment
        outs = [e[-1] for e in segments] # Extreme (end) of each segment
        l = len(ins) # number of segments
        # Distances from the begining of the first segment to the end each of the other segments
        ds = [(outs[j][0]-ins[0][0])**2 + (outs[j][1]-ins[0][1])**2  for j in range(1,l)] 
        # Distance between end-points of first segment
        d0 = (outs[0][0]-ins[0][0])**2 + (outs[0][1]-ins[0][1])**2
        # Select the closest segment
        index_min = min(range(len(ds)), key=ds.__getitem__)
        # if closest segment is farther away than closing the loop, closes the loop and creates new polygon
        if d0 < ds[index_min]:
            polygons.append(segments[0])
            del segments[0]
        else:
            # join the segments
            segments[0] = segments[index_min+1] + segments[0]
            del segments[index_min+1]
    polygons.append(segments[0])
    return polygons
        
def get_silhouette(polygons, use4=False):
    segments = get_edge_segments(polygons, use4=use4)
    return order_segments(segments)
    

#%% Reading commune, IRIS, & trafo data
print('loading data conso per IRIS')
fniris = 'IRIS_enedis_2017.csv'
print('IRIS Conso')
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\\' + fniris, 
                   engine='python', index_col=0)

print('SS Data')
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', index_col=0)

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
polygons = util.load_polygons_iris()

#%%
edges = {}
#%% Define outside shapes:
d = 0.00001
#edges = {}
i = 0 
ts = [time.time()]    
for ss in SS[SS.GRD == 'Enedis'].index:
    i += 1
    if i%10 == 0:
        ts.append(time.time())
        print(i, 'Elapsed time [s]:', np.round(ts[-1]-ts[0],1), 'Lap [s]:', np.round(ts[-1]-ts[-2],1))
    if ss in edges:
        continue
    irises = iris[iris.SS == ss].index
    polys = [p for irs in irises for p in polygons[irs]]
    if len(irises)==0:
        edges[ss] = [[]]
        continue
    sil = get_silhouette(polys, use4=True)
    edges[ss] = sil
    if (i+2)%200 == 0:
        f, ax = plt.subplots()
        util.plot_polygons(polys, ax, facecolors='lightgreen')
        util.plot_segments(sil, ax, color='k', linestyle='--')

edges = {ss : [p for p in edges[ss]] for ss in edges}
edges = pd.DataFrame(edges, index=['Polygon']).T


#%% Create SS polygons (& Plot SS)
polygons_ss= util.do_polygons(edges)
util.plot_polygons(util.list_polygons(polygons_ss, polygons_ss.keys()), edgecolors='k', linewidth=0.5)
util.plot_polygons(util.list_polygons(polygons_ss, SS[SS.Departement.isin(util.deps_idf)].index), edgecolors='k', linewidth=0.5)

#%% Do one - define outside shapes:
d = 0.00001
edges = {}
i = 0      
ss = 'ITTEVILLE'

irises = iris[iris.SS == ss].index
polys = [p for irs in irises for p in polygons[irs]]
if len(irises)==0:
    edges[ss] = []
sil = get_silhouette(polys, use4=True)
f, ax = plt.subplots()
util.plot_polygons(polys, ax, facecolors='lightgreen')
util.plot_segments(sil, ax, color='k', linestyle='--', ends=True)

#edges = pd.DataFrame(edges, index=['Polygon']).T
#%% Save
#edges = pd.DataFrame(edges, index=['Polygon']).T
edges.to_csv(r'c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source_polygons.csv')