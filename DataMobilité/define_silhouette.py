# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:49:29 2019
Define outside shapes of a set of polygons:
    In this case, used for defining borders of SS areas, based on individual IRIS
@author: U546416
"""
import mobility as mb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe
import assign_ss_modif as ass_ss

#%% Reading commune, IRIS, & trafo data
print('loading data conso per commune')
fniris = 'IRIS.csv'
print('IRIS Conso')
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\\' + fniris, 
                   engine='python', index_col=0)

print('SS Data')
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', index_col=0)

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\iris_polygons.csv',
                        engine='python', index_col=0)
iris_polygons = {}
j = 0
#%% Constructing polygons
print('Constructing IRIS polygons')
for i in iris_poly[iris_poly.type == 'Polygon'].index:
    if j%1000 == 0:
        print('IRIS # {}'.format(j))
    j += 1
    iris_polygons[int(i)] = ptc.Polygon(eval(iris_poly.coords[i]))
for i in iris_poly[iris_poly.type == 'MultiPolygon'].index:
    if j%1000 == 0:
        print('IRIS # {}'.format(j))
    j += 1
    iris_polygons[int(i)] = ptc.Polygon(eval(iris_poly.coords[i])[0])
    
#%% Define outside shapes:
d = 0.00001
edges = {}
i = 0        
for ss in SS[SS.GRD == 'Enedis'].index:
    if ss in edges:
        continue
    i += 1
    if i%10 == 0:
        print(i)
    irises = iris[iris.SS == ss].index
    polys = [iris_polygons[irs] for irs in irises]
    if len(irises)==0:
        edges[ss] = []
        continue
    es = []
    for p in polys:
        # iterate over polygons, and over vertices, to define if they are exterior or interior points
        # It creates segments of silhouette points
        j += 1
        vs = p.get_verts()
        e = []
        for v in vs:
            plus = sum(ps.contains_point(v+d) for ps in polys)
            minus = sum(ps.contains_point(v-d) for ps in polys)
            if plus + minus < 2:
                e.append(v)
            else:
                if len(e)>=1:
                    es.append(e)
                    e = []        
        if len(e)>1:
            es.append(e)            
    while len(es) > 1:
        # Orders the segments of silhouette points to create a polygon
        ins = [e[0] for e in es]
        outs = [e[-1] for e in es]
        l = len(ins)
        ds = [(outs[j][0]-ins[0][0])**2 + (outs[j][1]-ins[0][1])**2  for j in range(1,l)]
        index_min = min(range(len(ds)), key=ds.__getitem__)
        es[0] = es[index_min+1] + es[0]
        del es[index_min+1]
    edges[ss] = es
    if i%200 == 0:
        f, ax = plt.subplots()
        ass_ss.plot_polygons(polys, ax, facecolors='lightgreen', edgecolors='c')
        out = ptc.Polygon(edges[ss])
        ass_ss.plot_polygons([out], ax, 'b', 'k')

edges = pd.DataFrame(edges).transpose()
edges.columns = ['Polygon']
edges.to_csv(r'c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source_edges.csv')

#%% Create SS polygons (& Plot SS)
poly_ss= {ss: ptc.Polygon(edges.Polygon[ss]) for ss in edges.index if len(edges.Polygon[ss])>2}
ass_ss.plot_polygons([p for p in poly_ss.values()], edgecolors='k')