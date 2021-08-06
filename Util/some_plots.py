# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:40:58 2019

@author: U546416
"""
import numpy as np
import util
import matplotlib.pyplot as plt

kmvhiris = (hh.loc[iris_poly.COMM_CODE,:] * bins).sum()
kmvwiris = (hw.loc[iris_poly.COMM_CODE,:] * bins).sum()

kmvhiris.index = iris_poly.index
kmvwiris.index = iris_poly.index

#%%

ratio_wh = kmvwiris / kmvhiris
ratio_wh = ratio_wh.replace(np.inf, 100)
ratio_wh = ratio_wh.replace(np.nan, 0)
#ratio_wh = ratio_wh.clip_upper(2)


ratio_wh = ratio_wh.apply(lambda x: 2 - 1/x if x > 1 else x)
#%% 

cmap = plt.get_cmap('Purples')

polys = util.list_polygons(polygons, polygons.keys())

colors = cmap([ratio_wh[i] / 2 for i in polygons for p in polygons[i]])

palette = cmap([i/2/2 for i in range(5)])
labels=['High Workers', '.','.', 'Equivalent', '.','.', 'High Residents']

ax = util.plot_polygons(polys, color=colors)
util.aspect_carte_france(ax, palette=palette, labels=labels)
#%%
cmap = plt.get_cmap('viridis')

polys = util.list_polygons(polygons, polygons.keys())

colors = cmap([ratio_wh[i] / 2 for i in polygons for p in polygons[i]])

palette = cmap(np.linspace(0,1,7))
labels=['High Residents','.', '.', 'Equivalent', '.','.', 'High Workers']

ax = util.plot_polygons(polys, color=colors)
util.aspect_carte_france(ax, palette=palette, labels=labels)

#%% 
iris_idf = iris_poly[(iris_poly.COMM_CODE // 1000).astype(int).isin(util.deps_idf)].index
polysidf = util.list_polygons(polygons, iris_idf)
coloridf = cmap([ratio_wh[i] / 2 for i in iris_idf for p in polygons[i]])

ax = util.plot_polygons(polysidf, color=coloridf)
util.aspect_carte_france(ax, palette=palette, labels=labels, cns='idf')