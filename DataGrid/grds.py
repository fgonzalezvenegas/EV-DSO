# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:00:46 2019

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
import util
import requests

print('IRIS')
iris =pd.read_csv(r'c:\Users\U546416\Downloads\donnees_elec_2017.csv', 
                   engine='python', sep=';', decimal=',')
comms = iris[iris.TYPE == 'Commune'].set_index('CODE')
comms = comms[comms.OPERATEUR.isin(['EdF-SEI', 'Régie électrique de Villarlurin']) == False]
comms.index = comms.index.astype(int)

iris = iris[iris.TYPE == 'IRIS'].set_index('CODE')
iris.index = iris.index.astype(int)

print('Polygons')
iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\IRIS_all_geo_2016.csv',
                        engine='python', index_col=0)

#% Constructing polygons
print('Constructing polygons')
print('IRIS polygons')
polygons = util.do_polygons(iris_poly, plot=True)


def get_comm_str(inseecode):
    comm = str(inseecode)
    return '000'[0:5-len(comm)] + comm
##%% 
#iris_poly.columns = ['COMM_CODE', 'COMM_NAME', 'IRIS_NAME', 'IRIS_TYPE', 'PolygonType',
#       'Polygon', 'Lon', 'Lat', 'GRD']

#%%
GRD = iris[iris.OPERATEUR != 'RTE'].OPERATEUR.loc[iris_poly.index]
GRDnull = GRD[GRD.isnull()]

# remove repeated IRIS
a = pd.DataFrame([GRD.index, GRD.values]).T
GRD = GRD[(a.duplicated(subset=0, keep='last')==False).values]


#%% Check if GRD is in already given for the commune
for i in GRDnull.index:
    com = iris_poly.COMM_CODE[i]
    if com in comms.index:
        grd = comms.OPERATEUR[com]
        if type(grd) == str:
            GRD.loc[i] = grd
        else:
            GRD.loc[i] = grd.iloc[0]
GRDnull = GRD[GRD.isnull()]

#%% save
iris_poly.to_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\IRIS_all_geo_2016.csv')
 
    #%%

ads = r'http://listegrd.adeef.fr/index.php?eID=routing&route=annuaire_grd/grd/list/'
proxies = {
  'http': r'http://U546416:P0l3r412@http.ntlm.internetpsa.inetpsa.com:8080'
}

for k in range(10):
    print('KKKKKK=',k, '# to requests:', len(GRDnull))
    j=0
    for i in GRDnull.index:
        j+=1
        if j%10==0:
            print(j)
        comm = get_comm_str(i // 10000)
        r=requests.get(ads + comm, proxies=proxies)
        if r.status_code == 200:
            if 'nom' in r.json()[0]:
                GRD.loc[i] = r.json()[0]['nom']    
    
    
    GRDnull = GRD[GRD.isnull()]




#%%
palette=['b','g','yellow','violet','purple','y','mediumslateblue','orange', 'c', 'silver']
f,ax= plt.subplots()
i=0

dsos = ['Enedis', 'GEREDIS Deux-Sèvres', 'SRD', 
        'URM (Metz)', 'Electricité de Strasbourg',
        'RSE - Régie Services Energie','Gaz électricité de Grenoble', 
        'SICAE-Oise', 'SICAE-Est',
        'Autres']

for dso in dsos:
    if dso == 'Autres':
        ies = iris_poly[iris_poly.GRD.isin(dsos[:-1]) == False].index
    else:
        ies = iris_poly[iris_poly.GRD == dso].index
    polys = [p for pp in ies for p in polygons[pp]]
    util.plot_polygons(polys, facecolor=palette[i], ax=ax, label=dso)
    i=i+1
util.aspect_carte_france(ax, palette=palette, labels=dsos)