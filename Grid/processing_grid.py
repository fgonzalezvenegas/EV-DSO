# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:30:46 2020

@author: felip_001
"""

from grid import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from time import time
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
#import coord_transform as ct
import util

#%% OPTIONAL: Load processed data
print('Loading MV grid data')
folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\\'
subf = r'ProcessedData//'
hta = pd.read_csv(folder + subf + 'MVLines_full.csv', engine='python', index_col=0)
ps = pd.read_csv(folder + subf + 'SS_full.csv', engine='python', index_col=0)
bt = pd.read_csv(folder + subf + 'MVLV_full.csv', engine='python', index_col=0)
fnodes = pd.read_csv(folder + subf + 'Nodes_full.csv', engine='python', index_col=0)

hta.ShapeGPS = hta.ShapeGPS.apply(eval)
#%% OPTIONAL: Load IRIS data
print('Loading IRIS polygons')
# TODO: Load IRIS polygons
folder_iris = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
iris_poly = pd.read_csv(folder_iris+file_iris,
                        engine='python', index_col=0)
iris_poly.Polygon = pd.Series(util.do_polygons(iris_poly, plot=False))
print('\tDone loading polygons')
#%% OPTIONAL: Load conso data
print('Loading Conso per IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
print('Loading profiles in p.u.')
profiles = pd.read_csv(folder_consodata + r'\conso_all_pu.csv',
                       engine='python', index_col=0)
profiles.drop(['ENT', 'NonAffecte'], axis=1, inplace=True)

#%% OPTIONAL Load tech data
print('Loading tech data')
folder_tech = r'c:\user\U546416\Documents\PhD\Data\MVGrids\\'
file_tech = 'line_data_France_MV_grids.xlsx'
tech = pd.read_excel(folder_tech + file_tech, index_col=0)

#%% Loading number of LV transfos per IRIS
# load # BT/iris:
folder_lv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
lv_iris = pd.read_csv(folder_lv+'Nb_BT_IRIS2016.csv',
                      engine='python', index_col=0)
#%% Assigning Geographic zone to each LV load
if not 'Geo_Name' in bt.columns:
    assign_polys(bt, polys)
if not 'Annual_Load_MWh' in bt.columns:
    bt['Annual_Load_MWh'] = equal_total_load(bt,iris,lv_per_geo=lv_iris.Nb_BT)
if not 'Pmax_MW' in bt.columns:
    # creating profile for each IRIS:
    profs = {}
    for geo in bt.Geo:
        profs[geo] = (profiles * iris[['Conso_PRO', 'Conso_RES', 'Conso_Agriculture', 'Conso_Tertiaire', 'Conso_Industrie']].loc[geo].values).sum(axis=1)
    profs = pd.DataFrame(profs)
    profs = profs / profs.max() # profiles in pu, with peak load==1
    bad_cols = profs.max()[profs.max(axis=0).isnull()].index
    profs[bad_cols] = 0
    # profs[iris] is a yearly profile with max=1
    # peak_factor is the peak_load [MW] / annual_load [MW] (/2 bcs i have 30min values)
    power_factor = 1/(profs.sum()/2)
    bt['Pmax_MW'] = bt.Annual_Load_MWh * (power_factor[bt.Geo]).values
    

#%% Compute independent feeders, open plot to define open/closed lines, compute tech data (if needed), and save
if not ('Connected' in hta.columns):
    hta['Connected'] = True
# Main node
ps0 = 'Boriette'
n0 = ps.node[ps0]

dep = 19
polys = iris_poly[iris_poly.DEP_CODE == dep][['IRIS_NAME', 'Polygon', 'Lon', 'Lat']]
polys.columns = ['Name', 'Polygon', 'xGPS', 'yGPS']


off = on_off_lines(hta, n0, ss=ps, lv=bt, GPS=True, geo=polys, 
                   tech=tech, nodes=fnodes,
                   outputfolder=folder)



#%% Save data!
# Reduced data w/o non connected
htared = hta[~hta.Feeder.isnull()]
ns = unique_nodes(htared)
fnodesred = fnodes.loc[ns]
btred = bt[bt.node.isin(fnodesred.index)]
psred = ps[ps.node.isin(fnodesred.index)]

util.create_folder(folder + r'ProcessedData')
print('Saving reduced data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(htared), len(btred), len(ns)))
btred.to_csv(folder + r'ProcessedData\\' +  'MVLV.csv')
psred.to_csv(folder + r'ProcessedData\\' +  'SS.csv')
htared.to_csv(folder + r'ProcessedData\\' +  'MVLines.csv')
fnodesred.to_csv(folder + r'ProcessedData\\' +  'Nodes.csv')

print('Saving Full data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(hta), len(bt), len(unique_nodes(hta))))
bt.to_csv(folder + r'ProcessedData\\' +  'MVLV_full.csv')
ps.to_csv(folder + r'ProcessedData\\' +  'SS_full.csv')
hta.to_csv(folder + r'ProcessedData\\' +  'MVLines_full.csv')
fnodes.to_csv(folder + r'ProcessedData\\' +  'Nodes_full.csv')