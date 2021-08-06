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
import graph_reconstruction as gr

#%% Load IRIS data
print('Loading IRIS polygons')
folder_iris = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
iris_poly = pd.read_csv(folder_iris+file_iris,
                        engine='python', index_col=0)
print('\tDone loading polygons')
#%% Load conso data
print('Loading Conso per IRIS')
folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
print('Loading profiles in p.u.')
profiles = pd.read_csv(folder_consodata + r'\conso_all_pu.csv',
                       engine='python', index_col=0)
profiles.drop(['ENT', 'NonAffecte'], axis=1, inplace=True)

#%% Load tech data
print('Loading tech data')
folder_tech = r'c:\user\U546416\Documents\PhD\Data\MVGrids\\'
file_tech = 'line_data_France_MV_grids.xlsx'
tech = pd.read_excel(folder_tech + file_tech, index_col=0)

#%% Loading number of LV transfos per IRIS
# load # BT/iris:
print('Loading LV transformer data')
folder_lv = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\\'
lv_iris = pd.read_csv(folder_lv+'Nb_BT_IRIS2016.csv',
                      engine='python', index_col=0)

#%% Either load pre-processed grid or create one from Enedis raw data

print('Loading MV grid data')
folder = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\\'
folder_output = r'c:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\\'

v = util.input_y_n('Do you want to create graph from GIS data (Y) or load processed grid (N):')
    
if v in ['Y', 'y', True]:
    hta, bt, ps, nodes = gr.run_graph(folder, folder_output=folder_output)
else:
    hta = pd.read_csv(folder + 'MVLines_full.csv', engine='python', index_col=0)
    ps = pd.read_csv(folder + 'SS_full.csv', engine='python', index_col=0)
    bt = pd.read_csv(folder + 'MVLV_full.csv', engine='python', index_col=0)
    nodes = pd.read_csv(folder + 'Nodes_full.csv', engine='python', index_col=0)
    
    hta.ShapeGPS = hta.ShapeGPS.apply(eval)


#%% Assigning Geographic zone to each node and LV load

# Reducing polygons to consider to +- 0.5 degrees of latitude/longitude to data
dt = 0.5
lonmin, lonmax, latmin, latmax = nodes.xGPS.min(), nodes.xGPS.max(), nodes.yGPS.min(), nodes.yGPS.max()
polys = iris_poly[(iris_poly.Lon > lonmin-dt) &
                  (iris_poly.Lon < lonmax+dt) &
                  (iris_poly.Lat > latmin-dt) &
                  (iris_poly.Lat < latmax+dt)][['IRIS_NAME', 'Polygon', 'Lon', 'Lat']]
polys.columns = ['Name', 'Polygon', 'xGPS', 'yGPS']
polys.Polygon = pd.Series(util.do_polygons(polys, plot=False))


if not 'Geo_Name' in nodes.columns:
    assign_polys(nodes, polys)
    if nodes.Geo.isnull().sum():
        print('Warning: there are some nodes without assigned polygon, check input data')
if not 'Geo_Name' in bt.columns:
    assign_polys(bt, polys)
    if bt.Geo.isnull().sum():
        print('Warning: there are some MV-LV transformers without assigned polygon, check input data')
if not 'Annual_Load_MWh' in bt.columns:
    bt['Annual_Load_MWh'] = equal_total_load(bt,iris,lv_per_geo=lv_iris.Nb_BT)
if not 'Pmax_MW' in bt.columns:
    # creating profile for each IRIS:
    profs = get_profiles_per_geo(bt, profiles, iris, MW=False)
    # profs[iris] is a yearly profile with max=1
    # peak_factor is the peak_load [MW] / annual_load [MW] (/2 bcs i have 30min values)
    power_factor = 1/(profs.sum()/2)
    bt['Pmax_MW'] = bt.Annual_Load_MWh * (power_factor[bt.Geo]).values


    #%% Compute independent feeders, open plot to define open/closed lines, compute tech data (if needed), and save
if not ('Connected' in hta.columns):
    hta['Connected'] = True

# Main node
ps0 = 'SS3'
print('Check main substation: current choice {}!'.format(ps0))
n0 = ps.node[ps0]

off = on_off_lines(hta, n0, ss=ps, lv=bt, GPS=True, geo=polys, 
                   tech=tech, nodes=nodes,     
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

try:
    profsred = profs[btred.Geo.unique()]
    profs.to_csv(folder + r'ProcessedData\\' +  'profiles_per_iris_full.csv')
    profsred.to_csv(folder + r'ProcessedData\\' +  'profiles_per_iris.csv')
    print('Saving IRIS profiles:\n\IRIS: {}\n\t'.format(profs.shape[1]))
except:
    pass
