"""
Computes load profiles for substations
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


#%% Reading commune, IRIS, & trafo data
print('loading data conso per commune')
print('IRIS Conso')
fniris = 'IRIS_enedis_2017.csv'
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\\' + fniris, 
                   engine='python', index_col=0)
#print('GeoData')
#Tgeo = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/geoRefs.csv', 
#                 engine='python', delimiter=';', index_col=0)
print('SS Data')
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', index_col=0)
    
SS_polys = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source_polygons_dec2019.csv', 
                 engine='python', index_col=0)

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\IRIS_all_geo_2016.csv',
                        engine='python', index_col=0)
print('Load Profiles')
# Load conso profiles data (in pu (power, not energy))
load_profiles = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\conso_all_pu.csv', 
                           engine='python', delimiter=',', index_col=0)
#%% Constructing polygons
print('Constructing polygons')
print('IRIS polygons')
iris_poly.Polygon = iris_poly.Polygon.apply(lambda x: eval(x))
polygons = util.do_polygons(iris_poly, plot=True)
#test
util.plot_polygons([pp for p in polygons.values() for pp in p])
print('SS polygons')
SS_polys.Polygon = SS_polys.Polygon.apply(lambda x: [eval(x)])
polygons_ss = util.do_polygons(SS_polys, plot=True)
#test
print('Finished')

#%% Analysis of max load using load curves
print('Computing max load analysis')
SS_profile = {}
SS_load = {}
i=0
for ss in SS[SS.GRD == 'Enedis'].index:
    i +=1
    if i%100 == 0:
        print(i)
    profile = util.compute_load_from_ss(iris, load_profiles, ss)
    SS_profile[ss] = profile
    SS_load[ss] = [SS.Pmax[ss], profile.sum()/2/1000, profile.max(), profile.max()/SS.Pmax[ss], profile.idxmax()]
    

    
SS_load = pd.DataFrame(SS_load, index=['Pmax_SS[MW]', 'AnnualLoad[GWh]', 'MaxLoad[MW]','SSCharge[pu]', 'idMaxLoad']).transpose()
SS_profile = pd.DataFrame(SS_profile)

#SS_loadred = SS_load.loc[(SS_load.SSCharge > 0) & (SS_load.SSCharge < 9998)]
print('Finished computing')

#SS_load.to_excel('SSload.xlsx')

#% Plot histogram of Pmax[pu]
f, ax = plt.subplots()
hi, bins = np.histogram(SS_load['SSCharge[pu]'], bins=[i/20 for i in range(21)])
ax.bar(bins[0:-1] + 0.025, hi, width=1/20)
ax.set_xlim(0,1)
ax.set_ylabel('Number of SS')
ax.set_xlabel('Max load [pu of SS]')
ax.grid()
ax.set_title('Distribution of max load of Substation')

#% Plot map of Pmax[pu]
palette = ['lightgreen','y','coral','r','maroon']
ranges = [0,0.5,0.6,0.7,0.8,0.9,1]
f, ax = plt.subplots()
polyss = PatchCollection([polygons_ss[ss][0] for ss in polygons_ss
                          if len(polygons_ss[ss])>0], 
                         facecolors=[palette[max(0,int((SS_load['SSCharge[pu]'][ss]-0.5) // 0.1))]
                                     for ss in polygons_ss
                                     if len(polygons_ss[ss])>0], 
                         edgecolors='k', linestyle='--', linewidth=0.15)
ax.add_collection(polyss)
util.aspect_carte_france(ax, palette=palette, ranges=ranges, label_middle='%<x<', label_end='%')

#% Pmax[pu] pour IdF
deps_idf = [75, 77, 78, 91, 92, 93,  94, 95]
ssidf = SS[(SS.GRD == 'Enedis') & (SS.Departement.isin(deps_idf))].index
f, ax = plt.subplots()
polyss = PatchCollection([polygons_ss[ss][0] for ss in ssidf
                          if len(polygons_ss[ss])>0], 
                         facecolors=[palette[max(0,int((SS_load['SSCharge[pu]'][ss]-0.5) // 0.1))]
                                     for ss in ssidf
                                     if len(polygons_ss[ss])>0], 
                         edgecolors='k', linestyle='--', linewidth=0.15)
ax.add_collection(polyss)
util.aspect_carte_france(ax, palette=palette, ranges=ranges, label_middle='%<x<', label_end='%', cns='idf')
#%% Save
SS_profile.to_csv('SS_profiles.csv')
SS_load.to_excel('SS_load.xlsx')
