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
    
polygons_ss = util.load_polygons_SS()

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
polygons = util.load_polygons_iris()

print('Load Profiles')
# Load conso profiles data (in pu (power, not energy))
load_profiles = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\conso_all_pu.csv', 
                           engine='python', delimiter=',', index_col=0)


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

#%% Plot map of Pmax[pu]
palette = ['lightgreen','y','coral','r','maroon']
ranges = [0,0.5,0.6,0.7,0.8,0.9,1]

polyss = util.list_polygons(polygons_ss, polygons_ss.keys())
colorss = [palette[max(0,int((SS_load['SSCharge[pu]'][ss]-0.5) // 0.1))]
            for ss in polygons_ss
            for p in polygons_ss[ss]]
    
ssidf = SS[(SS.GRD == 'Enedis') & (SS.Departement.isin(util.deps_idf))].index
polys_idf = util.list_polygons(polygons_ss, ssidf)
colors_idf = [palette[max(0,int((SS_load['SSCharge[pu]'][ss]-0.5) // 0.1))]
            for ss in ssidf
            for p in polygons_ss[ss]]
#%%
labels = [str(ranges[i]) + '<PeakLoad<' + str(ranges[i+1]) for i in range(len(ranges)-1)]
ax1 = util.plot_polygons(polyss, facecolors=colorss, edgecolors='k', linewidth=0.2)
ax2 = util.plot_polygons(polys_idf, facecolors=colors_idf, edgecolors='k', linewidth=0.2)
util.aspect_carte_france(ax1, palette=palette, labels=labels)
util.aspect_carte_france(ax2, palette=palette, labels=labels, cns='idf')
#%% Save
#SS_profile.to_csv('SS_profiles.csv') #This is (way) too heavy (600MB, over 45s to load)
#SS_load.to_excel('SS_load.xlsx')

#%% Saving profiles one by one in a dedicated folder
folder = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\SS_profiles\\'
for ss in SS_profile:
    SS_profile[[ss]].to_csv(folder + ss + '.csv')