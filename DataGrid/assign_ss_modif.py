# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:11:14 2019
Assign IRIS/Commune to SS
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

def compute_distances(zones, substations):
    """Computes distances for two set of zones.
    This can refer to a geo area (IRIS, Communes) and substations.
    both data inputs are pandas dataframes with 'Lat', 'Lon' columns
    Returns a dataframe of distances, w columns zones, rows substations
    """
    dist = {}
    latlonz = np.array(zones.loc[:, ['Lat', 'Lon']])
    latlonss = np.array(substations.loc[:, ['Lat', 'Lon']])
    for i in range(len(zones.index)):
        z = zones.index[i]
        dist[z] = []
        for j in range(len(substations.index)):
            dist[z].append(util.computeDist(latlonz[i], latlonss[j]))
    return pd.DataFrame(dist, index=substations.index)


def assigns_min(dist):
    """ Assigns SS to zone based on min distance matrix
    data is a dataframe with zones in columns and substations in index
    returns a pd.Series of index zones, and value Substation
    """
    return dist.idxmin()

def compute_load_rate(zones_load, substations_pmax, assignment):
    """ 
    assignment is a pd.Series with index zones, and value substation
    zones_load is a pd.Series with index zones, and value annual load
    substation_pmax is a pd.Series with index substation and value pmax
    Returns a pd.Series of index Substation and value max_load / rated_load 
    """
    peakloadfactor = 0.207  #factor translating conso in annual GWh to MW peak
    maxload = {}
    for ss in substations_pmax.index:
        maxload[ss] = zones_load[assignment[assignment == ss].index].sum() * peakloadfactor
    return pd.Series(maxload)/substations_pmax

def update_distances(distances, load_rate, max_load=1, alpha=0.5):
    """
    distances is a pd.DataFrame of columns zones, rows substations
    load_rate is a pd.Series w index substations, 
    """
    factor = ((load_rate/max_load) ** alpha).clip_lower(1)
    return (distances.transpose() * factor).transpose()

def assign_algorithm(zones, substations, verbose=True, plotose=False, 
                     alpha=0.2, max_it=20, max_load=0.9, maxrplot=True, title=''):
    """
    zones is a pd.DataFrame with columns Lat, Long, load (annual GWh)
    substations is a pd.DataFrame with columns Lat, Long, Pmax (in MW)
    """
    if verbose:
        print('Starting assignment of {} zones to {} substations'.format(zones.shape[0], substations.shape[0]))
        print('Computing distances')
    dist = compute_distances(zones, substations)
    assignment = assigns_min(dist)
    if verbose:
        print('Distances computed')
        print('Computing substation load rates')
    load_rates = compute_load_rate(zones.Load_GWh, substations.Pmax, assignment)
    it = 0
    maxr = [load_rates.max()]
    assignmentmin = assignment
    while maxr[-1] > max_load:
        if plotose:
            plot_assignment(zones, substations, assignment, lines=True, title='')
        if it >= max_it:
            break
        if verbose:
            print('Iteration {}: \nMax load rate: Substation {}, {}'.format(it, load_rates.idxmax(), load_rates.max()))
        dist = update_distances(dist, load_rates, max_load, alpha)
        assignment = assigns_min(dist)
        load_rates = compute_load_rate(zones.Load_GWh, substations.Pmax, assignment)
        maxr.append(load_rates.max())
        it += 1
        if maxr[-1] == min(maxr):
            assignmentmin = assignment
    if maxrplot & (it >=5):
        f, ax = plt.subplots()
        ax.plot([i for i in range(len(maxr))], maxr)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max load')
        ax.set_title(title)
    if plotose:
        plot_assignment(zones, substations, assignment, lines=True)
    if verbose:
        print('Finished assignment, max load rate={}'.format(maxr[-1]))
    return assignmentmin
    
def plot_assignment(zones, substations, assignment, ax='', lines=False, ms=30, title=''):
    """
    """
    if ax=='':
        f, ax= plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']
    i = 0
    if lines:
        for ss in substations.index:
            latss, lonss = substations.Lat[ss], substations.Lon[ss]
            lines_c = [[(lonz, latz), (lonss, latss)]
                       for lonz, latz in 
                       zip(zones.Lon[assignment.index[assignment == ss]],
                           zones.Lat[assignment.index[assignment == ss]])]
            lc = LineCollection(lines_c, linestyle='-', color=colors[i%len(colors)], label='_')
            ax.add_collection(lc)
            i+=1
    i = 0
    for ss in substations.index:
        ax.scatter(zones.Lon[assignment.index[assignment == ss]], zones.Lat[assignment.index[assignment == ss]], 
                   s=ms, label='_', c=colors[i%len(colors)])
        ax.scatter(substations.Lon[ss], substations.Lat[ss], 
                   marker='*', s=ms*2, label=ss, c=colors[i%len(colors)])
        i += 1
    ax.autoscale()
    ax.set_title(title)
    plt.legend()
    return ax

def plot_polygon_assignment(zones, substations, assignment, polygons, ax='', lines=True):
    # define colors
    palette = ['b', 'g', 'r', 'm', 'c', 'y', 'orange', 'springgreen', 'crimson', 'peachpuff',   #10
           'silver', 'orchid', 'tan', 'lawngreen', 'darkslategray',                              #15  
           'indigo', 'mediumslateblue', 'olive', 'bisque', 'lightcoral']                    #20
    if ax=='':
        f, ax= plt.subplots()
    ass = assignment.unique()
    polys = []
    j=0
    for j in range(len(ass)):
        a = ass[j]
        # Add polygons collections
        polys.append([p for i in assignment[assignment == a].index for p in polygons[i]])
        collection = PatchCollection(polys[j], facecolors=palette[j%len(palette)], label=a)
        ax.add_collection(collection)
        j += 1
        ax.plot(substations.Lon[a], substations.Lat[a], 'v', color='k')
        # Add line collections
        if lines:
            latss, lonss = substations.Lat[a], substations.Lon[a]
            lines_c = [[(lonz, latz), (lonss, latss)]
                       for lonz, latz in 
                       zip(zones.Lon[assignment.index[assignment == a]],
                           zones.Lat[assignment.index[assignment == a]])]
            lc = LineCollection(lines_c, linestyle='--', color='k', label='_')
            ax.add_collection(lc)        
    ax.autoscale()
    plt.legend()
    return ax

#def plot_polygons(polys, ax='', **kwargs):
#    if ax == '':
#        f, ax = plt.subplots()
#    collection = PatchCollection(polys, **kwargs)
#    ax.add_collection(collection)
#    ax.autoscale()
        
#%% Reading commune, IRIS, & trafo data
print('loading data conso per commune')
fniris = 'IRIS_enedis_2017.csv'
print('IRIS Conso')
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\\' + fniris, 
                   engine='python', index_col=0)
#print('GeoData')
#Tgeo = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/geoRefs.csv', 
#                 engine='python', delimiter=';', index_col=0)
print('SS Data')
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', index_col=0)

#SS_polys = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source_polygons.csv', 
#                 engine='python', index_col=0)

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\GeoData\IRIS_all_geo_2016.csv',
                        engine='python', index_col=0)
polygons = util.load_polygons_iris()
print('Load Profiles')
# Load conso profiles data (in pu (power, not energy))
load_profiles = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\conso_all_pu.csv', 
                           engine='python', delimiter=',', index_col=0)

#%% Assign IRIS, iterating by department

consos = util.consos

deps = iris.Departement.unique()
deps.sort()
#depSS = SS.Commune.apply(lambda x: x//1000)

ml = []
assigns = pd.Series()
iris.Departement = iris.Departement.astype(int)
for d in deps:
    print('\nAssigning SS in département {}'.format(d))
    
    # Subset of SS and IRIS of each department
    irises = iris[iris.Departement == d]
    SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
    # Launch assignment algorithm
    assignment = assign_algorithm(irises, SSs, alpha=0.5, max_it=40, max_load=0.8,
                                  maxrplot=False, plotose=False, title='Dep '+str(d), verbose=False)
    assigns = pd.concat([assigns, assignment])
    

    #%%
iris['SS'] = assigns
#%% Plot departments

deps = iris.Departement.unique()
deps.sort()
f, ax = plt.subplots()
f.set_size_inches(14,12)
maxl = pd.Series(np.zeros(len(deps)), index=deps)
for d in deps:
    #un/comment to do only one 
#    d = 60
    print('Dep ', d)
    ax.clear()
    ax.set_aspect('equal')
    #plot all irises:
    iriDep = iris_poly[(iris_poly.COMM_CODE>d*1000)&(iris_poly.COMM_CODE<(d+1)*1000)]
    util.plot_polygons([p for i in iriDep.index for p in polygons[i]], 
                  ax=ax, facecolor='white', edgecolor='k', linestyle='--')
    #Select subset to plot
    irises = iris[iris.Departement == d]
    SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
    #assignment = assigns[irises.index]
    assignment = irises.SS
    plot_polygon_assignment(irises, SSs, assignment, polygons, ax=ax)
    #Plot SS names
    maxload = compute_load_rate(irises.Load_GWh, SSs.Pmax, assignment).max()
    maxl[d] = maxload
    ax.set_title('{} {:.0f}, max SS load {:.2f} p.u.'.format(irises.DEP_NAME.iloc[0], d, maxload))
    for ss in SSs.index:
        ax.text(SSs.Lon[ss], SSs.Lat[ss], ss, ha='center',
                path_effects=[pe.withStroke(linewidth=3, foreground='w')])
    #un/commment to do only one
    f.savefig(r'c:\user\U546416\Pictures\SS\IRIS\Dep{}.png'.format(d))
#    break

    
#%% redo one dep
f, ax = plt.subplots()
f.set_size_inches(14,12)
d = 38
print('\nAssigning SS in département {}'.format(d))
# Subset of SS and IRIS of each department
irises = iris[iris.Departement == d]
SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
# Launch assignment algorithm
assignment = assign_algorithm(irises, SSs, alpha=0.3, max_it=40, max_load=0.80,
                              plotose=False, title='Dep '+str(d), verbose=True)

iris.loc[assignment.index, 'SS'] = assignment
# Plot
ax.clear()
ax.set_aspect('equal')
#Select subset to plot
irises = iris[iris.Departement == d]
SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
#assignment = assigns[irises.index]
assignment = irises.SS
plot_polygon_assignment(irises, SSs, assignment, polygons, ax=ax)
#Plot SS names
loading = compute_load_rate(irises.Load_GWh, SSs.Pmax, assignment)
print(loading)
maxload = loading.max()
ax.set_title('{} {:.0f}, max SS load {:.2f} p.u.'.format(irises.DEP_NAME.iloc[0], d, maxload))
for ss in SSs.index:
    ax.text(SSs.Lon[ss], SSs.Lat[ss], ss, ha='center',
            path_effects=[pe.withStroke(linewidth=3, foreground='w')])
#un/commment to do only one
f.savefig(r'c:\user\U546416\Pictures\SS\IRIS\Dep{}.png'.format(d))
#    break



#%%save iris

iris.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\IRIS_enedis_2017.csv')