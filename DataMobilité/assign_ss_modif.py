# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:11:14 2019
Assign IRIS/Commune to SS
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
            dist[z].append(mb.computeDist(latlonz[i], latlonss[j]))
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
        maxload[ss] = zones_load[assignment.index[assignment == ss]].sum() * peakloadfactor
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
    load_rates = compute_load_rate(zones.Load, substations.Pmax, assignment)
    it = 0
    maxr = [load_rates.max()]
    while maxr[-1] > max_load:
        if plotose:
            plot_assignment(zones, substations, assignment, lines=True, title='')
        if it >= max_it:
            break
        if verbose:
            print('Iteration {}: \nMax load rate: Substation {}, {}'.format(it, load_rates.idxmax(), load_rates.max()))
        dist = update_distances(dist, load_rates, max_load, alpha)
        assignment = assigns_min(dist)
        load_rates = compute_load_rate(zones.Load, substations.Pmax, assignment)
        maxr.append(load_rates.max())
        it += 1
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
    return assignment
    
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
        polys.append([polygons[i] for i in assignment[assignment == a].index])
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

def plot_polygons(polys, ax='', **kwargs):
    if ax == '':
        f, ax = plt.subplots()
    collection = PatchCollection(polys, **kwargs)
    ax.add_collection(collection)
    ax.autoscale()
        
#%% Reading commune, IRIS, & trafo data
print('loading data conso per commune')
fniris = 'IRIS.csv'
print('IRIS Conso')
iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\\' + fniris, 
                   engine='python', index_col=0)
print('GeoData')
Tgeo = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/geoRefs.csv', 
                 engine='python', delimiter=';', index_col=0)
print('SS Data')
SS = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source.csv', 
                 engine='python', index_col=0)
SS_polys = pd.read_csv('c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau/postes_source_polys.csv', 
                 engine='python', index_col=0)

#iris_full = pd.read_csv(r'C:\Users\u546416\Downloads\consommation-electrique-par-secteur-dactivite-iris.csv', 
#                   engine='python', index_col=2, delimiter=';')
print('Polygons')
iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\iris_polygons.csv',
                        engine='python', index_col=0)
print('Load Profiles')
# Load conso profiles data (in pu (power, not energy))
load_profiles = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\conso_all_pu.csv', 
                           engine='python', delimiter=',', index_col=0)
#%% Constructing polygons
iris_polygons = {}
j = 0

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
    
# If IRIS-SS assignment is redefined, also should the SS polygons
print('Constructing SS polygons')
SS_polygons = {ss : ptc.Polygon(eval(SS_polys.Polygon[ss]))
                    for ss in SS_polys.index 
                    if len(eval(SS_polys.Polygon[ss])) > 2}



##%%
#polyIris = {}
#typePoly = {}
#j = 0
#print('Constructing IRIS polygons DF')
#for i in iris_full.index:
#    if j%1000 == 0:
#        print('IRIS # {}'.format(j))
#    j += 1
#    try:
#        poly = eval(iris_full.loc[i, 'geom'])
#        polyIris[int(i)] = poly['coordinates'][0]
#        typePoly[int(i)] = poly['type']
#
#    except:
#        pass    
#
#dfpolys = pd.DataFrame([polyIris, typePoly])


#%% Assign IRIS, iterating by department

deps = iris.Departement.unique()
deps.sort()
#depSS = SS.Commune.apply(lambda x: x//1000)
loadcols = ['RES', 'PRO', 'Agriculture', 'Industrie', 'Tertiaire']
iris['Load'] = iris[loadcols].sum(axis=1)/1000

ml = []
assigns = pd.Series()
for d in deps:
    print('\nAssigning SS in département {}'.format(d))
    
    # Subset of SS and IRIS of each department
    irises = iris[iris.Departement == d]
    SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
    # Launch assignment algorithm
    assignment = assign_algorithm(irises, SSs, alpha=0.5, max_it=40, max_load=0.8,
                                  plotose=False, title='Dep '+str(d), verbose=False)
    assigns = pd.concat([assigns, assignment])
#%% Plot departments

deps = iris.Departement.unique()
deps.sort()
f, ax = plt.subplots()
f.set_size_inches(14,12)
for d in deps:
    #un/comment to do only one 
#    d = 11
    print('Dep ', d)
    ax.clear()
    ax.set_aspect('equal')
    #Select subset to plot
    irises = iris[iris.Departement == d]
    SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
    #assignment = assigns[irises.index]
    assignment = irises.SS
    plot_polygon_assignment(irises, SSs, assignment, iris_polygons, ax=ax)
    #Plot SS names
    maxload = compute_load_rate(irises.Load, SSs.Pmax, assignment).max()
    ax.set_title('{} {:.0f}, max SS load {:.2f} p.u.'.format(irises.Dep_name.iloc[0], d, maxload))
    for ss in SSs.index:
        ax.text(SSs.Lon[ss], SSs.Lat[ss], ss, ha='center',
                path_effects=[pe.withStroke(linewidth=3, foreground='w')])
    #un/commment to do only one
    f.savefig(r'c:\user\U546416\Pictures\DataMobilité\Conso\IRIS\Dep{}.png'.format(d))
#    break

    
#%% redo one dep
f, ax = plt.subplots()
f.set_size_inches(14,12)
d = 12
print('\nAssigning SS in département {}'.format(d))
# Subset of SS and IRIS of each department
irises = iris[iris.Departement == d]
SSs = SS[(SS.Departement == d) & (SS.GRD == 'Enedis')]
# Launch assignment algorithm
assignment = assign_algorithm(irises, SSs, alpha=0.4, max_it=40, max_load=0.80,
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
plot_polygon_assignment(irises, SSs, assignment, iris_polygons, ax=ax)
#Plot SS names
maxload = compute_load_rate(irises.Load, SSs.Pmax, assignment).max()
ax.set_title('{} {:.0f}, max SS load {:.2f} p.u.'.format(irises.Dep_name.iloc[0], d, maxload))
for ss in SSs.index:
    ax.text(SSs.Lon[ss], SSs.Lat[ss], ss, ha='center',
            path_effects=[pe.withStroke(linewidth=3, foreground='w')])
#un/commment to do only one
f.savefig(r'c:\user\U546416\Pictures\DataMobilité\Conso\IRIS\Dep{}.png'.format(d))
#    break

#%% Analysis of max load using load curves
print('Computing max load analysis')
SS_load = {}
fs = {}
i = 0
loadcols = ['RES', 'PRO', 'Agriculture', 'Industrie', 'Tertiaire']

for ss in SS[SS.GRD == 'Enedis'].index:
    i +=1
    if i%100 == 0:
        print(i)
    irises = iris[iris.SS == ss]
    
    annual_load = irises[loadcols].sum()
    load = load_profiles[loadcols] * annual_load/8760
    fs[ss] = annual_load
    if SS.Pmax[ss] == 0:
        SS_load[ss] = [SS.Pmax[ss], annual_load.sum()/1000, load.sum(axis=1).max(), 
                9999, load.sum(axis=1).idxmax()]
    else:
        SS_load[ss] = [SS.Pmax[ss], annual_load.sum()/1000, load.sum(axis=1).max(), 
                load.sum(axis=1).max()/SS.Pmax[ss], load.sum(axis=1).idxmax()]

SS_load = pd.DataFrame(SS_load, index=['Pmax_SS[MW]', 'AnnualLoad[GWh]', 'MaxLoad[MW]','SSCharge[pu]', 'idMaxLoad']).transpose()
fs = pd.DataFrame(fs).transpose()
SS_load = pd.concat([SS_load, fs], axis=1)
#SS_loadred = SS_load.loc[(SS_load.SSCharge > 0) & (SS_load.SSCharge < 9998)]
print('Finished computing')

SS_load.to_excel('SSload.xlsx')

#%
f, ax = plt.subplots()
hi, bins = np.histogram(SS_load['SSCharge[pu]'], bins=[i/20 for i in range(21)])
ax.bar(bins[0:-1] + 0.025, hi, width=1/20)
ax.set_xlim(0,1)
ax.set_ylabel('Number of SS')
ax.set_xlabel('Max load [pu of SS]')
ax.grid()
ax.set_title('Distribution of max load of Substation')


#%% Load histograms of distances
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
                    engine='python', index_col=0)


#%% Plot type Iris
polyA = [iris_polygons[i] for i in iris[iris.Type_IRIS == 'A'].index]
polyH = [iris_polygons[i] for i in iris[iris.Type_IRIS == 'H'].index]
polyD = [iris_polygons[i] for i in iris[iris.Type_IRIS == 'D'].index]
polyZ = [iris_polygons[i] for i in iris[iris.Type_IRIS == 'Z'].index]
collectionA = PatchCollection(polyA, facecolors='r', label='Activité')
collectionH = PatchCollection(polyH, facecolors='b', label='Habitation')
collectionD = PatchCollection(polyD, facecolors='silver', label='Divers')
collectionZ = PatchCollection(polyZ, facecolors='g', label='Rural')

f, ax = plt.subplots()
ax.add_collection(collectionA)
ax.add_collection(collectionZ)
ax.add_collection(collectionD)
ax.add_collection(collectionH)
ax.autoscale()
plt.legend()
#%% Compute histograms per SS
print('Computing histograms per SS')
SS_hhome = {}
SS_hwork = {}
i = 0
hab_com = iris[['Code_commune','Habitants']].groupby('Code_commune').sum().squeeze()

hh = hhome.drop(['ZE','Status', 'UU', 'Dep'], axis=1)

for ss in SS[SS.GRD == 'Enedis'].index:
    i +=1
    if i%100 == 0:
        print(i)
    irises = iris[iris.SS == ss]
    coms = irises[['Habitants', 'Code_commune']].groupby('Code_commune').sum()
    if len(coms)>1:
        coms = coms.squeeze()
    else:
        coms = pd.Series(data=coms.Habitants, index=coms.index)
    ss_hh = (hh.loc[coms.index,:].transpose() * (coms/hab_com[coms.index])).transpose().sum(axis=0)
    SS_hhome[ss] = ss_hh


SS_hhome = pd.DataFrame(SS_hhome)
SS_hhome.transpose().to_csv(folder_hdata + r'\HistHomeModal_SS.csv')

#%% compute and plot Avg daily distance and KmVehicle per SS
bins = pd.Series(data=[i*2+1 for i in range(50)], index=SS_hhome.index)
means = (SS_hhome.transpose() * bins).transpose().sum(axis=0) / SS_hhome.sum(axis=0)
kmvoit = (SS_hhome.transpose() * bins).transpose().sum(axis=0)
# Plot Average distance

# names of communes in iris DataFrame
cnames = ['Paris 1', 'Marseille 1', 'Lyon 1', 'Toulouse', 'Bordeaux', 'Nantes', 'Lille', 'Rennes']
# Names to print in map
cns = ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Bordeaux', 'Nantes', 'Lille', 'Rennes']
latlons = [[iris[iris.Nom_commune == c].Lon.iloc[0], iris[iris.Nom_commune == c].Lat.iloc[0]] for c in cnames]

f, ax = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
for ss in means.index:
    irises = iris[iris.SS == ss].index
    polys = [iris_polygons[i] for i in irises]
    if np.isnan(means[ss]):
        print(ss)
    else:
        plot_polygons(polys, ax, facecolors=palette[int(means[ss] *2 // 15)]) #range every 15km of mean daily distance
ax.set_title('Average daily distance per Substation [km]')
for i in range(len(cns)):
    ax.text(latlons[i][0],latlons[i][1]+0.2, cns[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])
#% Plot km*voiture by batch of 150.000 km*Veh/day (+- 30 MWh/day for 50% = EVs)
f, ax = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
for ss in means.index:
    irises = iris[iris.SS == ss].index
    polys = [iris_polygons[i] for i in irises]
    if np.isnan(means[ss]):
        print(ss)
    else:
        plot_polygons(polys, ax, facecolors=palette[int(kmvoit[ss] // 150000)]) #range every 15km of mean daily distance
ax.set_title('km' + r'$\bullet$' + 'vehicle per substation')
for i in range(len(cns)):
    ax.text(latlons[i][0],latlons[i][1]+0.2, cns[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])
#%% plot using SS
#kmVehicle
f, ax = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
kmvoit_base = 150000
polyss = PatchCollection([SS_polygons[ss] for ss in SS_polygons], 
                         facecolors=[palette[int(kmvoit[ss] // kmvoit_base)]
                                     for ss in SS_polygons], 
                         edgecolors='k', linestyle='--', linewidth=0.2)
ax.add_collection(polyss)
ax.autoscale()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_title('km' + r'$\bullet$' + 'Vehicle per Substation')
for i in range(len(palette)):
    ax.plot(1,1,'s', color=palette[i], 
            label=str(int(i*kmvoit_base/1000))+'k<kmV<'+str(int((i+1)*kmvoit_base/1000))+'k')
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.legend(loc=3)

for i in range(len(cns)):
    ax.text(latlons[i][0],latlons[i][1]+0.2, cns[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])
    
#%% Avg daily distance
f, ax = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
km_base = 7.5
polyss = PatchCollection([SS_polygons[ss] for ss in SS_polygons
                          if np.isnan(means[ss])==False], 
                         facecolors=[palette[int(means[ss] // km_base)]
                                     for ss in SS_polygons
                                     if np.isnan(means[ss])==False], 
                         edgecolors='k', linestyle='--', linewidth=0.2)
ax.add_collection(polyss)
ax.autoscale()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_title('Average daily distance per Substation [km]')
for i in range(len(palette)):
    ax.plot(1,1,'s', color=palette[i], 
            label=str(int(i*km_base*2))+'<d<'+str(int((i+1)*km_base*2)))
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.legend(loc=3)

for i in range(len(cns)):
    ax.text(latlons[i][0],latlons[i][1]+0.2, cns[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])

#%% Plot for IdF
deps_idf = [75, 77, 78, 91, 92, 93,  94, 95]
ssidf = SS[(SS.GRD == 'Enedis') & (SS.Departement.isin(deps_idf))].index

# names of communes in iris DataFrame
cnames_idf = ['Paris 1', 'Versailles', 'Évry', 'Meaux', 'Bordeaux', 'Nemours', 'Cergy', 'Étampes', 'Provins']
# Names to print in map
cns_idf = ['Paris', 'Versailles', 'Évry', 'Meaux', 'Bordeaux', 'Nemours', 'Cergy', 'Étampes', 'Provins']
latlons_idf = [[iris[iris.Nom_commune == c].Lon.iloc[0], iris[iris.Nom_commune == c].Lat.iloc[0]] for c in cnames_idf]

f, ax = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
km_base = 7.5
polyss = PatchCollection([SS_polygons[ss] for ss in ssidf
                          if np.isnan(means[ss])==False], 
                         facecolors=[palette[int(means[ss] // km_base)]
                                     for ss in ssidf
                                     if np.isnan(means[ss])==False], 
                         edgecolors='k', linestyle='--', linewidth=0.2)
ax.add_collection(polyss)
ax.autoscale()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_title('Average daily distance per Substation [km]')
for i in range(len(palette)):
    ax.plot(1,1,'s', color=palette[i], 
            label=str(int(i*km_base*2))+'<d<'+str(int((i+1)*km_base*2)))
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.legend(loc=3)

for i in range(len(cns_idf)):
    ax.text(latlons_idf[i][0],latlons_idf[i][1]+0.02, cns_idf[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])
ax.set_aspect('equal')

# kmVoitures
f, ax = plt.subplots()
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
km_base = 7.5
polyss = PatchCollection([SS_polygons[ss] for ss in ssidf
                          if np.isnan(means[ss])==False], 
                         facecolors=[palette[int(kmvoit[ss] // kmvoit_base)]
                                     for ss in ssidf
                                     if np.isnan(means[ss])==False], 
                         edgecolors='k', linestyle='--', linewidth=0.2)
ax.add_collection(polyss)
ax.autoscale()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_title('km' + r'$\bullet$' + 'Vehicle per Substation')
for i in range(len(palette)):
    ax.plot(1,1,'s', color=palette[i], 
            label=str(int(i*kmvoit_base/1000))+'k<kmV<'+str(int((i+1)*kmvoit_base/1000))+'k')
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.legend(loc=3)
ax.set_aspect('equal')
for i in range(len(cns_idf)):
    ax.text(latlons_idf[i][0],latlons_idf[i][1]+0.02, cns_idf[i], ha='center',
       path_effects=[pe.withStroke(linewidth=2, foreground='w')])
    