# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:15:22 2021

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util

dir_results = r'c:\user\U546416\Documents\PhD\Data\Simulations\Thesis'
print('Loading SS')
folder_ssdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau'
folder_profiles = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\SS_profiles\\'
SS = pd.read_csv(folder_ssdata + r'\postes_source.csv',
                                  engine='python', index_col=0)

peakweek = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\ss_peakweek_2018.csv',
                       engine='python', index_col=0)

#%% Analyze results - computing peak load increase

outputfolders  = ['full_dumb_EV05_W03','full_dumb_ToU_EV05_W03']
data = []
for outputfolder in outputfolders:
    #reading results 
    ev_load_day = pd.read_csv(dir_results + r'\\' + outputfolder + r'\ev_load_day.csv',
                              engine='python', index_col=0)
    ev_load_night = pd.read_csv(dir_results + r'\\' + outputfolder + r'\ev_load_night.csv',
                              engine='python', index_col=0)
    SSs = ev_load_day.columns
    
    # SSs to drop
    dropss = []
    dropss += list(SS[SS.Departement.isin([79,86])].index) + ['MALESHERBES', 'MALAGUAY', 'LUISANT']
    #loads = pd.concat([pd.read_csv(folder_profiles + ss + '.csv', engine='python', index_col=0) for ss in SSs], axis=1)
    #global_data = pd.read_csv(dir_results  + r'\\' + outputfolder + r'\global_data.csv',
    #                          engine='python', index_col=0)
    # extra demand:
    #totdem = loads.sum()/2 #bcs half-hourly
    ev_load = (ev_load_day + ev_load_night)
    ratio_daynight = ev_load_day.sum() / ev_load.sum()
    extraload = (ev_load.mean() * 8760)/SS.AnnualDemand_MWh.loc[SSs] # bcs 15min res
        
    
    
    peakweek.index = ev_load.index
    totaldem = ev_load + peakweek
    increasepeak = totaldem.max() / peakweek.max() -1
    
    data = pd.concat([increasepeak, extraload, ratio_daynight], axis=1)
    data.drop(dropss, errors='ignore', inplace=True)
    data.columns = ['IncreasePeak_pu', 'IncreaseDemand_pu', 'ShareEVDayLoad']
    data.to_csv(dir_results + r'\\' + outputfolder + r'\ss_data.csv')

#peakweek.to_csv(dir_results+r'\\'+outputfolder + r'\ss_peakweek.csv')

#%% Doing plots for increased energy & increased peak load


outputfolders  = ['full_dumb_EV05_W03','full_dumb_ToU_EV05_W03']
data = []
for outputfolder in outputfolders:
    data.append(pd.read_csv(dir_results + r'\\' + outputfolder + r'\ss_data.csv',
                              engine='python', index_col=0))

labels = ['Uncontrolled', 'Off-peak']
    
f, ax = plt.subplots()
f.set_size_inches(3.5,3)
for i, outputfolder in enumerate(outputfolders):
    plt.hist(data[i].IncreasePeak_pu*100, bins=np.arange(0,50,2.5), label=labels[i], alpha=0.5)
plt.xlabel('Peak load increase [%]')
plt.ylabel('Count')
plt.legend()
plt.title('(a) Demand increase', y=-0.26)
plt.tight_layout()


plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\PeakloadIncrease_v2.pdf')
plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\PeakloadIncrease_v2.jpg', dpi=300)


f, ax = plt.subplots()
f.set_size_inches(3.5,3)
plt.hist(data[i].IncreaseDemand_pu*100, bins=np.arange(0,40,2), alpha=0.5)
plt.xlabel('Demand increase [%]')
plt.ylabel('Count')
#plt.legend()
plt.tight_layout()
plt.xlim(0,30)

plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\DemandIncrease.pdf')
plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\DemandIncrease.jpg', dpi=300)

#%% Scatter of demand vs peak load increase
cmap = plt.get_cmap('viridis')
colors = cmap(data[i].ShareEVDayLoad)

f, ax = plt.subplots()
f.set_size_inches(3.5,3)
for i, outputfolder in enumerate(outputfolders):
    plt.scatter(data[0].IncreaseDemand_pu*100, data[i].IncreasePeak_pu*100, s=1, label=labels[i], alpha=0.7)
plt.ylabel('Peak load increase [%]')
plt.xlabel('Demand increase [%]')
plt.legend()
plt.xlim(0,30)
plt.ylim(0,35)
plt.title('(b) Peak load')

plt.tight_layout()

plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\scatter_demvspeak_v2.pdf')
plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\scatter_demvspeak_v2.jpg', dpi=300)


#cmap = plt.get_cmap('viridis')
#colors = cmap(data[i].ShareEVDayLoad)
#
#f, ax = plt.subplots()
#f.set_size_inches(3.5,3)
#plt.scatter(data[0].IncreaseDemand_pu, data[0].IncreasePeak_pu-data[1].IncreasePeak_pu, s=1, c=colors)
#plt.ylabel('Peak load reduction')
#plt.xlabel('Demand increase')
#plt.tight_layout()
##plt.legend()


#%% Loading IRIS polygons
# loading iris polygons
print('Loading IRIS polygons')
folder_polys = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file_iris = 'IRIS_all_geo_'+str(2016)+'.csv'
iris_poly = pd.read_csv(folder_polys+file_iris,
                        engine='python', index_col=0)
dep_polys = pd.read_csv(folder_polys + 'departements_polygons.csv', engine='python', index_col=0)
polys_dep = util.do_polygons(dep_polys)
comms = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\geoRefs.csv',
                        engine='python', index_col=0, sep=';')

folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)

ies = iris[iris.SS.isin(data[0].index)].index
idfdeps = [75,78,77,91,92,93,94,95]
iesidf = iris[(iris.SS.isin(data[0].index)) & (iris.Departement.isin(idfdeps))].index
polygons = util.do_polygons(iris_poly.loc[ies], plot=False)

#%% plot increase in peak load brute force-France
# list of polygons

cases = [0,1]
name = {0:'unc',
        1:'tou'}

polys = util.list_polygons(polygons, ies)
cmap = plt.get_cmap('YlOrRd')
pmax = 0.3
xpalette = np.arange(0,pmax,0.05)
palette = cmap(xpalette/pmax)
labels = ['{}%'.format(int(i*100)) for i in xpalette]
        
for c in cases:
    colors = cmap([data[c].IncreasePeak_pu[iris.SS[i]] / pmax for i in ies for p in polygons[i]])
    
    f, ax = plt.subplots()
    util.plot_polygons(polys, color=colors, ax=ax)
    util.aspect_carte_france(ax, palette=palette, labels=labels)
    plt.xticks([])
    plt.yticks([])
    f.set_size_inches(4.67,  4.06)
    plt.tight_layout()

    plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SSFR_incpeak_{}.pdf'.format(name[c]))
    plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SSFR_incpeak_{}.jpg'.format(name[c]), dpi=300)

#%% plot increase in peak load brute force - IdF
polys = util.list_polygons(polygons, iesidf)
cmap = plt.get_cmap('YlOrRd')

for c in cases:
    colors = cmap([data[c].IncreasePeak_pu[iris.SS[i]] / 0.3 for i in iesidf for p in polygons[i]])
    
    f, ax = plt.subplots()
    util.plot_polygons(polys, color=colors, ax=ax)
    util.plot_polygons(util.list_polygons(polys_dep, idfdeps), facecolor='None', edgecolor='grey', linestyle='--', ax=ax)
    util.aspect_carte_france(ax, cns='idf', palette=palette, labels=labels)
    plt.xticks([])
    plt.yticks([])
    
    f.set_size_inches(4.67,  4.06)
    plt.tight_layout()


    plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SSidf_incpeak_{}.pdf'.format(name[c]))
    plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SSidf_incpeak_{}.jpg'.format(name[c]), dpi=300)


#%% Delta increase of peak load
# list of polygons

#cases = [0,1]
#name = {0:'unc',
#        1:'tou'}
#
#polys = util.list_polygons(polygons, ies)
#cmap = plt.get_cmap('YlOrRd')
#pmax = 0.1
#xpalette = np.arange(0,pmax,0.05)
#palette = cmap(xpalette/pmax)
#labels = ['{}%'.format(int(i*100)) for i in xpalette]
#        
#
#colors = cmap([(data[0].IncreasePeak_pu[iris.SS[i]]-data[1].IncreasePeak_pu[iris.SS[i]]) / pmax for i in ies for p in polygons[i]])
#
#f, ax = plt.subplots()
#util.plot_polygons(polys, color=colors, ax=ax)
#util.aspect_carte_france(ax, palette=palette, labels=labels)
#plt.xticks([])
#plt.yticks([])
#f.set_size_inches(4.67,  4.06)
#plt.tight_layout()
#
#plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SSFR_incpeak_delta.pdf')
#plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SSFR_incpeak_delta.jpg', dpi=300)




#%% Try to derive something
# So, we have iris_poly which has many demog indexes:
    
# Three categories according to AU_CATG
# Grand pole = 111
# peri urban = 112-120
# small pole = 200-300
# rural = 400
    
# I'll further differentiate using AU size
# Grand pole - Center of big cities: Paris,

catg = {111: 'GP',
        112: 'PU',
        120: 'PU',
        200: 'SP',
        211: 'SP',
        212: 'SP',
        221: 'SP',
        222: 'SP',
        300: 'SP',
        400: 'R'}
catS = {0:0,
        1:1,
        2:1,
        3:1,
        4:1,
        5:1,
        6:1,
        7:2,
        8:2,
        9:3,
        10:3,
        }

catiris = iris_poly.apply(lambda x: catg[x.AU_CATG] + '_' + str(catS[x.AU_SIZE]), axis=1)

# defining SS type
catSS =  {}
for ss in ev_load.columns:
    ies = iris[iris.SS==ss].index
    caties = catiris[ies]
    catSS[ss] = caties.value_counts().idxmax()
catSS = pd.Series(catSS)  
#%% plot catiris:
#'PU_0', 'PU_3', 'SP_0', 'SP_1', 'PU_1', 'GP_1', 'PU_2', 'GP_3','GP_2', 'R_0'
cats = ['GP_3', 'GP_2', 'GP_1', 'PU_3', 'PU_2', 'PU_1', 'PU_0', 'SP_1', 'SP_0', 'R_0']
colors = ['yellow', 'y', 'orange', 'cornflowerblue', 'b', 'navy', 'tomato', 'g', 'g','g']

f, ax = plt.subplots()
for i, cat in enumerate(cats):
    polys = util.list_polygons(polygons, catiris[catiris==cat].index)
    util.plot_polygons(polys, ax=ax, color=colors[i])
palette = ['yellow', 'y', 'orange', 'cornflowerblue', 'b', 'navy','tomato', 'g']

labels = ['LUA', 'MUA', 'SUA', 'LAB', 'MAB', 'SAB','RMP', 'R']
util.aspect_carte_france(ax, palette=palette, labels=labels)
plt.xticks([])
plt.yticks([])
f.set_size_inches(4.67,  4.06)
plt.tight_layout()

plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\AU_cats.pdf'.format(name[c]))
plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\AU_cats.jpg'.format(name[c]), dpi=300)

#%% Plot categories of 
f, axs= plt.subplots(4,3, sharey=True, sharex=True)
cats = ['GP_3','PU_3', '',
        'GP_2','PU_2', '', 
        'GP_1','PU_1', 'SP_1',
        'R_0', 'PU_0', 'SP_0']
for i, cat in enumerate(cats):
    plt.sca(axs[i//3,i%3])
    plt.scatter(data[0].IncreaseDemand_pu[catSS[catSS==cat].index],data[0].IncreasePeak_pu[catSS[catSS==cat].index], s=1)
    plt.text(x=0.2,y=0.1,
             s='ID{:.2f}\nIP{:.2f}'.format(data[0].IncreaseDemand_pu[catSS[catSS==cat].index].mean(),
             data[0].IncreasePeak_pu[catSS[catSS==cat].index].mean()))
    plt.xlim(0,0.3)
    plt.ylim(0,0.35)
    plt.title(cat)
plt.tight_layout()
#%% plot scatter by 3 types: GP, PU & SP-R
cats = ['GP_3', 'GP_2', 'GP_1', 'PU_3', 'PU_2', 'PU_1', 'PU_0', 'SP_1', 'SP_0', 'R_0']
color= {'GP_3':'r', 'GP_2':'r', 'GP_1':'r', 
        'PU_3':'b', 'PU_2':'b', 'PU_1':'b', 'PU_0':'b', 
        'SP_1':'g', 'SP_0':'g', 'R_0':'g'}
labels= {'GP_3':'Large Urban Pole', 'GP_2':'MUP', 'GP_1':'SUP', 'PU_3':'Large Agglomeration Belt', 
         'PU_2':'_', 'PU_1':'_', 'PU_0':'_', 'SP_1':'Small poles/rural', 'SP_0':'_', 'R_0':'Rural'}
#ies = iris[iris.SS.isin(data[0].index)].index
colors = [color[catSS[iris.SS[i]]] for i in ies]
cases = [0,1]
name = {0:'unc',
        1:'tou'}

cats = ['GP_3', 'PU_3', 'R_0']

for c in cases:
    f, ax=plt.subplots()#(1,3)
    for i, cat in enumerate(cats):
#        if 'GP' in cat:
#            plt.sca(ax[0])
#        elif 'PU' in cat:
#            plt.sca(ax[1])
#        else:
#            plt.sca(ax[2])
        plt.scatter(data[c].IncreaseDemand_pu[catSS[catSS==cat].index]*100,
                    data[c].IncreasePeak_pu[catSS[catSS==cat].index]*100, 
                    s=1, c=color[cat], label=labels[cat], alpha=0.7)
        plt.ylabel('Peak load increase [%]')  
        plt.xlabel('Demand increase [%]')
        plt.legend()
        plt.xlim(0,30)
        plt.ylim(0,35)
    #plt.title('(b) Peak load')
    
#    f.set_size_inches(7.5,3)
    f.set_size_inches(3.5,3)
    plt.tight_layout()
    plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SS_dempeak_AU_{}.pdf'.format(name[c]))
    plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\SS_dempeak_AU_{}.jpg'.format(name[c]), dpi=300)
#%%
cats = ['GP_3', 'GP_2', 'GP_1', 'PU_3', 'PU_2', 'PU_1', 'PU_0', 'SP_1', 'SP_0', 'R_0']

kpis_di = {c: [data[0].IncreaseDemand_pu[catSS==c].shape[0],
             data[0].IncreaseDemand_pu[catSS==c].mean(), 
             data[0].IncreasePeak_pu[catSS==c].mean(),
             data[1].IncreasePeak_pu[catSS==c].mean()] 
             for c in cats} 
kpis_di = pd.DataFrame(kpis_di, index=['N_SS', 'IncreaseDemand_pu', 'IncreasePeak_unc', 'IncreasePeak_tou']).T
