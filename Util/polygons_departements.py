# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:28:34 2020

@author: U546416
"""


import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt
import time
import sys
import csv

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

#folder = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau'
folder = r'C:\Users\u546416\Downloads'
folder_polys = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
    
#iris_prod = pd.read_csv(folder + r'\production-electrique-par-filiere-a-la-maille-iris.csv',
#                        engine='python', sep=';')
dep_parc_prod = pd.read_csv(folder + r'\parc-des-installations-de-production-raccordees-par-departement.csv',
                            engine='python', sep=';')

dep_parc_prod.columns = util.fix_wrong_encoding_str(pd.Series(dep_parc_prod.columns))
for c in dep_parc_prod.columns:
    if dep_parc_prod[c].dtype=='object':
        try:
            dep_parc_prod[c] = util.fix_wrong_encoding_str(dep_parc_prod[c])
        except:
            pass


# Getting polygons for department

try:
    dep_polys = pd.read_csv(folder_polys + 'departements_polygons.csv', engine='python', index_col=0)
    polys= util.do_polygons(dep_polys)
except:
    dep_polys = {}
    for i, t in dep_parc_prod.iterrows():
        if type(t['Geo Point']) == str:
            data = {}
            data['Polygon'] = eval(t['Geo Shape'])['coordinates']
            data['GeoPoint'] = eval(t['Geo Point'])
            data['DEP_NAME'] = t['Nom Département']
            data['REG_NAME'] = t['Région']
            data['REG_CODE'] = t['Code Région']
            dep_polys[t['Code Département']] = data
    
    dep_polys = pd.DataFrame(dep_polys).T
    
    dep_polys.Polygon = dep_polys.Polygon.apply(lambda x: [y[0] for y in x] if len(x[0]) == 1 else x)
    polys = util.do_polygons(dep_polys)
    # Fix encodings
    dep_polys.DEP_NAME = util.fix_wrong_encoding_str(dep_polys.DEP_NAME)
    dep_polys.REG_NAME = util.fix_wrong_encoding_str(dep_polys.REG_NAME)
    
    # Save!
    dep_polys.to_csv(folder_polys + r'departements_polygons.csv')
    
if not ('TYPE_PV' in dep_parc_prod.columns):
    if len(dep_parc_prod.columns) == 14: 
        cols = ['REG_CODE', 'REG_NAME','DEP_CODE', 'DEP_NAME',  'CODE_PROD', 
                             'TYPE_PROD', 'ID_TYPE_INJ', 'TYPE_INJ', 'TRANCHE', 
                             'FinTrimestre', 'Nb_Inst', 'P_MW', 'GeoShape', 'GeoPoint']
    else:
        cols = ['REG_CODE', 'REG_NAME','DEP_CODE', 'DEP_NAME',  'CODE_PROD', 
                             'TYPE_PROD', 'ID_TYPE_INJ', 'TYPE_INJ','STOCKAGE', 'TRANCHE', 
                             'FinTrimestre', 'Nb_Inst', 'P_MW', 'GeoShape', 'GeoPoint']        
    dep_parc_prod.columns = cols
    code = {'a-]0;36]':'home_rooftop',
            'b-]36;100]':'commercial_rooftop',
            'c-]100;250]':'commercial_rooftop',
            'd-]250;...[':'solar_farm'}
    dep_parc_prod['TYPE_PV']  = dep_parc_prod.TRANCHE.apply(lambda x: code[x])

dep_parc_prod = dep_parc_prod[dep_parc_prod.TYPE_PROD=='Photovoltaïque']

#%% plot PV installed per departement
pv_inst = dep_parc_prod.groupby('Code Département')['Puissance MW'].sum()
cmap = plt.get_cmap('plasma')
polys = util.list_polygons(util.do_polygons(dep_polys, plot=False), dep_polys.index)
color = cmap([pv_inst[d]/pv_inst.max() for d in pv_inst.index for p in dep_polys.Polygon[d]])

ax = util.plot_polygons(polys, color=color)
tranches = np.arange(0,7,1) * 100
labels = ['{:3} MW'.format(t) for t in tranches]
palette = list(cmap([i/600 for i in tranches]))
util.aspect_carte_france(ax, title='PV Installed capacity', palette=palette, labels=labels)


#%% plot PV per installed per type (tranche)

print('Loading Conso per IRIS')
#folder_consodata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso'
#iris = pd.read_csv(folder_consodata + r'\IRIS_enedis_2017.csv', 
#                    engine='python', index_col=0)

pv_dep = dep_parc_prod.groupby(['DEP_CODE', 'TRANCHE']).P_MW.sum()
pv_inst_perc = pv_dep / dep_parc_prod.groupby(['DEP_CODE']).P_MW.sum()

for i in dep_parc_prod.TYPE_PV.unique():
    cmap = plt.get_cmap('plasma')
    color = cmap([pv_inst_perc[d] for d in pv_inst_perc.index for p in dep_polys.Polygon[d[0]] if d[1]==i])
    
    ax = util.plot_polygons(polys, color=color)
    tranches = np.arange(0,1,0.2) 
    labels = ['{:d}%'.format(int(t * 100)) for t in tranches]
    palette = list(cmap([j for j in tranches]))
    util.aspect_carte_france(ax, title='Ratio of {} in PV installed capacity'.format(i), palette=palette, labels=labels)
    

#%% Creating polygons for Regions
try:
    poly_reg =  pd.read_csv(folder_polys + r'regions_polygons.csv', index_col=0)
    polyreg = util.do_polygons(poly_reg)
except:
    poly_reg = pd.read_csv(r'C:\Users\u546416\Downloads\flexibilites-participant-au-ma-ou-a-nebef-par-region.csv', engine='python', sep=';')
    poly_reg.columns = ['REG_NAME', 'REG_CODE', 'm', 'me',
           'meh', 'Polygon', 'GeoPoint']
    poly_reg.drop(['m', 'me', 'meh'], axis=1, inplace=True)
    poly_reg.set_index('REG_CODE', inplace=True)
    poly_reg.Polygon = poly_reg.Polygon.apply(lambda x: eval(x))
    poly_reg.Polygon = poly_reg.Polygon.apply(lambda x: x['coordinates'] if x['type'] == 'Polygon' else x['coordinates'][0])
    polyreg = util.do_polygons(poly_reg)
     
    # Save!
    folder_polys = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
    poly_reg.to_csv(folder_polys + r'regions_polygons.csv')

#%% Plot according to region

pv_reg = dep_parc_prod.groupby(['REG_CODE', 'TYPE_PV']).P_MW.sum()
pv_reg_perc = pv_reg / dep_parc_prod.groupby(['REG_CODE']).P_MW.sum()

for i in dep_parc_prod.TYPE_PV.unique():
    cmap = plt.get_cmap('plasma')
    color = cmap([pv_reg_perc[r, i] for r in polyreg.keys() for p in polyreg[r]])
    
    ax = util.plot_polygons(util.list_polygons(polyreg, polyreg.keys()), color=color)
    tranches = np.arange(0,1,0.2) 
    labels = ['{:d}%'.format(int(t * 100)) for t in tranches]
    palette = list(cmap([j for j in tranches]))
    util.aspect_carte_france(ax, title='Ratio of {} in PV installed capacity'.format(i), palette=palette, labels=labels)
    
#%% plot in bars
f,ax=plt.subplots()
idx= pv_reg_perc.xs('home_rooftop',level='TYPE_PV').sort_values().index

regs = [dep_parc_prod[dep_parc_prod.REG_CODE==c].REG_NAME.iloc[0] for c in idx]


x=[-i for i in range(pv_reg_perc.xs('home_rooftop',level='TYPE_PV').shape[0])]
plt.barh(x,pv_reg_perc.xs('home_rooftop',level='TYPE_PV')[idx], label='Small-scale')
plt.barh(x,pv_reg_perc.xs('commercial_rooftop',level='TYPE_PV')[idx], 
         left=pv_reg_perc.xs('home_rooftop',level='TYPE_PV')[idx], label='Medium-scale')
plt.barh(x,pv_reg_perc.xs('solar_farm',level='TYPE_PV')[idx], 
         left=pv_reg_perc.xs('home_rooftop',level='TYPE_PV')[idx]+pv_reg_perc.xs('commercial_rooftop',level='TYPE_PV')[idx],
         label='Large-scale')
plt.title('(b) Share of PV sizes', y=-0.15)
plt.yticks(x, regs)
plt.xlim(0,1)
f.set_size_inches(4.3,  4.06)#(4,3.3)
plt.tight_layout()
f.legend(loc=9, ncol=4)
pos = ax.get_position()
dy = 0.04
ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\France_PV\share_pvtype_regions_small2021.pdf')
plt.savefig(r'c:\user\U546416\Pictures\France_PV\share_pvtype_regions_small2021.png')

    
#%% Computing PV penetration per region RES

pv_size = 0.003 #MW
nhomes = iris.Nb_RES * (100 - iris.Taux_logements_collectifs)/100
df = pd.DataFrame([nhomes, iris.REGION_CODE], index=['nhomes', 'REGION_CODE']).T
nres_reg = df.groupby('REGION_CODE').nhomes.sum()
nres_reg.index = [int(i) for i in nres_reg.index]

pv_perc = pv_reg[:, 'home_rooftop'] / (pv_size * nres_reg)

cmap = plt.get_cmap('plasma')
color = cmap([pv_perc[r] / 0.06 for r in polyreg.keys() for p in polyreg[r]])

ax = util.plot_polygons(util.list_polygons(polyreg, polyreg.keys()), color=color)
tranches = np.arange(0,0.06,0.01) 
labels = ['{:d}%'.format(int(t * 100)) for t in tranches]
palette = list(cmap([j/0.06 for j in tranches]))
util.aspect_carte_france(ax, title='Residential PV penetration per region'.format(i), palette=palette, labels=labels)


#%% Computing PV penetration per region - COMMERCIAL

pv_size = 0.120 #MW
ncomm = (iris.Conso_Tertiaire // 600) + (iris.Conso_Industrie // 600) + (iris.Nb_Agriculture * 2)
df = pd.DataFrame([ncomm, iris.REGION_CODE], index=['ncomm', 'REGION_CODE']).T
ncomm_reg = df.groupby('REGION_CODE').ncomm.sum()
ncomm_reg.index = [int(i) for i in ncomm_reg.index]

pv_comm_perc = pv_reg[:, 'commercial_rooftop'] / (pv_size * ncomm_reg)

max_p = 0.25

cmap = plt.get_cmap('plasma')
color = cmap([pv_comm_perc[r] / max_p for r in polyreg.keys() for p in polyreg[r]])

ax = util.plot_polygons(util.list_polygons(polyreg, polyreg.keys()), color=color)
tranches = np.arange(0,max_p,max_p/5) 
labels = ['{:d}%'.format(int(t * 100)) for t in tranches]
palette = list(cmap([j/max_p for j in tranches]))
util.aspect_carte_france(ax, title='Commercial PV penetration per region', palette=palette, labels=labels)

#%% Computing PV penetration per region - FARM

pv_size = 2 #MW
nrural_reg = iris[iris.IRIS_TYPE == 'Z'].groupby('REGION_CODE').IRIS_NAME.count()
nrural_reg.index = [int(i) for i in ncomm_reg.index]

pv_farm_perc = pv_reg[:, 'solar_farm'] / (pv_size * nrural_reg)

max_p = 0.5

cmap = plt.get_cmap('plasma')
color = cmap([pv_farm_perc[r] / max_p for r in polyreg.keys() for p in polyreg[r]])

ax = util.plot_polygons(util.list_polygons(polyreg, polyreg.keys()), color=color)
tranches = np.arange(0,max_p,(max_p+0.001)/5) 
labels = ['{:d}%'.format(int(t * 100)) for t in tranches]
palette = list(cmap([j/max_p for j in tranches]))
util.aspect_carte_france(ax, title='Solar Farm penetration per region', palette=palette, labels=labels)

#%% Total MW per region

cmap = plt.get_cmap('plasma')
pv_region = dep_parc_prod.groupby(['REG_CODE']).P_MW.sum()
max_p = 2000
color = cmap([pv_region[r] / max_p for r in polyreg.keys() for p in polyreg[r]])

ax = util.plot_polygons(util.list_polygons(polyreg, polyreg.keys()), color=color)
tranches = np.arange(0,max_p+0.001,(max_p)/5) 
labels = ['{:d} MW'.format(int(t)) for t in tranches]
palette = list(cmap([j/max_p for j in tranches]))
util.aspect_carte_france(ax,palette=palette, labels=labels)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.gcf().set_size_inches(4.3,  4.06)
plt.title('(a) PV installed capacity', y=-0.1)
plt.tight_layout()
plt.savefig(r'c:\user\U546416\Pictures\France_PV\installedcap_fr_2021.png')
plt.savefig(r'c:\user\U546416\Pictures\France_PV\installedcap_fr_2021.pdf')

#%% 
f, axs = plt.subplots(1,3)
maxt = np.round(pv_reg.max()/1000)*1000
tits = ['(a) Small-scale', '(b) Medium-scale', '(c) Large-scale']
for j,i in enumerate(['home_rooftop', 'commercial_rooftop', 'solar_farm']):
    cmap = plt.get_cmap('plasma')
    color = cmap([pv_reg[r, i]/maxt for r in polyreg.keys() for p in polyreg[r]])
    ax=axs[j]
    util.plot_polygons(util.list_polygons(polyreg, polyreg.keys()), color=color,ax=ax)
    tranches = np.arange(0,1,0.2) 
#    labels = ['{:d}%'.format(int(t * 100)) for t in tranches]
#    palette = list(cmap([j for j in tranches]))
    util.aspect_carte_france(ax)#, palette=palette, labels=labels)
    ax.set_title(tits[j],y=-0.15)
    plt.sca(ax)
    plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
plt.gcf().set_size_inches(7.48,3)
plt.tight_layout()

