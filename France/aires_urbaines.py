# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:54:23 2020
Aires urbaines
@author: U546416
"""

import util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#polygons = util.load_polygons_iris()
#%% Read data

# iris info, geographie 2016
folder = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
file = 'IRIS_all_geo_2016.csv'
iris_poly = pd.read_csv(folder+file,
                        engine='python', index_col=0)
polygons = util.do_polygons(iris_poly, plot=False)
# AU info
AU = pd.read_excel(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\AU_coms.xlsx')
AU_taille = pd.read_excel(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\AU.xls')

# Modifications of communes since 2016
all_comms = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\Modifs_commune\France2018.txt',
                        engine='python', sep='\t')
all_comms = all_comms.drop(all_comms[all_comms.DEP.isin(['2A', '2B', '971', '972'])].index, axis=0)
code_comms = all_comms.DEP.astype(int) * 1000 + all_comms.COM
rev = pd.Series(all_comms.index, index=code_comms)
# Modifs 2019

modifs2019 = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\Modifs_commune\mvtcommune-01012019.csv',
                        engine='python', sep=',')
#%% Cleaning files
AU = AU.set_index('CODGEO')
AU = AU.drop(AU[AU.DEP.isin(['2A', '2B', '971', '972'])].index)
AU.index = AU.index.astype(int)
AU_taille = AU_taille.set_index('AU_code')


modifs2019.com_av = modifs2019.com_av.apply(lambda x: 0 if x[1] in ['A', 'B'] else int(x))
modifs2019.com_ap = modifs2019.com_ap.apply(lambda x: 0 if x[1] in ['A', 'B'] else int(x))

#%% Getting 2019 commune for each 2016 iris
AU_iris = AU.AU2010[iris_poly.COMM_CODE]
nans = AU_iris[AU_iris.isnull()]

COMM_2019 = {}
c = 0
arrspml = [75001 + i for i in range(20)] + [13201 + i for i in range(16)] + [69381 + i for i in range(9)]
for i in iris_poly.index:
    c+=1
    if c%1000 == 0:
        print(c)
    com = iris_poly.COMM_CODE[i]
    if com in AU.index:
        COMM_2019[i] = com
#    elif com in arrspml:
#        # arrondisement Paris, lyon, marseille
#        COMM_2019[i] = com
    else:
        idx = rev[com]
        if type(idx)!= np.int64:
            idx = idx.iloc[0]
        if type(all_comms.POLE[idx]) == str:
            COMM_2019[i] = int(all_comms.POLE[idx])
        else:
             # check modifs 2019
            COMM_2019[i] = modifs2019[(modifs2019.com_av == com) & (modifs2019.typecom_ap == 'COM')].com_ap.iloc[0]

# Re-checing some communes 
COMM_2019 = pd.Series(COMM_2019)

AU_iris = AU.AU2010[COMM_2019]
nans = AU_iris[AU_iris.isnull()]

for com in nans.index:
    c = modifs2019[(modifs2019.com_av == com) & (modifs2019.typecom_ap == 'COM')].com_ap.iloc[0]
    for i in COMM_2019[COMM_2019 == com].index:
        COMM_2019.loc[i] = c

AU_iris = AU.AU2010[COMM_2019]
nans = AU_iris[AU_iris.isnull()]
AU_iris.index = iris_poly.index
#%% Set AU data in IRIS file
iris_poly['AU_CODE'] = AU_iris
auname = AU.LIBAU2010[COMM_2019]
ausize = AU_taille.TAU2016[AU.AU2010[COMM_2019]]
aucatg = AU.CATAEU2010[COMM_2019]
auname.index = iris_poly.index
ausize.index = iris_poly.index
aucatg.index = iris_poly.index
iris_poly['AU_NAME'] = auname
iris_poly['AU_CATG'] = aucatg
iris_poly['AU_SIZE'] = ausize

iris_poly.to_csv(folder + file)
#%% plot according to AU size

def get_idx_tranches(tranches, data):
    idxs = []
    for t in range(len(tranches)-1):
        idxs.append(data[(data>=tranches[t]) & (data<tranches[t+1])].index)
    return idxs

cmap = plt.get_cmap('plasma')
tranches_size = [0,1,7,9,10,11]
idxs = get_idx_tranches(tranches_size, iris_poly.AU_SIZE)
f, ax = plt.subplots()
for i in range(len(idxs)):
    ps = util.list_polygons(polygons, idxs[i])
    util.plot_polygons(ps, color=cmap(i/(len(idxs)-1)), ax=ax)
plt.title('Taille Aire Urbaine')
labels = ['Rural', 'AU<100k', '100k<AU<500k', '500k<AU', 'Paris']
cols = cmap([i/(len(idxs)-1) for i in range(len(idxs))])
util.aspect_carte_france(ax, palette=cols, labels=labels) 
#ax.set_title('Population of Urban Area (AU)')
plt.xticks([])
plt.yticks([])
f.set_size_inches(4.67,  4.06)
plt.tight_layout()
plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\aireUrbaine.pdf')
plt.savefig(r'c:\user\U546416\Pictures\SS - results\Thesis\aireUrbaine.jpg', dpi=300)


#%% Plot 2 according to Category of commune:
# Grand pole = 111
# peri urban = 112-120
# small pole = 200-300
# rural = 400
tranches_cat = [100, 111.5, 200, 400, 500]

idxs = get_idx_tranches(tranches_cat, iris_poly.AU_CATG)
cmap = plt.get_cmap('plasma')
f, ax = plt.subplots()
for i in range(len(idxs)):
    ps = util.list_polygons(polygons, idxs[i])
    util.plot_polygons(ps, color=cmap(1-i/(len(idxs)-1)), ax=ax)
labels = ['Urban', 'Peri-urban', 'Small pole', 'Rural']
cols = cmap([1-i/(len(idxs)-1) for i in range(len(idxs))])
util.aspect_carte_france(ax, palette=cols, labels=labels) 
ax.set_title('Category of commune')