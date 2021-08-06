# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:05:59 2020

@author: U546416
"""
import pandas as pd
import util
import numpy as np
import matplotlib.pyplot as plt
import time
# Load data:
ts = []
# Polygons:
ts.append(time.time())
#print('load iris')
#iris = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\IRIS_all_geo_2016.csv',
#                        engine='python', index_col=0)
#ts.append(time.time())
#print('Time: {}s'.format(round(ts[-1]-ts[-2],1)))
#print('polygons')
#polygons = util.do_polygons(iris)
#polygons_ss = util.load_polygons_SS()
ts.append(time.time())
print('Time: {}s'.format(round(ts[-1]-ts[-2],1)))
print('communes')
comms = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\COMM_all_geo_2019.csv',
                        engine='python', index_col=0)
polygons_comm = util.do_polygons(comms)
ts.append(time.time())
print('Time: {}s'.format(round(ts[-1]-ts[-2],1)))

# Histograms of distance data
print('histograms')
folder_hdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité'
hhome = pd.read_csv(folder_hdata + r'\HistHomeModal.csv', 
                    engine='python', index_col=0)
hwork = pd.read_csv(folder_hdata + r'\HistWorkModal.csv', 
                    engine='python', index_col=0)
hhome = hhome.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)
hwork = hwork.drop(['ZE', 'Status', 'UU', 'Dep'], axis=1)

comm2019 = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\Modifs_commune\Comms2016_2019.csv',
                       engine='python', index_col=0)
ts.append(time.time())
print('Time: {}s'.format(round(ts[-1]-ts[-2],1)))
print('AU')
AU = pd.read_excel(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\AU_coms.xlsx')
AU = AU.set_index('CODGEO')
AU = AU.drop(AU[AU.DEP.isin(['2A', '2B', '971', '972'])].index)
AU.index = AU.index.astype(int)
ts.append(time.time())
print('Time: {}s'.format(round(ts[-1]-ts[-2],1)))

#Setting plt settings to small
font = {'size':8}
plt.rc('font', **font)
#%% Plot avg distance

nevsh = hhome.sum(axis=1)
nevsw = hhome.sum(axis=1)
min_nev = 50
nevsh = nevsh[nevsh>min_nev]
nevsw = nevsw[nevsw>min_nev]
bins = np.array([i*2+1 for i in range(50)])
avgh = (hhome.loc[nevsh.index] * bins).sum(axis=1) / nevsh
avgw = (hwork.loc[nevsw.index] * bins).sum(axis=1) / nevsw

#%% Plot using colormap

idxs = comm2019.CODE_COMM_2019[nevsh.index].dropna()
idxs = idxs[idxs.isin(polygons_comm)]
cmap = plt.get_cmap('plasma')
polys = util.list_polygons(polygons_comm, idxs)
colors = cmap([avgh[i]/50 for i in idxs.index for p in polygons_comm[idxs[i]]])

ax = util.plot_polygons(polys, color=colors)
tranches = [10, 30, 50, 70, 90]
labels = ['{:3}km<d<{:3}km'.format(str(t-10),str(t+10)) for t in tranches]
palette = list(cmap([i/100 for i in tranches]))
util.aspect_carte_france(ax, title='Average daily commuting distance [km]\n by commune of residence', palette=palette, labels=labels)

#%% Plot using green-blue-red palette
palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
wbin = 15 #km of daily distance
tranches = [i*wbin for i in range(len(palette))]
labels = ['{:3}km<d<{:3}km'.format(str(t),str(t+wbin)) for t in tranches]
labels[-1] = labels[-1][:-6]

colors = [palette[min(int(avgh[i]/(wbin/2)), len(palette)-1)] for i in idxs.index for p in polygons_comm[idxs[i]]]
ax = util.plot_polygons(polys, facecolor=colors)
#util.aspect_carte_france(ax, title='Average daily commuting distance [km]\n by commune of residence', palette=palette, labels=labels)

util.aspect_carte_france(ax, palette=palette, labels=labels)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

plt.tight_layout()
plt.gcf().set_size_inches(4.67,  4.06)

plt.xlim(-6.8,8.9012)
a = ax.axis()
r = util.compute_aspect_carte(*a)
ax.set_aspect(r)
#%%
#ax.figure.dpi=300
#plt.gcf().set_size_inches(4.67,  4.06)
plt.savefig(r'c:\user\U546416\Pictures\DataMobilité\DataDistance\dist_thesis.pdf')
plt.savefig(r'c:\user\U546416\Pictures\DataMobilité\DataDistance\dist_thesis.png', dpi=300)
plt.savefig(r'c:\user\U546416\Pictures\DataMobilité\DataDistance\dist_thesis.jpg',quality=95, dpi=300)

#%% Compute distance profiles by comm AU_CATG
Catgs = {'Urban': [111], 'PeriUrban': [112, 120], 'Small pole': [211, 212, 220, 300], 'Rural' : [400]}
f, ax = plt.subplots()
avg_dd = pd.DataFrame(columns=hhome.columns)
for cat in Catgs:
    comms = AU[AU.CATAEU2010.isin(Catgs[cat])].index
    comms = comms[comms.isin(hhome.index)]
    d_dist = hhome.loc[comms].sum(axis=0) / hhome.loc[comms].sum(axis=1).sum()
    avg_dd.loc[cat] = d_dist
hp = hhome[(hhome.index / 1000 >75) & (hhome.index / 1000 <76)]
avg_dd.loc['Paris'] = hp.sum(axis=0)/hp.sum().sum()

#avg = round((avg_dd.loc['Paris'] * bins).sum(),1)
#ax.plot(bins/2, avg_dd.loc['Paris'], label='Paris' + ', avg={} km'.format(avg*2))
for cat in avg_dd.index:
    if cat=='Paris':
        continue
    avg = round((avg_dd.loc[cat] * bins).sum(),1)
    ax.plot(bins, avg_dd.loc[cat], label=cat + ', avg={} km'.format(avg))
plt.xlabel('Distance [km]')
plt.ylabel('Density')
plt.title('One-way commuting distance distribution')
plt.legend()