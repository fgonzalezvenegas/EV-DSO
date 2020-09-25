# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 02:21:38 2020

@author: U546416
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as ptc
from matplotlib.collections import LineCollection, PatchCollection
import util

folder = r'Results_log//'
av_w  = ['0_24', '17_20']
nameset = ['Company', 'Commuter_HP', 'Commuter_LP']
v2g = ['V2G', 'V1G']
baseline = ['Enedis', 'UKPN']

data = []
c = [0,3,4,5,8,9,10,13,14]
for w in av_w:
    for n in nameset:
        for v in v2g:
            for b in baseline:                
                filehead = n + '_' + w + '_' + v + '_' + b + '.csv'
                stat = pd.read_csv(folder + filehead, index_col=0)
                data.append([n, v, b, w] + list(stat.iloc[-1,c].values))

cols = ['Fleet', 'VxG', 'Baseline', 'Window'] + list(stat.columns[c])
data = pd.DataFrame(data, columns=cols)

#%% Compute errors

cs = [i + j for i in ['Bids', 'Payments'] for j in ['_Avg', '_perc_h', '_perc_l']]
data[cs] = data[cs]/stat.index[-1]


data['Bids_err_l'] = data.Bids_Avg - data.Bids_perc_l 
data['Bids_err_h'] = data.Bids_perc_h - data.Bids_Avg 
data['Pay_err_l'] = data.Payments_Avg - data.Bids_perc_l
data['Pay_err_h'] = data.Payments_perc_h - data.Bids_Avg 

#%% Plot 2d Bids/EV vs Remuneration/kW, separated for V2G & V1G
color = ['b', 'r', 'g']
markers = ['o', '<']
bl = ['30-min', 'Unique']
for k, v in enumerate(v2g):
    plt.subplots()
    for i, n in enumerate(nameset):
        vv = data[data.VxG == v]
        d = vv[vv.Fleet == n]
        for j, b in enumerate(baseline):
            dd = d[d.Baseline == b]
            plt.scatter(dd.Bids_Avg, dd.Payments_Avg/dd.Bids_Avg, 
                        label=n+ ' ' + bl[j], color=color[i], marker=markers[j])
            
    if k == 0:
        plt.hlines(40, xmin=0, xmax=100, color='k', linestyle='--')
        plt.axis((0,10,0,55))
        plt.annotate(s='Evening Window', xy=(1,42))
        plt.annotate(s='', xy=(0.9,45), xytext=(0.9,41), arrowprops=dict(arrowstyle='->', color='black'))
        plt.annotate(s='Full-day Window', xy=(1,38), horizontalalignment='left', verticalalignment='top')
        plt.annotate(s='', xy=(0.9,35), xytext=(0.9,39), arrowprops=dict(arrowstyle='->', color='black'))
#        plt.legend(loc=4)
    else:
        plt.vlines(1.1, ymin=0, ymax=100, color='k', linestyle='-.')
        plt.axis((0,2,0,55))
        plt.annotate(s='Evening Window', xy=(1.15,25))
        plt.annotate(s='', xy=(1.5,28), xytext=(1.15,28), arrowprops=dict(arrowstyle='->', color='black'))
        plt.annotate(s='Full-day Window', xy=(1.05,25), horizontalalignment='right')
        plt.annotate(s='', xy=(0.7,28), xytext=(1.05,28), arrowprops=dict(arrowstyle='->', color='black'))
#        plt.legend(loc=0)
    
    plt.hlines(50, xmin=0, xmax=100, color='k', linestyle='--', lw=0.5, alpha=0.5)
    plt.xlabel('Bid per EV [kW]') 
    plt.ylabel('Remuneration per bid kW [€/kW]')


#%% Plot 2d Bids/EV vs Remuneration with, all in one graph
color = ['b', 'r', 'g']
markers = ['o', 'd']
bl = ['30-min', 'Unique']

f, ax = plt.subplots()

xline = 2.3
plt.vlines(xline, ymin=0, ymax=500, color='k', linestyle='-.')
plt.axvspan(0,xline,0,55,facecolor='paleturquoise', alpha=0.3)
plt.axvspan(xline,10,0,55,facecolor='yellow', alpha=0.3)

for i, n in enumerate(nameset):
    d = data[data.Fleet == n]
    for j, b in enumerate(baseline):
        dd = d[d.Baseline == b]
        plt.scatter(dd.Bids_Avg, dd.Payments_Avg, 
                    label=n+ ' ' + bl[j], zorder=10, 
                    marker=markers[j], edgecolor='k', 
                    linewidths=0.5, color=color[i])
        


plt.plot([0,10], [0,500], color='k', linestyle='--', lw=0.5, alpha=0.5)
plt.xlabel('Bid per EV [kW]') 
plt.ylabel('Remuneration [€]')
plt.axis((0,10,0,500))
#% Add rectangles for Windows
ang = np.arctan(50)*180/np.pi
a = ptc.Polygon([(0.1,0), 
                 (0.1, 50), 
                 (1,50), 
                 (1, 0)], 
        facecolor='none', edgecolor='purple', linestyle='--')
b = ptc.Polygon([(1.15,0), 
                 (1.15, 100), 
                 (2,100), 
                 (2, 0)], 
        facecolor='none', edgecolor='midnightblue', linestyle='--')
c = ptc.Polygon([(2.5, 29 *2.5), 
                 (2.5, 40 * 2.5), 
                 (2.5+3.8, (2.5+3.8)*40), 
                 (2.5+3.8, 29 * 2.5)], 
        facecolor='none', edgecolor='purple', linestyle='--')
d = ptc.Polygon([(3.8, 43*3.8), 
#                 (3.8, (3.8+5.4) * 52),
                 (3.8, (3.8+5.4) * 52),
                 (3.8+5.4, (3.8+5.4) * 52),
                 (3.8+5.4, 43*(3.8+5.4)),
                 ], 
        facecolor='none', edgecolor='midnightblue', linestyle='--')

for r in [a, b, c, d]:
    ax.add_patch(r)
plt.show()

#% annotate
plt.text(x=0.6, y=70, s='Full-day\nWindow', horizontalalignment='center', verticalalignment='bottom')
plt.text(x=1.65, y=120, s='Evening\nWindow', horizontalalignment='center', verticalalignment='bottom')
plt.text(x=2.5+3, y=80, s='Full-day Window', horizontalalignment='right', verticalalignment='bottom')
plt.text(x=3.9, y=43*(3.8+5.3), s='Evening Window', horizontalalignment='left', verticalalignment='bottom')
#%
plt.text(x=xline-0.7, y=380, s='V1G', horizontalalignment='center', verticalalignment='bottom', fontsize='xx-large')
plt.text(x=xline+0.7, y=380, s='V2G', horizontalalignment='center', verticalalignment='bottom', fontsize='xx-large')
plt.legend(loc=4)

#%% Plot 2d Bids/EV vs Remuneration with, all in one graph
color = ['b', 'r', 'g']
markers = ['o', 'd']
bl = ['30-min', 'Unique']

f, ax = plt.subplots()

xline = 2.3
plt.vlines(xline, ymin=0, ymax=500, color='k', linestyle='-.')
plt.axvspan(0,xline,0,55,facecolor='paleturquoise', alpha=0.3)
plt.axvspan(xline,10,0,55,facecolor='yellow', alpha=0.3)

for i, n in enumerate(nameset):
    d = data[data.Fleet == n]
    for j, b in enumerate(baseline):
        dd = d[d.Baseline == b]
        plt.scatter(dd.Bids_Avg, dd.Payments_Avg, 
                    label=n+ ' ', zorder=10, 
                    marker=markers[j], edgecolor='k', 
                    linewidths=0.5, color=color[i])
        if j==0:
            break
        


plt.plot([0,10], [0,500], color='k', linestyle='--', lw=0.5, alpha=0.5)
plt.xlabel('Bid per EV [kW]') 
plt.ylabel('Remuneration [€]')
plt.axis((0,10,0,500))
#% Add rectangles for Windows
ang = np.arctan(50)*180/np.pi
a = ptc.Polygon([(0.1,0), 
                 (0.1, 50), 
                 (1,50), 
                 (1, 0)], 
        facecolor='none', edgecolor='purple', linestyle='--')
b = ptc.Polygon([(1.15,0), 
                 (1.15, 100), 
                 (2,100), 
                 (2, 0)], 
        facecolor='none', edgecolor='midnightblue', linestyle='--')
angc=np.arctan((240-60)/(6.3-2.3))*180/np.pi
w=np.sqrt((240-60)**2+(6.3-2.3)**2)
c = ptc.Ellipse(xy= (4.3,150), width=w, height=1.5, angle=angc,
        facecolor='none', edgecolor='purple', linestyle='--')
angd=np.arctan((460-150)/(9.2-3.3))*180/np.pi
w=np.sqrt((460-150)**2+(9.2-3.3)**2)
d = ptc.Ellipse(xy= (6.25,305), width=w, height=1.5, angle=angd,
        facecolor='none', edgecolor='midnightblue', linestyle='--')

for r in [a, b, c, d]:
    ax.add_patch(r)
plt.show()

#% annotate
plt.text(x=0.6, y=70, s='Full-day\nWindow', horizontalalignment='center', verticalalignment='bottom')
plt.text(x=1.65, y=120, s='Evening\nWindow', horizontalalignment='center', verticalalignment='bottom')
plt.text(x=2.5+3, y=50, s='Full-day Window', horizontalalignment='right', verticalalignment='bottom')
plt.text(x=2.1+3, y=290,s='Evening Window', horizontalalignment='right', verticalalignment='bottom')
#%
plt.text(x=xline-0.7, y=380, s='V1G', horizontalalignment='center', verticalalignment='bottom', fontsize='xx-large')
plt.text(x=xline+0.7, y=380, s='V2G', horizontalalignment='center', verticalalignment='bottom', fontsize='xx-large')
plt.legend(loc=4)

#%% Plot 2d Bids/EV vs Remuneration/kW with, all in one graph
color = ['b', 'r', 'g']
markers = ['o', 'd']
#bl = ['30-min', 'Unique']
bl = ['Unique', '30-min']

f, ax = plt.subplots()

xline = 2.3
plt.vlines(xline, ymin=0, ymax=100, color='k', linestyle='-.')
plt.axvspan(0,xline,0,55,facecolor='paleturquoise', alpha=0.3)
plt.axvspan(xline,10,0,55,facecolor='yellow', alpha=0.3)

for i, n in enumerate(nameset):
    d = data[data.Fleet == n]
    for j, b in enumerate(baseline):
        dd = d[d.Baseline == b]
        plt.scatter(dd.Bids_Avg, dd.Payments_Avg/dd.Bids_Avg, 
                    label=n+ ' ' + bl[j], zorder=10, 
                    marker=markers[j], edgecolor='k', 
                    linewidths=0.5, color=color[i])
        


plt.hlines(50, xmin=0, xmax=100, color='k', linestyle='--', lw=0.5, alpha=0.5)
plt.xlabel('Bid per EV [kW]') 
plt.ylabel('Remuneration per bid kW [€/kW]')
plt.axis((0,10,0,55))
#% Add rectangles for Windows
a = ptc.Rectangle(xy=(0.1,8), width=0.9, height=44, facecolor='none', edgecolor='purple', linestyle='--')
b = ptc.Rectangle(xy=(1.15,26), width=0.9, height=26, facecolor='none', edgecolor='midnightblue', linestyle='--')
c = ptc.Rectangle(xy=(2.5,29), width=3.8, height=11, facecolor='none', edgecolor='purple', linestyle='--')
d = ptc.Rectangle(xy=(3.8,43), width=5.4, height=9, facecolor='none', edgecolor='midnightblue', linestyle='--')
for r in [a, b, c, d]:
    ax.add_patch(r)
plt.show()

#% annotate
plt.text(x=1.65, y=25, s='Evening\nWindow', horizontalalignment='center', verticalalignment='top')
plt.text(x=0.6, y=7, s='Full-day\nWindow', horizontalalignment='center', verticalalignment='top')
plt.text(x=3.8+5.3, y=43, s='Evening Window', horizontalalignment='right', verticalalignment='bottom')
plt.text(x=2.5+3.7, y=29, s='Full-day Window', horizontalalignment='right', verticalalignment='bottom')
#%
plt.text(x=1.7, y=13, s='V1G', horizontalalignment='center', verticalalignment='bottom', fontsize='xx-large')
plt.text(x=3.5, y=13, s='V2G', horizontalalignment='center', verticalalignment='bottom', fontsize='xx-large')
plt.legend(loc=4)
    

 
#%% Show minimum fleet size according to bids
    
folder = r'Results_log//'
av_w  = ['0_24', '17_20']
nameset = ['Company', 'Commuter_HP', 'Commuter_LP']
v2g = ['V2G', 'V1G']
baseline = ['Enedis', 'UKPN']

data_bs = []
c = [0,3,4,5,8,9,10,13,14]
for w in av_w:
    for n in nameset:
        for v in v2g:
            for b in baseline:
                if b == baseline[0]:                
                    filehead = n + '_' + w + '_' + v + '_' + b + '.csv'
                    stat = pd.read_csv(folder + filehead, index_col=0)
                    bid50 = 50/(stat.Bids_max/stat.index)
                    bid50 = bid50[bid50 < 1000]
                    data_bs.append([n, v, b, w] + [np.ceil(bid50.max()) , np.ceil((500/(stat.Bids_Avg.iloc[-1]/stat.index[-1])))])

cols = ['Fleet', 'VxG', 'Baseline', 'Window', 'minbid_50', 'minbid_500']
data_bs = pd.DataFrame(data_bs, columns=cols)
print(data_bs)

#%%
f,ax=plt.subplots()

for i in range(0,180, 30):
    z = ptc.Rectangle((0.5,0.5), 0.2,0.2,angle=i, alpha=0.2)
    ax.add_patch(z)
plt.show()