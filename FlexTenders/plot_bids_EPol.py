# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 01:01:03 2020

@author: U546416
"""
    
import numpy as np
from matplotlib import pyplot as plt
#import EVmodel
#import scipy.stats as stats
#import time
#import util
#import flex_payment_funcs
import pandas as pd


#%% Load data
sets = [ 'Company', 'Commuter_HP', 'Commuter_LP']
sstr = [s.replace('_',' ') for s in sets]
folder = r'C:\Users\u546416\AnacondaProjects\EV-DSO\FlexTenders\EnergyPolicy\\'
av_w  = ['0_24', '17_20']

#%% Get data for diff service times
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
res = {}
times = [30,60,120]
conf = 0.9
thr = 0.6
pen=0
fleet=31
#fleet = 71
#fleet=500

for s in sets:
    for a in av_w:
        for f in times:
            filehead = 'full_' + s + '_' + a + '_'
            data = pd.read_csv(folder + filehead + 'VxG.csv', index_col=[0,1,2,3,4,5])
            res['v1g', s, a, f] = data[bds].loc['v1g', fleet,f,conf,thr,pen].values
            res['v2g', s, a, f] = data[bds].loc['v2g', fleet,f,conf,thr,pen].values
res = pd.DataFrame(res, index=bds).T

##%% Plot
#c = ['b', 'r', 'g']
#ls=['-',':', '--']
#for a in av_w:
#    f, ax = plt.subplots()
#    for i, s in enumerate(sets):
#        avg = res.Bids_Avg['v2g',s, a].values
#        ax.plot(avg, c[i] + '*' + '-', label='V2G ' + sstr[i])
#        avg = res.Bids_Avg['v1g',s, a].values
#        ax.plot(avg, c[i] + 'o'+ '--', alpha=0.5, label='V1G ' + sstr[i])
#    f.suptitle(a)
#    plt.legend()
#    plt.xticks([0,1,2], [30,60,120])
#    plt.xlabel('Service duration')
#    plt.ylabel('Bid [kW]')
#    
#%% Plot bids with diff service times with bars
c = ['b', 'r', 'g']
ls=['-',':', '--']
av_w  = ['17_20', '0_24']
hatches = ['', "\\\\",'xx']
chat = ['none', 'maroon', 'k']
x = np.arange(0,3)
f, ax = plt.subplots()
for i, s in enumerate(sets):
    for j, a in enumerate(av_w):
        avg = res.Bids_Avg['v1g',s, a].values
        ax.bar(x+3*i + 9*j, avg, width=0.9, facecolor=c[i], label=('_' if j else '') + 'V1G ' + sstr[i], hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
        avg = res.Bids_Avg['v2g',s, a].values
        ax.bar(x+3*i + 9*j, avg, width=0.9, facecolor=c[i], alpha=0.5, label=('_' if j else '') + 'V2G ' + sstr[i], hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
        print(s, a, 'v1g', hatches[i], chat[i])

plt.axvline(8.5, color='k', linestyle='--')
plt.legend(loc=7)
plt.ylabel('Bid [kW]')
plt.xlabel('Service duration [minutes]')
plt.xticks(np.arange(0,3*6), np.tile(np.array([30,60,120]), 6))    
plt.text(3*6*1/4,6.8, 'Evening Window',horizontalalignment='center', fontweight='bold')
plt.text(3*6*3/4,6.8, 'Full-day Window',horizontalalignment='center', fontweight='bold')
plt.grid('--', alpha=0.3)
plt.xlim(-1,18)
f.set_size_inches(8.5,4)
#f.suptitle(fleet)

#%% Get data for diff fleet sizes
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
pay = ['Payments_Avg', 'Payments_perc_l', 'Payments_perc_h']

res = {}
t = 30
conf = 0.9
thr = 0.6
pen=0
#fleet=31
#fleet = 71
fleet_range = [10,500]
nrange = 25
fleets = np.logspace(np.log10(fleet_range[0]), np.log10(fleet_range[1]), num=nrange).round(0)

for s in sets:
    for a in av_w:
        for f in fleets:
            filehead = 'full_' + s + '_' + a + '_'
            data = pd.read_csv(folder + filehead + 'VxG.csv', index_col=[0,1,2,3,4,5])
            res['v1g', s, a, f] = data[bds+pay].loc['v1g', f,t,conf,thr,pen].values
            res['v2g', s, a, f] = data[bds+pay].loc['v2g', f,t,conf,thr,pen].values
res = pd.DataFrame(res, index=[bds+pay]).T



        
        
#%% Get all data: only 30min
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
pay = ['Payments_Avg', 'Payments_perc_l', 'Payments_perc_h']
und = ['UnderDel_Avg', 'UnderDel_perc_l', 'UnderDel_perc_h']
res = {}
t = 30
conf = [0.5, 0.9, 0.99]
thr = [0.6,0.8]
pen = [0,17.5,50]
#fleet=31
#fleet = 71
fleet_range = [10,500]
nrange = 25
fleets = np.logspace(np.log10(fleet_range[0]), np.log10(fleet_range[1]), num=nrange).round(0)

for a in av_w:
    for s in sets:    
        filehead = 'full_' + s + '_' + a + '_'
        data = pd.read_csv(folder + filehead + 'VxG.csv', index_col=[0,1,2,3,4,5])
        for f in fleets:
            for c in conf:
                for th in thr:
                    for p in pen:   
                        res['v1g', a, s, f, c, th, p] = data[bds+pay+und].loc['v1g', f,t,c,th,p].values
                        res['v2g', a, s, f, c, th, p] = data[bds+pay+und].loc['v2g', f,t,c,th,p].values
res = pd.DataFrame(res, index=[bds+pay+und]).T

#%% Plot Bids @ various conf, V2G & V1G
c = ['b', 'r', 'g']
x = np.arange(0,3)
a = '17_20'
fl=31
vxg = ['v1g', 'v2g']
#f = 71
#f=500
th = 0.6
p=0
alphas = [1,0.5]
vst = ['V1G', 'V2G']
conf = [0.5, 0.9, 0.99]

hatches = ['', "\\\\",'xx']
chat = ['none', 'maroon', 'k']

f, ax = plt.subplots()
for k, a in enumerate(av_w):    
    for i, s in enumerate(sets):
        for j, v in enumerate(vxg):
        #    for j, a in enumerate(av_w):
            avg = res.Bids_Avg.loc[v, a, s, fl, :, th,p].values
            low = res.Bids_perc_l.loc[v, a, s, fl, :, th,p].values
            high = res.Bids_perc_h.loc[v, a, s, fl, :, th,p].values
            
            ax.bar(x+3*i+9*k, avg, width=0.9, color=c[i], alpha=alphas[j], 
                   label=('_' if k else '') + vst[j] + ' ' + sstr[i],
                   hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
            ax.errorbar(x+3*i+9*k, avg, 
                         yerr=[avg - low, high - avg],
                         linestyle='',
                         elinewidth=0.8, capsize=1.5, ecolor='k')
        
    plt.legend(loc=7)
    plt.ylabel('Bid [kW]')
    plt.xlabel('Confidence level')
    plt.xticks(np.arange(0,3*6), np.tile(np.array(conf), 6))
    plt.axvline(8.5, color='k', linestyle='--')    
    plt.text(3*6*1/4,6.8, 'Evening Window',horizontalalignment='center', fontweight='bold')
    plt.text(3*6*3/4,6.8, 'Full-day Window',horizontalalignment='center', fontweight='bold')
    plt.grid('--', alpha=0.3)
    plt.xlim(-1,18)
#    avg = res.Bids_Avg['v1g',s, a].values
#    ax.bar(x+3*i + 9*j, avg, width=0.9, color=c[i], label=('_' if j else '') + 'V1G ' + sstr[i])
#    f.suptitle('nfleet {}'.format(fl))
    f.set_size_inches(8.5,4)
    
#%% Remuneration for diffs thresholds, independent plots/figures
    
c = ['b', 'r', 'g']
x = np.arange(0,3)
a = '17_20'
fl=98
vxg = ['v1g', 'v2g']
#f = 71
#f=500
th = 0.6
p=0
alphas = [1,0.5]
#ths_ps = [{'t':t,'p':p} for t in [0.6,0.8] for p in [0,17.5,50]]
ths_ps = [{'t':0.6,'p':0},{'t':0.8,'p':17.5},{'t':0.8,'p':50}]
conf = [0.5, 0.9, 0.99]

vst = ['V1G', 'V2G']
for tp in ths_ps:
    f, ax = plt.subplots()
    for k, a in enumerate(av_w):    
        for i, s in enumerate(sets):
            for j, v in enumerate(vxg):
            #    for j, a in enumerate(av_w):
                avg = res.Payments_Avg.loc[v, a, s, fl, :, tp['t'],tp['p']].values
                low = res.Payments_perc_l.loc[v, a, s, fl, :, tp['t'],tp['p']].values
                high = res.Payments_perc_h.loc[v, a, s, fl, :,  tp['t'],tp['p']].values
                
                ax.bar(x+3*i+9*k, avg, width=0.9, color=c[i], alpha=alphas[j], 
                       label=('_' if k else '') + vst[j] + ' ' + sstr[i], 
                       hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
                ax.errorbar(x+3*i+9*k, avg, 
                             yerr=[avg - low, high - avg],
                             linestyle='',
                             elinewidth=0.8, capsize=1.5, ecolor='k')
            
        plt.legend(loc=7)
        plt.ylabel('Remuneration [€/EV]')
        plt.xlabel('Confidence level')
        plt.xticks(np.arange(0,3*6), np.tile(np.array(conf), 6))
        plt.axvline(8.5, color='k', linestyle='--')    
        plt.text(3*6*1/4,370, 'Evening Window',horizontalalignment='center', fontweight='bold')
        plt.text(3*6*3/4,370, 'Full-day Window',horizontalalignment='center', fontweight='bold')
        plt.grid('--', alpha=0.3)
        plt.xlim(-1,18)
    #    avg = res.Bids_Avg['v1g',s, a].values
    #    ax.bar(x+3*i + 9*j, avg, width=0.9, color=c[i], label=('_' if j else '') + 'V1G ' + sstr[i])
        f.suptitle('nfleet {}, thr_pen {}'.format(fl, tp))
        f.set_size_inches(8.5,4)
       
plt.legend(loc=(0.72,0.48))   
    
#%% Remuneration for diffs thresholds, Three plots, one figure
    
c = ['b', 'r', 'g']
x = np.arange(0,3)
a = '17_20'
fl=31
vxg = ['v1g', 'v2g']
#f = 71
#f=500
th = 0.6
p=0
alphas = [1,0.5]
#ths_ps = [{'t':t,'p':p} for t in [0.6,0.8] for p in [0,17.5,50]]
ths_ps = [{'t':0.6,'p':0},{'t':0.8,'p':17.5},{'t':0.8,'p':50}]
conf = [0.5, 0.9, 0.99]

vst = ['V1G', 'V2G']
f, ax = plt.subplots(3)
for l, tp in enumerate(ths_ps):
    for k, a in enumerate(av_w):    
        for i, s in enumerate(sets):
            for j, v in enumerate(vxg):
            #    for j, a in enumerate(av_w):
                avg = res.Payments_Avg.loc[v, a, s, fl, :, tp['t'],tp['p']].values
                low = res.Payments_perc_l.loc[v, a, s, fl, :, tp['t'],tp['p']].values
                high = res.Payments_perc_h.loc[v, a, s, fl, :,  tp['t'],tp['p']].values
                
                ax[l].bar(x+3*i+9*k, avg, width=0.9, color=c[i], alpha=alphas[j], 
                       label=('_' if k else '') + vst[j] + ' ' + sstr[i],
                       hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
                ax[l].errorbar(x+3*i+9*k, avg, 
                             yerr=[avg - low, high - avg],
                             linestyle='',
                             elinewidth=0.8, capsize=1.5, ecolor='k')
            
#        ax[l].legend(loc=7)
        ax[l].set_ylabel('th={}%; p={}%\nRemuneration [€/EV]'.format(int(tp['t']*100), int(tp['p']/50*100)), 
          fontweight='bold')
#        ax[l].set_xlabel('Confidence level')
        ax[l].set_xticks(np.arange(0,3*6))
        ax[l].set_xticklabels(np.tile(np.array(conf), 6))
        ax[l].axvline(8.5, color='k', linestyle='--')    
        ax[l].text(3*6*1/4,350, 'Evening Window',horizontalalignment='center', fontweight='bold')
        ax[l].text(3*6*3/4,350, 'Full-day Window',horizontalalignment='center', fontweight='bold')
        ax[l].grid('--', alpha=0.3)
        ax[l].set_xlim(-1,18)
    #    avg = res.Bids_Avg['v1g',s, a].values
    #    ax.bar(x+3*i + 9*j, avg, width=0.9, color=c[i], label=('_' if j else '') + 'V1G ' + sstr[i])
#        f.suptitle('nfleet {}, thr_pen {}'.format(fl, tp))
ax[0].legend(loc=7)        
#plt.legend(loc=(0.72,0.48))    
ax[l].set_xlabel('Confidence level')   
f.set_size_inches(8.5,   4*3)
    
#%% Remuneration with shaded area for diffs fleet sizes
c = ['royalblue', 'orangered', 'mediumseagreen','navy', 'firebrick', 'g']
ls=[['--',':','-.'], ['-','-','-']]
av_w  = ['17_20', '0_24']
av_wstr = ['Evening', 'Full-day']
NUM_COLORS = 6
cm = plt.get_cmap('Paired')
colors = [cm(1.*(i*2+1)/(NUM_COLORS*2)) for i in range(NUM_COLORS)]

alphas = [0.9,1]

folder_figs = r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\\'

idx = pd.IndexSlice

conf = 0.9
th = 0.8
p=17.5

mrks = [['','',''],['x','','o']]
mrksize = [4,0,3]
#ls = ['--',':','-.']

f, ax = plt.subplots(1,2)
for k, a in enumerate(av_w):
    for i, s in enumerate(sets):
        for j, v in enumerate(vxg):
            avg = res.Payments_Avg.loc[v, a, s, :, conf, th,p].values
            low = res.Payments_perc_l.loc[v, a, s, :, conf, th,p].values
            high = res.Payments_perc_h.loc[v, a, s, :, conf, th,p].values
            ax[k].plot(fleets, avg, linewidth=1.2, color=c[i+3*j], linestyle=ls[j][i], alpha=alphas[j],
                               label='V{}G {}'.format(v[1],sstr[i]),
                               marker=mrks[j][i], markersize=mrksize[i])
            ax[k].fill_between(fleets, low, high, 
                             alpha=0.08, color=c[i+3*j], edgecolor=None, label='_90% range')
#        v1g = stats_V1G.loc[idx[:,f,j,0.6,0], idx[:]]
#        plt.plot(x, v1g.Payments_Avg, linewidth=1.5, color=colors[i], linestyle='--',
#                           label='V1G at {} confidence'.format(j))
#        plt.fill_between(x, v1g.Payments_perc_l, v1g.Payments_perc_h, 
#                         alpha=0.1, color=colors[i], label='_90% range')
    ax[k].legend()
#    ax[k].set_title(a)
    ax[k].set_xlabel('Fleet size')
    ax[k].set_ylabel('Revenue [€/EV.y]')
    ax[k].set_xlim(0,fleets.max())
    ax[k].set_ylim(0,np.round(res.Payments_perc_h.loc[v,a,:,:,conf, th, p].max(),-1)+10)
    ax[k].text(x=500/2, y=(np.round(res.Payments_perc_h.loc[v,a,:,:,conf, th, p].max(),-1)+10)*0.95, 
      s='{} window'.format(av_wstr[k]), 
      horizontalalignment='center', fontweight='bold')
    ax[k].grid(linestyle='--', alpha=0.8)
    ax[k].axvline(31, linestyle='--', alpha=0.8, color='k')
    ax[k].text(x=35, y=[300, 42][k], s='30 EV fleet')
#f.suptitle(v + str(a)+ str(conf) + str(th) + str(p))
f.set_size_inches(10.5,   4.8)
ax[1].legend(loc=4)
#    plt.savefig(folder_figs + '{}_Rev_{}m_{}.png'.format(nameset,j,aw_s))

