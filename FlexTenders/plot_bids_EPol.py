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
import os

font = {'size':8}
plt.rc('font', **font)

#%% Load data
sets = ['Company', 'Commuter_HP', 'Commuter_LP']
sstr = [s.replace('_',' ') for s in sets]
folder = r'C:\Users\u546416\AnacondaProjects\EV-DSO\FlexTenders\EnergyPolicy\Paretto\\'
av_w  = ['0_24', '17_20']
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
pay = ['Payments_Avg', 'Payments_perc_l', 'Payments_perc_h']
und = ['UnderDel_Avg', 'UnderDel_perc_l', 'UnderDel_perc_h']

# read all files and and put them in one big DF
fs = [f for f in os.listdir(folder) if f.endswith('.csv')]

colsidx = ['VxG','Fleet', 'AvWindow', 'nevs', 'nactivation', 'service_duration', 'confidence',
               'penalty_threshold', 'penalties']
if 'full_data.csv' in fs:
    res = pd.read_csv(folder + 'full_data.csv', engine='python')
    if 'sensi_param' in res.columns:
        colsidx += ['sensi_param', 'sensi_val']
    res.set_index(colsidx, inplace=True)
else:
    res = pd.DataFrame()
    for f in fs:
        data = pd.read_csv(folder + f, engine='python')
        avw = '0_24' if '0_24' in f else '17_20'
        fleet = 'Company' if 'Company' in f else '' + 'Commuter_HP' if 'HP' in f else 'Commuter_LP' if 'LP' in f else 'Commuter_EN'
        data['AvWindow'] = avw
        data['Fleet'] = fleet
        res = pd.concat([res, data], ignore_index=True)
    if 'sensi_param' in res.columns:
        colsidx += ['sensi_param', 'sensi_val']
    res.set_index(colsidx, inplace=True, drop=True)
    res.to_csv(folder + 'full_data.csv')
#    
#%% Plot bids with diff service times with bars
colors = ['b', 'r', 'g']
ls=['-',':', '--']
av_w  = ['17_20', '0_24']
hatches = ['', "\\\\",'xx']
chat = ['none', 'maroon', 'k']
x = np.arange(0,3)
f, ax = plt.subplots()

nevs = 31.0
nac = 10
c = 0.9
p = 0
th = 0.6

for i, s in enumerate(sets):
    for j, a in enumerate(av_w):
        avg = res.Bids_Avg['v1g',s, a,nevs,nac,:,c,th,p].values
        ax.bar(x+3*i + 9*j, avg, width=0.9, facecolor=colors[i], 
               label=('_' if j else '') + 'V1G ' + sstr[i], 
               hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
        avg = res.Bids_Avg['v2g',s, a,nevs,nac,:,c,th,p].values
        ax.bar(x+3*i + 9*j, avg, width=0.9, facecolor=colors[i], alpha=0.5, 
               label=('_' if j else '') + 'V2G ' + sstr[i], 
               hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
#        print(s, a, 'v1g', hatches[i], chat[i])

plt.axvline(8.5, color='k', linestyle='--')
plt.legend(loc=7)
plt.ylabel('Bid [kW]')
plt.xlabel('Service duration [minutes]')
plt.xticks(np.arange(0,3*6), np.tile(np.array([30,60,120]), 6))    
plt.text(3*6*1/4,6.8, 'Evening Window',horizontalalignment='center', fontweight='bold')
plt.text(3*6*3/4,6.8, 'Full-day Window',horizontalalignment='center', fontweight='bold')
plt.grid('--', alpha=0.3)
plt.xlim(-1,18)
f.set_size_inches(7,3.3)
plt.tight_layout()

plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\bids_serv_dur_v2.pdf')
plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\bids_serv_dur_v2.png', dpi=300)
#f.suptitle(fleet)


#%% Plot Bids @ various conf, V2G & V1G
c = ['b', 'r', 'g']
x = np.arange(0,3)
#a = '17_20'
fl=31.0
vxg = ['v1g', 'v2g']
#f = 71
#f=500
sd = 30
th = 0.6
p=0
alphas = [1,0.5]
vst = ['V1G', 'V2G']
nac = 10
conf = [0.5, 0.9, 0.99]

hatches = ['', "\\\\",'xx']
chat = ['none', 'maroon', 'k']

f, ax = plt.subplots()
for k, a in enumerate(av_w):    
    for i, s in enumerate(sets):
        for j, v in enumerate(vxg):
        #    for j, a in enumerate(av_w):
            avg = res.Bids_Avg.loc[v, s, a, fl, nac, sd, :, th,p].values
            low = res.Bids_perc_l.loc[v, s, a, fl, nac, sd, :, th,p].values
            high = res.Bids_perc_h.loc[v, s, a, fl, nac, sd, :, th,p].values
            
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
    
##%% Remuneration for diffs thresholds, independent plots/figures
#    
#c = ['b', 'r', 'g']
#x = np.arange(0,3)
#a = '17_20'
#fl=98
#vxg = ['v1g', 'v2g']
##f = 71
##f=500
#th = 0.6
#p=0
#alphas = [1,0.5]
##ths_ps = [{'t':t,'p':p} for t in [0.6,0.8] for p in [0,17.5,50]]
#ths_ps = [{'t':0.6,'p':0},{'t':0.8,'p':17.5},{'t':0.9,'p':35}]
#conf = [0.5, 0.9, 0.99]
#
#vst = ['V1G', 'V2G']
#for tp in ths_ps:
#    f, ax = plt.subplots()
#    for k, a in enumerate(av_w):    
#        for i, s in enumerate(sets):
#            for j, v in enumerate(vxg):
#            #    for j, a in enumerate(av_w):
#                avg = res.Payments_Avg.loc[v, s, a, fl, :, tp['t'],tp['p']].values
#                low = res.Payments_perc_l.loc[v, s, a, fl, :, tp['t'],tp['p']].values
#                high = res.Payments_perc_h.loc[v, s, a, fl, :,  tp['t'],tp['p']].values
#                
#                ax.bar(x+3*i+9*k, avg, width=0.9, color=c[i], alpha=alphas[j], 
#                       label=('_' if k else '') + vst[j] + ' ' + sstr[i], 
#                       hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
#                ax.errorbar(x+3*i+9*k, avg, 
#                             yerr=[avg - low, high - avg],
#                             linestyle='',
#                             elinewidth=0.8, capsize=1.5, ecolor='k')
#            
#        plt.legend(loc=7)
#        plt.ylabel('Remuneration [€/EV]')
#        plt.xlabel('Confidence level')
#        plt.xticks(np.arange(0,3*6), np.tile(np.array(conf), 6))
#        plt.axvline(8.5, color='k', linestyle='--')    
#        plt.text(3*6*1/4,370, 'Evening Window',horizontalalignment='center', fontweight='bold')
#        plt.text(3*6*3/4,370, 'Full-day Window',horizontalalignment='center', fontweight='bold')
#        plt.grid('--', alpha=0.3)
#        plt.xlim(-1,18)
#    #    avg = res.Bids_Avg['v1g',s, a].values
#    #    ax.bar(x+3*i + 9*j, avg, width=0.9, color=c[i], label=('_' if j else '') + 'V1G ' + sstr[i])
#        f.suptitle('nfleet {}, thr_pen {}'.format(fl, tp))
#        f.set_size_inches(8.5,4)
#       
#plt.legend(loc=(0.72,0.48))   
    
#%% Remuneration for diffs penalty thresholds, Three plots, one figure
    
c = ['b', 'r', 'g']
hatches = ['', "\\",'xx']

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
ths_ps = [{'t':0.6,'p':0},{'t':0.8,'p':17.5},{'t':0.9,'p':35}]
conf = [0.5, 0.9, 0.99]
nac = 10

vst = ['V1G', 'V2G']
f, ax = plt.subplots(3)
for l, tp in enumerate(ths_ps):
    for k, a in enumerate(av_w):    
        for i, s in enumerate(sets):
            for j, v in enumerate(vxg):
            #    for j, a in enumerate(av_w):
                avg = res.Payments_Avg.loc[v, s, a, fl, nac, sd, :, tp['t'],tp['p']].values
                low = res.Payments_perc_l.loc[v, s, a, fl, nac, sd, :, tp['t'],tp['p']].values
                high = res.Payments_perc_h.loc[v, s, a, fl, nac, sd, :, tp['t'],tp['p']].values
                
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
    

#%% Remuneration for diffs penalty cases, one conf level
# old bar plot

c = ['b', 'r', 'g']
hatches = ['', "\\",'xx']

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
ths_ps = [{'t':0.6,'p':0},{'t':0.8,'p':17.5},{'t':0.9,'p':35}]
conf = 0.9 #[0.5, 0.9, 0.99]
nac = 10

vst = ['V1G', 'V2G']
f, ax = plt.subplots()
for l, tp in enumerate(ths_ps):
    for k, a in enumerate(av_w):    
        for i, s in enumerate(sets):
            for j, v in enumerate(vxg):
            #    for j, a in enumerate(av_w):
                avg = res.Payments_Avg.loc[v, s, a, fl, nac, sd, :, tp['t'],tp['p']].values
                low = res.Payments_perc_l.loc[v, s, a, fl, nac, sd, :, tp['t'],tp['p']].values
                high = res.Payments_perc_h.loc[v, s, a, fl, nac, sd, :, tp['t'],tp['p']].values
                
                ax[l].bar(x+3*i+9*k, avg, width=0.9, color=c[i], alpha=alphas[j], 
                       label=('_' if k else '') + vst[j] + ' ' + sstr[i],
                       hatch=hatches[i], edgecolor=np.repeat(chat[i],3))
                ax[l].errorbar(x+3*i+9*k, avg, 
                             yerr=[avg - low, high - avg],
                             linestyle='',
                             elinewidth=0.8, capsize=1.5, ecolor='k')
            
#        ax[l].legend(loc=7)
#        ax[l].set_ylabel('th={}%; p={}%\nRemuneration [€/EV]'.format(int(tp['t']*100), int(tp['p']/50*100)), 
#          fontweight='bold')
                
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



#%% Remuneration using box plot

folder_r = folder + r'raw\\'

c = ['b', 'r', 'g']
hatches = ['', "\\",'xx']

x = np.arange(0,3)
a = '[17, 20]'
fl=31
vxg = ['v1g', 'v2g']
serv_t = 30
#f = 71
#f=500
th = 0.6
p=0
alphas = [1,0.5]
#ths_ps = [{'t':t,'p':p} for t in [0.6,0.8] for p in [0,17.5,50]]
ths_ps = [{'t':0.6,'p':0},{'t':0.8,'p':17.5},{'t':0.9,'p':35}]
conf = [0.5, 0.9, 0.99]
nscenarios=500
nac = 10

av_wi = ['[17, 20]', '[0, 24]']

vst = ['V1G', 'V2G']
f, ax = plt.subplots(3)
for l, tp in enumerate(ths_ps):
    for k, a in enumerate(av_wi):    
        for i, s in enumerate(sets):
            v1gs = []
            v2gs = []
#            print(tp, s, a)
            for m, cf in enumerate(conf):
                file = 'fleet' + s + '_avw' + a + '_ev'+ str(fl) + '.0_nact' + str(int(nac))  + '_servt' + str(serv_t) + '_confth' + str(cf) + '_penaltyth' + str(tp['t']) + '.npy'
                v1gbids, v1gpay, v1gund, v2gbids, v2gpay, v2gund = np.load(folder_r + file)
                v1gs.append((v1gpay - v1gbids.repeat(nscenarios) * v1gund * tp['p'])/fl)
                v2gs.append((v2gpay - v2gbids.repeat(nscenarios) * v2gund * tp['p'])/fl)
#                print(cf, np.mean((v2gpay - v2gbids.repeat(nscenarios) * v2gund * tp['p'])/fl))    
            bplot = ax[l].boxplot(v1gs, positions=np.arange(0,3) + 3*i + k*9, patch_artist=True,
                      widths=0.9, labels=conf)
            for patch in bplot['boxes']:
                patch.set_facecolor(c[i])
                patch.set_hatch(hatches[i])
                patch.set_edgecolor(c[i])
            bplot = ax[l].boxplot(v2gs, positions=np.arange(0,3) + 3*i + k*9, patch_artist=True,
                      widths=0.9, labels=conf)
            for patch in bplot['boxes']:
                patch.set_facecolor(c[i])
                patch.set_edgecolor(c[i])
                patch.set_alpha(0.5)
                patch.set_hatch(hatches[i])
    ax[l].set_xlim(-1,18)
    ax[l].set_xticks(np.arange(0,3*6))
    ax[l].set_xticklabels(np.tile(np.array(conf), 6))
    ax[l].axvline(8.5, color='k', linestyle='--')    
    ax[l].text(3*6*1/4,350, 'Evening Window', horizontalalignment='center', fontweight='bold')
    ax[l].text(3*6*3/4,350, 'Full-day Window',horizontalalignment='center', fontweight='bold')
    ax[l].grid('--', alpha=0.3)
    ax[l].set_ylabel('th={}%; p={}%\nRemuneration [€/EV]'.format(int(tp['t']*100), int(tp['p']/50*100)), 
          fontweight='bold')
        
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

#folder_figs = r'c:userU546416PicturesFlexTendersEnergyPolicy\\'

idx = pd.IndexSlice

nac=10
sd=30
conf = 0.9
th = 0.6
p=0

mrks = [['','',''],['x','','o']]
mrksize = [4,0,3]
fleets = res.index.levels[3]
vxg = ['v1g', 'v2g']
#ls = ['--',':','-.']

f, ax = plt.subplots(1,2)
for k, a in enumerate(av_w):
    for i, s in enumerate(sets):
        for j, v in enumerate(vxg):
            avg = res.Payments_Avg.loc[v, s, a, :, nac, sd, conf, th,p].values
            low = res.Payments_perc_l.loc[v, s, a, :, nac, sd, conf, th,p].values
            high = res.Payments_perc_h.loc[v, s, a, :, nac, sd, conf, th,p].values
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
    ax[k].set_ylim(0,np.round(res.Payments_perc_h.loc[v,:,a,:,nac,sd,conf,th,p].max(),-1)+10)
    ax[k].text(x=500/2, y=(np.round(res.Payments_perc_h.loc[v,:,a,:,nac,sd,conf,th,p].max(),-1)+10)*0.95, 
      s='{} window'.format(av_wstr[k]), 
      horizontalalignment='center', fontweight='bold')
    ax[k].grid(linestyle='--', alpha=0.8)
    ax[k].axvline(31, linestyle='--', alpha=0.8, color='k')
    ax[k].text(x=35, y=[300, 42][k], s='30 EV fleet')
#f.suptitle(v + str(a)+ str(conf) + str(th) + str(p))
f.set_size_inches(10.5,   4.8)
ax[1].legend(loc=4)
#    plt.savefig(folder_figs + '{}_Rev_{}m_{}.png'.format(nameset,j,aw_s))

#%% THESIS: Remuneration with shaded area for diffs fleet sizes 
c = ['navy', 'firebrick', 'g','royalblue', 'orangered', 'mediumseagreen',]
ls=[ ['-','-','-'],['--',':','-.']]
av_w  = ['17_20', '0_24']
av_wstr = ['Evening', 'Full-day']
NUM_COLORS = 6
cm = plt.get_cmap('Paired')
colors = [cm(1.*(i*2+1)/(NUM_COLORS*2)) for i in range(NUM_COLORS)]

alphas = [0.9,1]

#folder_figs = r'c:userU546416PicturesFlexTendersEnergyPolicy\\'

idx = pd.IndexSlice

nac=10
sd=30
conf = 0.9
th = 0.6
p=0

mrks = [['','',''],['x','','o']]
mrks = [['o','o','o'],['','','']]
mrksize = [2,2,2]
fleets = res.index.levels[3]
vxg = ['v1g', 'v2g']
#ls = ['--',':','-.']

f, axs = plt.subplots(1,2)
for k, a in enumerate(av_w):
    ax = axs[k]
    for i, s in enumerate(sets):
        for j, v in enumerate(vxg):
            avg = res.Payments_Avg.loc[v, s, a, :, nac, sd, conf, th,p].values
            low = res.Payments_perc_l.loc[v, s, a, :, nac, sd, conf, th,p].values
            high = res.Payments_perc_h.loc[v, s, a, :, nac, sd, conf, th,p].values
            ax.plot(fleets, avg, linewidth=1.2, color=c[i+3*j], linestyle=ls[j][i], alpha=alphas[j],
                               label='V{}G {}'.format(v[1],sstr[i]),
                               marker=mrks[j][i], markersize=mrksize[i])
            ax.fill_between(fleets, low, high, 
                             alpha=0.1, color=c[i+3*j], edgecolor=None, label='_90% range')
#        v1g = stats_V1G.loc[idx[:,f,j,0.6,0], idx[:]]
#        plt.plot(x, v1g.Payments_Avg, linewidth=1.5, color=colors[i], linestyle='--',
#                           label='V1G at {} confidence'.format(j))
#        plt.fill_between(x, v1g.Payments_perc_l, v1g.Payments_perc_h, 
#                         alpha=0.1, color=colors[i], label='_90% range')
#    ax[k].legend()
#    ax[k].set_title(a)
    ax.set_xlabel('Fleet size')
    ax.set_ylabel('Revenue [€/EV.y]')
    ax.set_xlim(0,fleets.max())
    ax.set_ylim(0,np.round(res.Payments_perc_h.loc[v,:,a,:,nac,sd,conf,th,p].max(),-1)+10)
#    ax[k].text(x=500/2, y=(np.round(res.Payments_perc_h.loc[v,:,a,:,nac,sd,conf,th,p].max(),-1)+10)*0.95, 
#      s='{} window'.format(av_wstr[k]), 
#      horizontalalignment='center', fontweight='bold')
    ax.set_title('({}) {} window'.format('abc'[k], av_wstr[k]), y=-0.25)
    ax.grid(linestyle='--', alpha=0.8)
    ax.axvline(31, linestyle='--', alpha=0.8, color='k')
    ax.text(x=35, y=[300, 42][k], s='30 EV fleet')
#f.suptitle(v + str(a)+ str(conf) + str(th) + str(p))
f.set_size_inches(7,3.5)
f.legend(loc=9, ncol=3)
f.tight_layout()
f.tight_layout()
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.09
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])
#    plt.savefig(folder_figs + '{}_Rev_{}m_{}.png'.format(nameset,j,aw_s))

plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Remuneration_fleetsizes.png', dpi=300)
plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Remuneration_fleetsizes.pdf')

#%% Print tables:
nevs=31.0
nac=10
sd=30
conf=0.9
th=0.6
p=0
prices = [12.5,50,200]
vsg = ['v1g','v2g']
av_w  = ['17_20', '0_24']
remu = [np.array([res.Payments_Avg.loc[v, s, a, nevs, nac,sd,conf,th][p] for s in sets for a in av_w]) * price/50
        for v in vsg for price in prices]
idx = pd.MultiIndex.from_arrays([np.repeat(['V1G', 'V2G'],3), np.tile(prices,2)], names=['VxG', 'Prices'])
cols = pd.MultiIndex.from_arrays([np.repeat(sets, 2), np.tile(av_w,3)], names=['Sets', 'AvWindow'])
remu = pd.DataFrame(remu, index=idx, columns=cols).round(decimals=1)
#remu.to_csv(folder+ 'remuneration_table.csv')


# baselines
nevs=31.0
bds = ['Bids_perc_l', 'Bids_Avg', 'Bids_perc_h']
bls = [np.concatenate([res[bds].loc['v1g', s, a, nevs, nac,sd,conf,th].loc[p].values for a in av_w])
        for s in sets]
idx = sets
cols = pd.MultiIndex.from_arrays([np.repeat(av_w, 3), np.tile(['Min', 'Avg', 'Max'],2)], names=['AvWindow', 'Stat'])
bls = pd.DataFrame(bls, index=idx, columns=cols).round(decimals=2)
#bls.to_csv(folder+ 'baselines_table.csv')


#%% Plot paretto for one penalty condition
c = ['royalblue', 'orangered', 'mediumseagreen','navy', 'firebrick', 'g']
ls=[['--',':','-.'], ['-','-','-']]
mrkrs = ['x','o']
nevs = 31.0
nac = 10
sd = 30
prices = 50
p = 0
th = 0.6
vsg = ['v1g', 'v2g']
av_w  = ['17_20', '0_24']

for a in av_w:
    f,ax = plt.subplots()
    for j, v in enumerate(vsg):
        for i, s in enumerate(sets):
            avgr = res.Payments_Avg.loc[v, s, a, nevs, nac, sd, :, th, p]
            cvar = res.Payments_CVaR.loc[v, s, a, nevs, nac, sd, :, th, p]
            ax.plot(cvar, avgr, color=c[i], linestyle='', marker=mrkrs[j], fillstyle='none', label='_'.format(v[1],s))
            parettopoints = [i for i in avgr.index 
                                if not ((cvar[i]<=cvar.drop(i)) & (avgr[i]<=avgr.drop(i))).any()]
            if len(parettopoints):
                ax.plot(cvar[parettopoints], avgr[parettopoints], color=c[i], 
                        linestyle=ls[j][i], marker=mrkrs[j], 
                        fillstyle='none', 
                        label='V{}G {}'.format(v[1],s))
                maxi = (cvar[parettopoints] + avgr[parettopoints]).idxmax()
                plt.text(cvar[maxi], avgr[maxi], maxi)
            else:
                ax.plot(0,0, color=c[i], 
                        linestyle=ls[j][i], marker=mrkrs[i], 
                        fillstyle='none', 
                        label='V{}G {}'.format(v[1],s.replace('_',' ')))

            
#            for ii in range(3):
#                cl = [0.5,0.9,0.99]
#                idx= [8,17,19]
#                plt.text(cvar[idx[ii]]+5,avgr[idx[ii]]+5, str(cl[ii]))
#            plt.text(cvar[-1], avgr[-1], '100% confidence')
            plt.title('Paretto frontier, {} window'.format('Evening' if a=='17_20' else 'Fullday'))
            plt.legend()
            plt.xlabel('C-VaR [€/EV/y]')
            plt.ylabel('Average revenue [€/EV/y]')

#%% Plot paretto for three penalty conditions
c = ['royalblue', 'orangered', 'mediumseagreen','navy', 'firebrick', 'g']
ls=[[':',':',':'], ['-','-','-']]
mrkrs = ['x','o', '*']
nevs = 31.0
nac = 10
sd = 30
prices = 50
p = [0,17.5,35] 
ths = [0.6, 0.8, 0.9]
vsg = ['v1g', 'v2g']
av_w  = ['17_20', '0_24']

#av_w  = ['0_24']
#vsg=['v2g']



texts = []
for a in av_w:
    for j, v in enumerate(vsg):
        for i, s in enumerate(sets):
            f,ax = plt.subplots()
            for k, th in enumerate(ths):
                avgr = res.Payments_Avg.loc[v, s, a, nevs, nac, sd, :, th, p[k]]
                cvar = res.Payments_CVaR.loc[v, s, a, nevs, nac, sd, :, th, p[k]]
                ax.plot(cvar, avgr, color=c[k], linestyle='', marker=mrkrs[k], fillstyle='none', label='_'.format(v[1],s))
                parettopoints = [i for i in avgr.sort_values().index 
                                    if not ((cvar[i]<=cvar.drop(i)) & (avgr[i]<=avgr.drop(i))).any()]
                if len(parettopoints):
                    ax.plot(cvar[parettopoints], avgr[parettopoints], color='k', 
                            markeredgecolor=c[k], 
                            linestyle=ls[0][k], marker=mrkrs[k], 
                            fillstyle='none', 
                            label='p={}%, th={}%'.format(int(p[k]/50*100), int(th*100)))
                    maxi = (cvar[parettopoints] + avgr[parettopoints]).idxmax()
                    plt.text(cvar[maxi]+1, avgr[maxi]+1, maxi)
                else:
                    ax.plot(0,0, color='k', 
                            markeredgecolor=c[k], 
                            linestyle=ls[0][k], marker=mrkrs[k], 
                            fillstyle='none', 
                            label='p={}%, th={}%'.format(int(p[k]/50*100), int(th*100)))
    #            plt.text(cvar[-1], avgr[-1], '100% confidence')
                plt.title('Paretto frontier, {} window\nV{}G {} fleet'.format('Evening' if a=='17_20' else 'Fullday',
                          v[1], s.replace('_',' ')))
                plt.legend()
                plt.xlabel('C-VaR [€/EV/y]')
                plt.ylabel('Average revenue [€/EV/y]')
                plt.axhline(0,linestyle='--', color='grey', alpha=0.5)
                plt.axvline(0,linestyle='--', color='grey', alpha=0.5)
            texts.append(ax.texts)
                
def modify_xy(t, dx=0, dy=0):
    x,y=t.get_position()
    t.set_position((x+dx,y+dy))
    
#%% Plot paretto THESIS, One fig, 3plots per Av Window
c = ['royalblue', 'orangered', 'mediumseagreen','navy', 'firebrick', 'g']
labels = ['Low penalties', 'Medium penalties', 'High penalties']
ls=[[':',':',':'], ['-','-','-']]
mrkrs = ['x','o', '*']
nevs = 31.0
nac = 10
sd = 30
prices = 50
p = [0,17.5,35] 
ths = [0.6, 0.8, 0.9]
#vsg = ['v1g', 'v2g']
av_w  = ['17_20', '0_24']

#av_w  = ['0_24']
vsg= 'v2g'
abc = 'abcdef'

avw = ['Evening window', 'Full-day window' ]
texts = []
fs = []

for j, a in enumerate(av_w):
    f, axs = plt.subplots(1,3)
    fs.append(f)
    for i, s in enumerate(sets):
        ax = axs[i]
        for k, th in enumerate(ths):
            avgr = res.Payments_Avg.loc[vsg, s, a, nevs, nac, sd, :, th, p[k]]
            cvar = res.Payments_CVaR.loc[vsg, s, a, nevs, nac, sd, :, th, p[k]]
            ax.plot(cvar, avgr, color=c[k], linestyle='', marker=mrkrs[k], fillstyle='none',label='_')
            parettopoints = [i for i in avgr.sort_values().index 
                                if not ((cvar[i]<=cvar.drop(i)) & (avgr[i]<=avgr.drop(i))).any()]
            label = labels[k] if i==0 else '_'
            if len(parettopoints):
                ax.plot(cvar[parettopoints], avgr[parettopoints], color='k', 
                        markeredgecolor=c[k], 
                        linestyle=ls[0][k], marker=mrkrs[k], 
                        fillstyle='none', 
                        label=label)
                maxi = (cvar[parettopoints] + avgr[parettopoints]).idxmax()
                ax.text(cvar[maxi]+1, avgr[maxi]+1, maxi)
            else:
                ax.plot(0,0, color='k', 
                        markeredgecolor=c[k], 
                        linestyle=ls[0][k], marker=mrkrs[k], 
                        fillstyle='none', 
                        label=label)
#            plt.text(cvar[-1], avgr[-1], '100% confidence')
            ax.set_title('({}) {}\n{}'.format(abc[i + j*3], avw[j], s.replace('_',' ')), y=-0.35)
#            ax.legend()
            ax.set_xlabel('C-VaR [€/EV/y]')
            if i == 0:
                ax.set_ylabel('Average revenue [€/EV/y]')
            ax.axhline(0,linestyle='--', color='grey', alpha=0.5)
            ax.axvline(0,linestyle='--', color='grey', alpha=0.5)
        texts.append(ax.texts)
                
    if j==0:
        f.set_size_inches(7,3.5)
        f.legend(loc=9, ncol=3)
        # resizing axs to leave space for legend
        f.tight_layout()
        f.tight_layout()
        for i, ax in enumerate(axs):
            pos = ax.get_position()
            dy = 0.06
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])
    else:
        f.set_size_inches(7,3.2)
        f.tight_layout()
        f.tight_layout()
        

    
    
def modify_xy(t, dx=0, dy=0, x=None, y=None):
    if x is None:
        x,y=t.get_position()
    t.set_position((x+dx,y+dy))
    
def correct_limits(f, lims):
    for i, l in enumerate(lims):
        if l =='no':
            continue
        f.axes[i].set_xlim(l[0],l[1])
        f.axes[i].set_ylim(l[2],l[3])
        
def correct_texts(texts, xys):
    for i, xyss in enumerate(xys):
        for j, xy in enumerate(xyss):
            modify_xy(texts[i][j], x=xy[0], y=xy[1])
        
lims_ew = ['no', (-120,120, -30,150), (-70,70,-30,80)]
lims_fd = ['no', (-100,50, -20,150), (-100,50,-20,60)]

xy_ew = [[(285,355), (500000,555000), (50000,50000)],
         [(85.1414,120.851), (63.7458,89.8515), (48.2991,65.2688)],
         [(36.3654,58.7373),(28.6439,41.248),(22.8416,31.9706)]]
xy_fd = [[(134.348,249.681), (51.7873,201.294),(-12.3107,73.9605)],
         [(35.7703,107.337), (16.4371,50.0197),(10.0364,15.3659)],
         [(20.3781,45.474), (13.866,25.5527), (4.58399,17.7584)]]    

correct_limits(fs[0], lims_ew)
correct_limits(fs[1], lims_fd)
correct_texts(texts, xy_ew + xy_fd)

ax=fs[0].axes[0]
plt.sca(ax)
plt.arrow(20, 10, 300, 0, width=1, head_width=8, color='olive')
plt.arrow(10, 20, 0, 300, width=1, head_width=8, color='olive')
plt.text(s='Low risk', x=20+300/2, y=20, horizontalalignment='center')
plt.text(s='High remuneration', x=20, y=20+150, va='center', rotation=90)

fs[0].savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\Evening_v3.png', dpi=300)
fs[0].savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\Evening_v3.pdf')
fs[1].savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\FullDay_v3.png', dpi=300)
fs[1].savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\FullDay_v3.pdf')



#%% Plot bid for diff confs levels THESIS, One fig, 3plots per Av Window
c = ['royalblue', 'orangered', 'mediumseagreen','navy', 'firebrick', 'g']
labels = ['Low penalties', 'Medium penalties', 'High penalties']
ls=[[':',':',':'], ['-','-','-']]
mrkrs = ['x','o', '*']
nevs = 31.0
nac = 10
sd = 30
prices = 50
p = [0,17.5,35] 
ths = [0.6, 0.8, 0.9]
vsg = ['v1g', 'v2g']
av_w  = ['17_20', '0_24']

#av_w  = ['0_24']
#vsg= 'v2g'
abc = 'abc'
th, p = 0.6, 35

texts = []
fs = []
labels = ['Company', 'Commuter HP', 'Commuter LP']

colors = ['royalblue', 'orangered', 'mediumseagreen','royalblue', 'orangered', 'mediumseagreen']
colorsm = [['k' for i in range(6)],
        ['royalblue', 'orangered', 'mediumseagreen','royalblue', 'orangered', 'mediumseagreen']]

confsint = [(0.05, 0.5) for i in range(6)]


f, ax = plt.subplots()
for j, v in enumerate(vsg):
    bids = []
    meds = []
    for k, a in enumerate(av_w):
        for i, s in enumerate(sets):
            bid = res.Bids_Avg.loc[v, s, a, nevs, nac, sd, :, th, p]
            bids = bids + [bid]
            meds.append(bid[0.9])
#            ax.plot([i + k*3 for b in bid], bid, color=c[i], linestyle='', marker=mrkrs[j], fillstyle='none',label='_')
    bplots = ax.boxplot(bids, patch_artist=True,
               labels=labels + labels,
               usermedians=meds,
               conf_intervals=confsint)
    # changing colors to bplots
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    # changing colors to medians
    for patch, color in zip(bplots['medians'], colorsm[j]):
        patch.set_color(color)
        patch.set_linewidth(2)
    
#            plt.text(cvar[-1], avgr[-1], '100% confidence')
#        ax.set_title('({}) {}'.format(abc[i], s.replace('_',' ')), y=-0.26)
#            ax.legend()
#        ax.set_xlabel('C-VaR [€/EV/y]')
ax.set_ylabel('Average Bid [kW/EV]')
            
ax.axvline(3.5, color='k', linestyle='--')    
ax.text(2,7, 'Evening Window', horizontalalignment='center', fontweight='bold')
ax.text(5,7, 'Full-day Window',horizontalalignment='center', fontweight='bold')
    
f.set_size_inches(9,4)
#f.legend(loc=9, ncol=3)
f.tight_layout()
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])
    
#%% Plot bid for diff confs levels THESIS, One fig, 3plots per Av Window - BOXPLOT
c = ['navy', 'firebrick', 'g','royalblue', 'orangered', 'mediumseagreen']
labels = ['Low penalties', 'Medium penalties', 'High penalties']
ls=[[':',':',':'], ['-','-','-']]
mrkrs = ['x','o', '*']
nevs = 31.0
nac = 10
sd = 30
prices = 50
p = [0,17.5,35] 
ths = [0.6, 0.8, 0.9]
vsg = ['v2g']
av_w  = ['17_20', '0_24']

#av_w  = ['0_24']
#vsg= 'v2g'
abc = 'abc'
th, p = 0.6, 35

texts = []
fs = []
labels = ['Company', 'Commuter HP', 'Commuter LP']

#colors = ['royalblue', 'orangered', 'mediumseagreen','royalblue', 'orangered', 'mediumseagreen']
#colorsm = [['k' for i in range(6)],
#        ['royalblue', 'orangered', 'mediumseagreen','royalblue', 'orangered', 'mediumseagreen']]
ls = ['-','--']
alpha = [1,0.8]
mrks = ['o','']
confsint = [(0.05, 0.5) for i in range(6)]


f, axs = plt.subplots(1,2, sharey=True)
for j, v in enumerate(vsg):
#    bids = []
#    meds = []
    for k, a in enumerate(av_w):
        ax = axs[k]
        for i, s in enumerate(sets):
            bid = res.Bids_Avg.loc[v, s, a, nevs, nac, sd, :, th, p]
            if k==0:
                label= 'V{}G'.format(j+1+1) + ' ' + labels[i]
            else:
                label='_'
            ax.plot(bid.index, bid, color=c[i+j*3], linestyle=ls[j],alpha=alpha[j], 
                    marker=mrks[j], markersize=2, label=label) #marker=mrkrs[i], fillstyle='none',label='_')
        ax.set_xlabel('Confidence level')
        ax.set_xticks([i/10 for i in range(0,11,2)])
        ax.set_ylim(0,7.5)
        ax.grid(alpha=0.5)
#            plt.text(cvar[-1], avgr[-1], '100% confidence')
#        ax.set_title('({}) {}'.format(abc[i], s.replace('_',' ')), y=-0.26)
#            ax.legend()
#        ax.set_xlabel('C-VaR [€/EV/y]')

axs[0].set_ylabel('Bid [kW/EV]')
axs[0].set_title('(a) Evening window', y=-0.3)
axs[1].set_title('(b) Full-day window', y=-0.3)

axs[0].axvline(0.9, color='k', linestyle=':',alpha=0.8,label='_')    
axs[1].axvline(0.9, color='k', linestyle=':',alpha=0.8,label='_')    

#ax.text(2,7, 'Evening Window', horizontalalignment='center', fontweight='bold')
#ax.text(5,7, 'Full-day Window',horizontalalignment='center', fontweight='bold')
hs, ls = axs[0].get_legend_handles_labels()
order = [0,1,2,]#[0,3,1,4,2,5]
hs = [hs[o] for o in order]
ls = [ls[o] for o in order]
f.set_size_inches(7,3.5)
f.legend(hs, ls, loc=9, ncol=3) #loc=9
f.tight_layout()
# resizing axs to leave space for legend
f.tight_layout()
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.1 #0.1
    dx = 0
    ax.set_position([pos.x0-dx*i, pos.y0, pos.width-dx, pos.height-dy])
    

#f.savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\bids_conflvl.png', dpi=300)
#f.savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\bids_conflvl.pdf')
#fs[1].savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\FullDayW.png')
#fs[1].savefig(r'c:\user\U546416\Pictures\FlexTenders\EnergyPolicy\Paretto\FullDayW.pdf')

#%% Plot bid for diff confs levels THESIS, One fig, 3plots per Av Window
c = ['navy', 'firebrick', 'g','royalblue', 'orangered', 'mediumseagreen']
labels = ['Low penalties', 'Medium penalties', 'High penalties']
ls=[[':',':',':'], ['-','-','-']]
mrkrs = ['x','o', '*']
nevs = 31.0
nac = 10
sd = 30
prices = 50
p = [0,17.5,35] 
ths = [0.6, 0.8, 0.9]
vsg = ['v2g']
av_w  = ['17_20', '0_24']

#av_w  = ['0_24']
#vsg= 'v2g'
abc = 'abc'
th, p = 0.6, 35

texts = []
fs = []
labels = ['Company', 'Commuter HP']
sets = ['Company', 'Commuter_HP']

#colors = ['royalblue', 'orangered', 'mediumseagreen','royalblue', 'orangered', 'mediumseagreen']
#colorsm = [['k' for i in range(6)],
#        ['royalblue', 'orangered', 'mediumseagreen','royalblue', 'orangered', 'mediumseagreen']]
ls = ['-','--']
alpha = [1,0.8]
mrks = ['o','']
confsint = [(0.05, 0.5) for i in range(6)]


for j, v in enumerate(vsg):
#    bids = []
#    meds = []
    for k, a in enumerate(av_w):
#        ax = axs[k]
        f, ax = plt.subplots(1, sharey=True)
        for i, s in enumerate(sets):
            bid = res.Bids_Avg.loc[v, s, a, nevs, nac, sd, :, th, p]
#            if k==0:
            label= labels[i]
  #          else:
 #               label='_'
            ax.plot(bid.index, bid, color=c[i+j*3], linestyle=ls[j],alpha=alpha[j], 
                    marker=mrks[j], markersize=2, label=label) #marker=mrkrs[i], fillstyle='none',label='_')
        ax.set_xlabel('Aggregator\'s Confidence level')
        ax.set_xticks([i/10 for i in range(0,11,2)])
        ax.set_ylim(0,7.5)
        ax.grid(alpha=0.5)
#            plt.text(cvar[-1], avgr[-1], '100% confidence')
#        ax.set_title('({}) {}'.format(abc[i], s.replace('_',' ')), y=-0.26)
#            ax.legend()
#        ax.set_xlabel('C-VaR [€/EV/y]')

        ax.set_ylabel('Bid [kW/EV]')
        ax.legend(loc=3)
#axs[0].set_title('(a) Evening window', y=-0.3)
#axs[1].set_title('(b) Full-day window', y=-0.3)


#ax.text(2,7, 'Evening Window', horizontalalignment='center', fontweight='bold')
#ax.text(5,7, 'Full-day Window',horizontalalignment='center', fontweight='bold')
#hs, ls = axs[0].get_legend_handles_labels()
#order = [0,3,1,4,2,5]
#hs = [hs[o] for o in order]
#ls = [ls[o] for o in order]
#f.set_size_inches(9,4)
#f.legend(hs, ls, loc=9, ncol=3) #loc=9
#f.tight_layout()
# resizing axs to leave space for legend    
        f.set_size_inches(3.9,2.1)
        f.tight_layout()
#for i, ax in enumerate(axs):
#    pos = ax.get_position()
#    dy = 0.1 #0.1
#    dx = 0
#    ax.set_position([pos.x0-dx*i, pos.y0, pos.width-dx, pos.height-dy])
#    

        f.savefig(r'c:\user\U546416\Pictures\FlexTenders\IAEE\bids_conflvl_IAEE_{}.png'.format(a), dpi=300)
        f.savefig(r'c:\user\U546416\Pictures\FlexTenders\IAEE\bids_conflvl_IAEE_{}.pdf'.format(a))

#%% Print tables of opt flex:
optflex = {}

sets = ['Company', 'Commuter_HP', 'Commuter_LP']
vxg = ['v1g', 'v2g']
av_w = ['17_20','0_24']
ps = [(0.6,0, '0low'), (0.8,17.5, '1med'), (0.9,35, '2high')]

nevs = 31
nact = 10
sd = 30


for s in sets:
    for v in vxg:
        for avw in av_w:
            for (t,p,pl) in ps:
                data = res.xs(level=['VxG', 'Fleet', 'AvWindow', 'nevs', 
                               'nactivation', 'service_duration', 
                               'penalty_threshold', 'penalties'], 
                                key=[v,s,avw,nevs,nact,sd,t,p])
                idx = (data.Payments_Avg + data.Payments_CVaR).sort_index(ascending=False).idxmax()
                optflex[s,v,avw,pl] = (idx, data.Bids_Avg[idx], data.Payments_Avg[idx], data.Payments_CVaR[idx], data.UnderDel_Avg[idx])
optflex = pd.DataFrame(optflex, index=['conf', 'Bid', 'Payments_Avg', 'Payments_CVaR', 'UnderDel_Avg']).T

optflex.to_csv(folder + 'Optimal_bids_atconf.csv')
