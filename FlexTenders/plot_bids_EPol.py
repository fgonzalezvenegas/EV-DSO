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

#%% Load data
sets = ['Company', 'Commuter_HP', 'Commuter_LP']
sstr = [s.replace('_',' ') for s in sets]
folder = r'C:\Users\u546416\AnacondaProjects\EV-DSO\FlexTenders\EnergyPolicy\correct\\'
av_w  = ['0_24', '17_20']
bds = ['Bids_Avg', 'Bids_perc_l', 'Bids_perc_h']
pay = ['Payments_Avg', 'Payments_perc_l', 'Payments_perc_h']
und = ['UnderDel_Avg', 'UnderDel_perc_l', 'UnderDel_perc_h']

# read all files and and put them in one big DF
fs = [f for f in os.listdir(folder) if f.endswith('.csv')]

if 'full_data.csv' in fs:
    res = pd.read_csv(folder + 'full_data.csv', engine='python', index_col=np.arange(0,9))
else:
    res = pd.DataFrame()
    for f in fs:
        data = pd.read_csv(folder + f, engine='python')
        avw = '0_24' if '0_24' in f else '17_20'
        fleet = 'Company' if 'Company' in f else '' + 'Commuter_HP' if 'HP' in f else 'Commuter_LP'
        data['AvWindow'] = avw
        data['Fleet'] = fleet
        res = pd.concat([res, data], ignore_index=True)
    
    res.set_index(['VxG','Fleet', 'AvWindow', 'nevs', 'nactivation', 'service_duration', 'confidence',
           'penalty_threshold', 'penalties'], inplace=True, drop=True)
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
f.set_size_inches(8.5,4)
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
    
#%% Remuneration for diffs thresholds, Three plots, one figure
    
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
th = 0.8
p=17.5

mrks = [['','',''],['x','','o']]
mrksize = [4,0,3]
fleets = res.index.levels[3]
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
remu = [np.array([res.Payments_Avg.loc[v, s, a, nevs, nac,sd,conf,th,p] for s in sets for a in av_w]) * price/50
        for v in vxg for price in prices]
idx = pd.MultiIndex.from_arrays([np.repeat(['V1G', 'V2G'],3), np.tile(prices,2)], names=['VxG', 'Prices'])
cols = pd.MultiIndex.from_arrays([np.repeat(sets, 2), np.tile(av_w,3)], names=['Sets', 'AvWindow'])
remu = pd.DataFrame(remu, index=idx, columns=cols).round(decimals=1)
remu.to_csv(folder+ 'remuneration_table.csv')


# baselines
nevs=136.0
bds = ['Bids_perc_l', 'Bids_Avg', 'Bids_perc_h']
bls = [np.concatenate([res[bds].loc['v1g', s, a, nevs, nac,sd,conf,th,p].values for a in av_w])
        for s in sets]
idx = sets
cols = pd.MultiIndex.from_arrays([np.repeat(av_w, 3), np.tile(['Min', 'Avg', 'Max'],2)], names=['AvWindow', 'Stat'])
bls = pd.DataFrame(bls, index=idx, columns=cols).round(decimals=2)
bls.to_csv(folder+ 'baselines_table.csv')


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