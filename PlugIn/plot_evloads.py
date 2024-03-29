# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:04:42 2020

@author: U546416
"""
import numpy as np
from matplotlib import pyplot as plt
import EVmodel
import scipy.stats as sts
import time
import util
import flex_payment_funcs as fpf
import pandas as pd

#%% 0 Params
t = []
t.append(time.time())

n_evs = 1000
ovn=True

# FLEET PARAMS
ch_power = 7.3

t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))  

# Load arr_dep_distr
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)

# creating structures to be used by EVmodel for Commuters
bins=np.arange(0,24.5,0.5)
adwd_LP = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe_LP = dict(pdf_a_d=arr_dep_we.values, bins=bins)     

# Energy usage = daily needs in kWh are lognorm with mu=1.44, sigma=0.79
# Transformed at daily distances, considering driving efficiency of 0.2kWh/km:
# dist/trip follows a lognorm of mu=1.44+ln(1/0.2/2), sigma=0.79
#dist = stats.lognorm.pdf(bins/0.4, loc=0, s=0.79, scale=np.exp(1.44)/0.4)
#dist = dist / dist.sum()
#plt.plot(bins, energy)
#plt.plot(bins/0.4, dist)
t = []
t.append(time.time())

########
#EV PARAMS TO EVALUATE
########
alpha = 1.31
ev_params = dict(small_ns=dict(n_if_needed=alpha,
                            batt_size=25,
                            driving_eff=25*.0009+.14),
            medium_ns=dict(n_if_needed=alpha,
                            batt_size=50,
                            driving_eff=50*.0009+.14),
            large_ns=dict(n_if_needed=alpha,
                            batt_size=75,
                            driving_eff=75*.0009+.14),
            small_s=dict(n_if_needed=10000,
                            batt_size=25,
                            driving_eff=25*.0009+.14),
            medium_s=dict(n_if_needed=10000,
                            batt_size=50,
                            driving_eff=50*.0009+.14),
            large_s=dict(n_if_needed=10000,
                            batt_size=75,
                            driving_eff=75*.0009+.14))#,
#            small_hp=dict(n_if_needed=1.5,
#                            batt_size=25,
#                            driving_eff=0.16),
#            medium_hp=dict(n_if_needed=1.5,
#                            batt_size=50,
#                            driving_eff=0.18),
#            large_hp=dict(n_if_needed=1.5,
#                            batt_size=75,
#                            driving_eff=0.20))


# DSO SERVICE PARAMS:
service_time= 15
# I will select how many EVs do i need
t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))                             


#%% 1 Compute EVs simulation: 
t.append(time.time())             
# SIMS PARAMS:
nweeks = 20
ndays = 7 * nweeks  # 50 weeks, + one extra day
step = 30 # minutes

# DO SIMULATIONS (takes about 1-2min)
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs = {}
for name, params in ev_params.items():
    evs[name] = grid.add_evs(nameset=name, n_evs=n_evs, ev_type='dumb', 
                             charging_power=ch_power,
                             arrival_departure_data={'wd':adwd_LP,
                                                     'we':adwe_LP},
                             **params)

alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=n_evs)
    
# setting the same dist and alpha param to every ev
for i, ev in evs.items():
    for j, e in enumerate(ev):
        e.dist_we = evs['small_s'][j].dist_we
        e.dist_wd = evs['small_s'][j].dist_wd
    if '_ns' in i:
        for j, e in enumerate(ev):
            e.n_if_needed = alphs[j]
    
grid.do_days()
#grid.plot_ev_load(day_ini=7, days=14)
#grid.plot_flex_pot(day_ini=7, days=14)
t.append(time.time())                              
print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))

datas = grid.get_ev_data()
j = 0
for c, ev in evs.items():
    print(c, ':\t', round(grid.ev_load[c].max()*1000/n_evs,2), 'kW/EV,\t',  datas['Avg_plug_in_ratio'][j]*7, 'charging sessions/week')
    j +=1
#%% Do fleets:
av_window = [0, 24] 
aw_s = str(av_window[0]) + '_' + str(av_window[1])
if ovn:
    shift = 12
else:
    shift = 0    
av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
if av_window == [0, 24]:
    av_window_idxs = [0, int(24*60/step)]
av_days = 'weekdays'

sets = grid.ev_sets
ch_profs = {}
dn_profs = {}
fch = {}
fdn = {}
for s in sets:
    ch_profs[s], dn_profs[s] = fpf.get_ev_profs(grid, ovn=True, av_days=av_days, nameset=s, step=step)
#    fch[s], fdn[s] = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], 
#                                         nfleets=1, nevs_fleet=n_evs)


fleet_size = 20
nfleets = 500
ch_fleets = {}
for s in sets:
    fch, _ = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], nevs_fleet=fleet_size, nfleets=nfleets)
    (k, nd, ns) = fch.shape
#    print(fch.max(axis=(1,2)).mean())
    fch = np.reshape(fch, (k*nd,ns))
    fch.sort(axis=0)
    ch_fleets[s] = fch
    print(s, fch.max().max().max())



#%% Plot in three plots:

f,axs= plt.subplots(1,3, sharey=True)
sets_n = [['small_ns', 'small_s'],
          ['medium_ns', 'medium_s'],
          ['large_ns', 'large_s']]
labels = ['Non-systematic', 'Systematic']
texts = ['25 kWh', '50 kWh', '75 kWh']
titles = ['(a) 25 kWh', '(b) 50 kWh', '(c) 75 kWh']
colors=  ['b', 'r']

fleet_size = 20
knd = ch_fleets[sets_n[0][0]].shape[0]

x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls=['--','-']

mf = [[1,1],[1,1],[1,1]]

ymax = np.round(fleet_size * ch_power * 0.6/10,0)*10
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets[sets_n[i][j]].mean(axis=0), label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets[sets_n[i][j]][int(knd*conf-1)]*mf[i][j], 
                    ch_fleets[sets_n[i][j]][int(knd*(1-conf))],
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_v3.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_v3.png'.format(fleet_size))

#%% Plot in three plots, in pu:

f,axs= plt.subplots(1,3, sharey=True)
sets_n = [['small_ns', 'small_s'],
          ['medium_ns', 'medium_s'],
          ['large_ns', 'large_s']]
labels = ['Non-systematic', 'Systematic']
texts = ['25 kWh', '50 kWh', '75 kWh']
titles = ['(a) 25 kWh', '(b) 50 kWh', '(c) 75 kWh']
colors=  ['b', 'r']

fleet_size = 20
knd = ch_fleets[sets_n[0][0]].shape[0]

x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls=['--','-']

mf = [[1,1],[1,1],[1,1]]

#ymax = np.round(fleet_size * ch_power * 0.6/10,0)*10
ymax=5
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets[sets_n[i][j]].mean(axis=0)/fleet_size, label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets[sets_n[i][j]][int(knd*conf-1)]*mf[i][j]/fleet_size, 
                    ch_fleets[sets_n[i][j]][int(knd*(1-conf))]/fleet_size,
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW/EV]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_vpu.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_vpu.png'.format(fleet_size))

#%% Plot 1000 EVs laod

fleet_size = 1000
nfleets = 1
ch_fleets1000 = {}
for s in sets:
    fch1000, _ = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], nevs_fleet=fleet_size, nfleets=nfleets)
    (k, nd, ns) = fch1000.shape
#    print(fch.max(axis=(1,2)).mean())
    fch1000 = np.reshape(fch1000, (k*nd,ns))
    fch1000.sort(axis=0)
    ch_fleets1000[s] = fch1000
    print(s, fch1000.max().max().max())
    
f,axs= plt.subplots(1,3, sharey=True)
sets_n = [['small_ns', 'small_s'],
          ['medium_ns', 'medium_s'],
          ['large_ns', 'large_s']]
labels = ['Non-systematic', 'Systematic']
texts = ['25 kWh', '50 kWh', '75 kWh']
titles = ['(a) 25 kWh', '(b) 50 kWh', '(c) 75 kWh']
colors=  ['b', 'r']


x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls=['--','-']

#mf = [[1,1],[1,.98],[1,.95]]

ymax = np.round(ch_fleets1000[sets_n[0][0]].max().max()/100+2)*100
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets1000[sets_n[i][j]].mean(axis=0), label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets1000[sets_n[i][j]][int(k*nd*conf-1)]*mf[i][j], 
                    ch_fleets1000[sets_n[i][j]][int(k*nd*(1-conf))],
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_v3.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_v3.png'.format(fleet_size))


f,axs= plt.subplots(1,3, sharey=True)
#mf = [[1,1],[1,.98],[1,.95]]

ymax = 5
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets1000[sets_n[i][j]].mean(axis=0)/fleet_size, label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets1000[sets_n[i][j]][int(k*nd*conf-1)]*mf[i][j]/fleet_size, 
                    ch_fleets1000[sets_n[i][j]][int(k*nd*(1-conf))]/fleet_size,
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_vpu.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_vpu.png'.format(fleet_size))


#%% Do all combinations of systematic & non syst  + batt sizes:: PARAMS
t = []
t.append(time.time())
# Case 1 : Company fleet IN-BUILDING
    # This means they will charge during the night
# Case 2: Commuters with medium probability of plugin (n=1)
# Case 3: Commuters with medium probability of plugin (n=5)
# Case x: JPL

n_evs = 1000
ovn=True

# FLEET PARAMS
ch_power = 7.3

t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))  

# Load arr_dep_distr
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)

# creating structures to be used by EVmodel for Commuters
bins=np.arange(0,24.5,0.5)
adwd_LP = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe_LP = dict(pdf_a_d=arr_dep_we.values, bins=bins)     

# Energy usage = daily needs in kWh are lognorm with mu=1.44, sigma=0.79
# Transformed at daily distances, considering driving efficiency of 0.2kWh/km:
# dist/trip follows a lognorm of mu=1.44+ln(1/0.2/2), sigma=0.79
#dist = stats.lognorm.pdf(bins/0.4, loc=0, s=0.79, scale=np.exp(1.44)/0.4)
#dist = dist / dist.sum()
#plt.plot(bins, energy)
#plt.plot(bins/0.4, dist)
t = []
t.append(time.time())

########
#EV PARAMS TO EVALUATE
########
ev_params = {}
for b in [25,50,75]:
    for a in [0.5,1.31,3.34,10000]:
        sb = {25:'small',
             50:'medium',
             75:'large'}
        sa = {0.5:'_nsl',
              1.31:'_nsa',
              3.34:'_nsh',
              10000:'_s'}
        ev_params[sb[b]+sa[a]] = dict(n_if_needed=a,
                                        batt_size=b,
                                        driving_eff=b*.0009+.14)



# DSO SERVICE PARAMS:
service_time= 15
# I will select how many EVs do i need
t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))                             


#%% 1 Compute EVs simulation: 
t.append(time.time())             
# SIMS PARAMS:
nweeks = 20
ndays = 7 * nweeks  # 50 weeks, + one extra day
step = 30 # minutes

# DO SIMULATIONS (takes about 5min)
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs = {}
for name, params in ev_params.items():
    evs[name] = grid.add_evs(nameset=name, n_evs=n_evs, ev_type='dumb', 
                             charging_power=ch_power,
                             arrival_departure_data_wd=adwd_LP,
                             arrival_departure_data_we=adwe_LP,
                             **params)
a0 = 1.31
alphs = np.random.lognormal(mean=np.log(a0), sigma=1, size=n_evs)
    
# setting the same dist and alpha param to every ev
for i, ev in evs.items():
    for j, e in enumerate(ev):
        e.dist_we = evs['small_s'][j].dist_we
        e.dist_wd = evs['small_s'][j].dist_wd
    if '_ns' in i:
        for j, e in enumerate(ev):
            a1 = 1.31
            if '_nsh' in i:
                a1 = 3.34
            if '_nsl' in i:
                a1 = 0.5
            e.n_if_needed = alphs[j]*a1/a0
    
grid.do_days()
#grid.plot_ev_load(day_ini=7, days=14)
#grid.plot_flex_pot(day_ini=7, days=14)
t.append(time.time())                              
print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))

datas = grid.get_ev_data()
j = 0
for c, ev in evs.items():
    print(c, ':\t', round(grid.ev_load[c].max()*1000/n_evs,2), 'kW/EV,\t',  datas['Avg_plug_in_ratio'][j]*7, 'charging sessions/week')
    j +=1
    
#%%# Do charging profiles:
av_window = [0, 24] 
aw_s = str(av_window[0]) + '_' + str(av_window[1])
if ovn:
    shift = 12
else:
    shift = 0    
av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
if av_window == [0, 24]:
    av_window_idxs = [0, int(24*60/step)]
av_days = 'weekdays'

sets = grid.ev_sets
ch_profs = {}
dn_profs = {}
fch = {}
fdn = {}
for s in sets:
    ch_profs[s], dn_profs[s] = fpf.get_ev_profs(grid, ovn=True, av_days=av_days, nameset=s, step=step)
#    fch[s], fdn[s] = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], 
#                                         nfleets=1, nevs_fleet=n_evs)

#%% Do fleets of given size

fleetsizes = [10,100,1000]
nfleets = 500
fs = []
for fleet_size in fleetsizes:
    ch_fleets = {}
    for s in sets:
        fch, _ = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], nevs_fleet=fleet_size, nfleets=nfleets)
        (k, nd, ns) = fch.shape
        avgm = fch.max(axis=(1,2)).mean()
        fch = np.reshape(fch, (k*nd,ns))
        fch.sort(axis=0)
        ch_fleets[s] = fch
        print(fleet_size, s, ':\t', round(fch.max().max().max()), ':\t', round(avgm))
        
    # Plot everything in a 4x3 plot
    f,axs= plt.subplots(4,3)
    fs.append(f)
    bb = [25,50,75]
    aa = [0.5,1.31,3.34,10000]
    sb = {25:'small',
         50:'medium',
         75:'large'}
    sa = {0.5:'_nsl',
          1.31:'_nsa',
          3.34:'_nsh',
          10000:'_s'}
    
    lb = {25:'25 kWh',
         50:'50 kWh',
         75:'75 kWh'}
    la = {0.5:'Low plug-in',
          1.31:'Average plug-in',
          3.34:'High plug-in',
          10000:'Systematic'}
    
    sets_n = [[sb[b] + sa[a] for b in bb] for a in aa]
    labels = [[lb[b] + ', ' + la[a] for b in bb] for a in aa]
    #texts = ['25 kWh', '50 kWh', '75 kWh']
    colors=  ['b', 'r','g']
    
    
    x = np.arange(0,24,step/60)
    xticks = np.arange(0,25,4)
    
    conf = 1
    
    ls = [':','--','-.','-',]
    
    ymax = np.ceil((avgm*1.2)/10)*10+10
    
    
    for i, axx in enumerate(axs):
        for j, ax in enumerate(axx):
            plt.sca(ax)
            plt.plot(x, ch_fleets[sets_n[i][j]].mean(axis=0), color=colors[j],
                     linestyle=ls[i])
            plt.fill_between(x, ch_fleets[sets_n[i][j]][int(k*nd*conf-1)], 
                        ch_fleets[sets_n[i][j]][int(k*nd*(1-conf))],
                        alpha=0.1, color=colors[j], label='_90% range')
            plt.xlabel('Time [h]')
            plt.ylabel('Power [kW]')
            plt.text(12,ymax*.9, labels[i][j], fontweight='bold', horizontalalignment='center')
            plt.xticks(xticks, (xticks-12)%24)
            plt.xlim(0,23.5)
            plt.ylim(0,ymax)
            plt.grid('--', alpha=0.5)
    f.set_size_inches(11,4.76*4.5)   
    f.tight_layout()
#    plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v2.pdf'.format(fleet_size))
#    plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v2.png'.format(fleet_size))
for i in range(len(fs)):
    fs[i].savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v2.pdf'.format(fleetsizes[i]))
    fs[i].savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v2.png'.format(fleetsizes[i]))
#%%
f.tight_layout()
#f.legend(ncol=2, loc=8)
## resizing axs to leave space for legend
#for i, ax in enumerate(axs):
#    pos = ax.get_position()
#    dy = 0.06
#    ax.set_position([pos.x0, pos.y0+dy, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v2.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v2.png'.format(fleet_size))


#%% Do fleets of various sizes
fleet_sizes = [10,100,1000]
nfleets = 500
ch_fleets = {}
for fs in fleet_sizes:
    for s in sets:
        fch, _ = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], nevs_fleet=fs, nfleets=nfleets)
        (k, nd, ns) = fch.shape
        avgm = fch.max(axis=(1,2)).mean()
        fch = np.reshape(fch, (k*nd,ns))
        fch.sort(axis=0)
        ch_fleets[fs, s] = fch/fs # fleet profiles per EV
    print(fs, s, ':\t', round(fch.max().max().max()), ':\t', round(avgm))
    
#%% Plot everything in a 4x3 plot
f,axs= plt.subplots(4,3)

bb = [25,50,75]
aa = [0.5,1.31,3.34,10000]
sb = {25:'small',
     50:'medium',
     75:'large'}
sa = {0.5:'_nsl',
      1.31:'_nsa',
      3.34:'_nsh',
      10000:'_s'}

lb = {25:'25 kWh',
     50:'50 kWh',
     75:'75 kWh'}
la = {0.5:'Low plug-in',
      1.31:'Average plug-in',
      3.34:'High plug-in',
      10000:'Systematic'}

sets_n = [[sb[b] + sa[a] for b in bb] for a in aa]
labels = [[lb[b] + ', ' + la[a] for b in bb] for a in aa]
#texts = ['25 kWh', '50 kWh', '75 kWh']
colors=  ['b', 'r','g']

fleetsizes = [10,100,1000]

x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls = [':','--','-.','-',]

ymax = ch_power


for i, axx in enumerate(axs):
    for j, ax in enumerate(axx):
        plt.sca(ax)
        for m, fs in enumerate(fleetsizes):
            plt.plot(x, ch_fleets[fs, sets_n[i][j]].mean(axis=0), color=colors[m],
                     linestyle=ls[m], label='{} EVs'.format(fs))
            plt.fill_between(x, ch_fleets[fs, sets_n[i][j]][int(k*nd*conf-1)], 
                        ch_fleets[fs, sets_n[i][j]][int(k*nd*(1-conf))],
                        alpha=0.1, color=colors[m], label='_90% range')
        plt.xlabel('Time [h]')
        plt.ylabel('Power [kW/EV]')
        plt.text(12,ymax*.9, labels[i][j], fontweight='bold', horizontalalignment='center')
        plt.xticks(xticks, (xticks-12)%24)
        plt.xlim(0,23.5)
        plt.ylim(0,ymax)
        plt.grid('--', alpha=0.5)
f.set_size_inches(11,4.76*4.5)   
f.tight_layout()
#%%
f.tight_layout()
f.legend(ncol=3, loc=8)
# resizing axs to leave space for legend
for i, axss in enumerate(axs):
    for j, ax in enumerate(axss):
        pos = ax.get_position()
        dy = 0.006
        ax.set_position([pos.x0, pos.y0+dy*(i+1), pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v3.pdf'.format(fleetsizes))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_all_v3.png'.format(fleetsizes))



#%% 0 Params --- Off Peak Charging
t = []
t.append(time.time())

n_evs = 1000
ovn=True

# FLEET PARAMS
ch_power = 7.3

t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))  

# Load arr_dep_distr
folder_en = r'c:\user\U546416\Documents\PhD\Data\ElectricNation\Outputs\\'
arr_dep_wd = pd.read_csv(folder_en + 'EN_arrdep_wd.csv', index_col=0)
arr_dep_we = pd.read_csv(folder_en + 'EN_arrdep_we.csv', index_col=0)

# creating structures to be used by EVmodel for Commuters
bins=np.arange(0,24.5,0.5)
adwd_LP = dict(pdf_a_d=arr_dep_wd.values, bins=bins)
adwe_LP = dict(pdf_a_d=arr_dep_we.values, bins=bins)     

# Energy usage = daily needs in kWh are lognorm with mu=1.44, sigma=0.79
# Transformed at daily distances, considering driving efficiency of 0.2kWh/km:
# dist/trip follows a lognorm of mu=1.44+ln(1/0.2/2), sigma=0.79
#dist = stats.lognorm.pdf(bins/0.4, loc=0, s=0.79, scale=np.exp(1.44)/0.4)
#dist = dist / dist.sum()
#plt.plot(bins, energy)
#plt.plot(bins/0.4, dist)
t = []
t.append(time.time())

########
#EV PARAMS TO EVALUATE
########
alpha = 1.31
ev_params = dict(small_ns=dict(n_if_needed=alpha,
                            batt_size=25,
                            driving_eff=25*.0009+.14),
            medium_ns=dict(n_if_needed=alpha,
                            batt_size=50,
                            driving_eff=50*.0009+.14),
            large_ns=dict(n_if_needed=alpha,
                            batt_size=75,
                            driving_eff=75*.0009+.14),
            small_s=dict(n_if_needed=10000,
                            batt_size=25,
                            driving_eff=25*.0009+.14),
            medium_s=dict(n_if_needed=10000,
                            batt_size=50,
                            driving_eff=50*.0009+.14),
            large_s=dict(n_if_needed=10000,
                            batt_size=75,
                            driving_eff=75*.0009+.14))#,
#            small_hp=dict(n_if_needed=1.5,
#                            batt_size=25,
#                            driving_eff=0.16),
#            medium_hp=dict(n_if_needed=1.5,
#                            batt_size=50,
#                            driving_eff=0.18),
#            large_hp=dict(n_if_needed=1.5,
#                            batt_size=75,
#                            driving_eff=0.20))


# DSO SERVICE PARAMS:
service_time= 15
# I will select how many EVs do i need
t.append(time.time())                            
print('Loaded params, t={:.2f} seg'.format(t[-1]-t[-2]))                             


#%% 1 Compute EVs simulation -- OFF-PEAK CHARGING: 
t.append(time.time())             
# SIMS PARAMS:
nweeks = 20
ndays = 7 * nweeks  # 50 weeks, + one extra day
step = 30 # minutes

# DO SIMULATIONS (takes about 1-2min)
grid = EVmodel.Grid(ndays=ndays, step=step, verbose=False)
evs = {}
for name, params in ev_params.items():
    evs[name] = grid.add_evs(nameset=name, n_evs=n_evs, ev_type='dumb', 
                             charging_power=ch_power,
                             tou_ini=23,
                             tou_end=6,
                             tou_we=True,
                             tou_ini_we=23,
                             tou_end_we=6,
                             arrival_departure_data_wd=adwd_LP,
                             arrival_departure_data_we=adwe_LP,
                             **params)

alphs = np.random.lognormal(mean=np.log(alpha), sigma=1, size=n_evs)
    
# setting the same dist and alpha param to every ev
for i, ev in evs.items():
    for j, e in enumerate(ev):
        e.dist_we = evs['small_s'][j].dist_we
        e.dist_wd = evs['small_s'][j].dist_wd
    if '_ns' in i:
        for j, e in enumerate(ev):
            e.n_if_needed = alphs[j]
    
grid.do_days()
#grid.plot_ev_load(day_ini=7, days=14)
#grid.plot_flex_pot(day_ini=7, days=14)
t.append(time.time())                              
print('Simulated Grid, t={:.2f} seg'.format(t[-1]-t[-2]))

datas = grid.get_ev_data()
j = 0
for c, ev in evs.items():
    print(c, ':\t', round(grid.ev_load[c].max()*1000/n_evs,2), 'kW/EV,\t',  datas['Avg_plug_in_ratio'][j]*7, 'charging sessions/week')
    j +=1
#%% Do fleets -- OFF-PEAK CHARGING:
av_window = [0, 24] 
aw_s = str(av_window[0]) + '_' + str(av_window[1])
if ovn:
    shift = 12
else:
    shift = 0    
av_window_idxs = [int(((av_window[0]-shift)%24)*60/step), int(((av_window[1]-shift)%24)*60/step)]
if av_window == [0, 24]:
    av_window_idxs = [0, int(24*60/step)]
av_days = 'weekdays'

sets = grid.ev_sets
ch_profs = {}
dn_profs = {}
fch = {}
fdn = {}
for s in sets:
    ch_profs[s], dn_profs[s] = fpf.get_ev_profs(grid, ovn=True, av_days=av_days, nameset=s, step=step)
#    fch[s], fdn[s] = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], 
#                                         nfleets=1, nevs_fleet=n_evs)


fleet_size = 20
nfleets = 500
ch_fleets = {}
for s in sets:
    fch, _ = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], nevs_fleet=fleet_size, nfleets=nfleets)
    (k, nd, ns) = fch.shape
#    print(fch.max(axis=(1,2)).mean())
    fch = np.reshape(fch, (k*nd,ns))
    fch.sort(axis=0)
    ch_fleets[s] = fch
    print(s, fch.max().max().max())



#%% Plot in three plots -- OFF-PEAK CHARGING:

f,axs= plt.subplots(1,3, sharey=True)
sets_n = [['small_ns', 'small_s'],
          ['medium_ns', 'medium_s'],
          ['large_ns', 'large_s']]
labels = ['Non-systematic', 'Systematic']
texts = ['25 kWh', '50 kWh', '75 kWh']
titles = ['(a) 25 kWh', '(b) 50 kWh', '(c) 75 kWh']
colors=  ['b', 'r']

fleet_size = 20
knd = ch_fleets[sets_n[0][0]].shape[0]

x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls=['--','-']

mf = [[1,1],[1,1],[1,1]]

ymax = np.round(fleet_size * ch_power /10,0)*10
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets[sets_n[i][j]].mean(axis=0), label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets[sets_n[i][j]][int(knd*conf-1)]*mf[i][j], 
                    ch_fleets[sets_n[i][j]][int(knd*(1-conf))],
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_v3.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_v3.png'.format(fleet_size))

#%% Plot in three plots, in pu-- OFF-PEAK CHARGING:

f,axs= plt.subplots(1,3, sharey=True)
sets_n = [['small_ns', 'small_s'],
          ['medium_ns', 'medium_s'],
          ['large_ns', 'large_s']]
labels = ['Non-systematic', 'Systematic']
texts = ['25 kWh', '50 kWh', '75 kWh']
titles = ['(a) 25 kWh', '(b) 50 kWh', '(c) 75 kWh']
colors=  ['b', 'r']

fleet_size = 20
knd = ch_fleets[sets_n[0][0]].shape[0]

x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls=['--','-']

mf = [[1,1],[1,1],[1,1]]

#ymax = np.round(fleet_size * ch_power * 0.6/10,0)*10
ymax=8
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets[sets_n[i][j]].mean(axis=0)/fleet_size, label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets[sets_n[i][j]][int(knd*conf-1)]*mf[i][j]/fleet_size, 
                    ch_fleets[sets_n[i][j]][int(knd*(1-conf))]/fleet_size,
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW/EV]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_vpu.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_vpu.png'.format(fleet_size))

#%% Plot 1000 EVs laod -- OFF-PEAK CHARGING

fleet_size = 1000
nfleets = 1
ch_fleets1000 = {}
for s in sets:
    fch1000, _ = fpf.get_fleet_profs(ch_profs[s], dn_profs[s], nevs_fleet=fleet_size, nfleets=nfleets)
    (k, nd, ns) = fch1000.shape
#    print(fch.max(axis=(1,2)).mean())
    fch1000 = np.reshape(fch1000, (k*nd,ns))
    fch1000.sort(axis=0)
    ch_fleets1000[s] = fch1000
    print(s, fch1000.max().max().max())
    
f,axs= plt.subplots(1,3, sharey=True)
sets_n = [['small_ns', 'small_s'],
          ['medium_ns', 'medium_s'],
          ['large_ns', 'large_s']]
labels = ['Non-systematic', 'Systematic']
texts = ['25 kWh', '50 kWh', '75 kWh']
titles = ['(a) 25 kWh', '(b) 50 kWh', '(c) 75 kWh']
colors=  ['b', 'r']


x = np.arange(0,24,step/60)
xticks = np.arange(0,25,4)

conf = 1

ls=['--','-']

#mf = [[1,1],[1,.98],[1,.95]]

ymax = np.round(ch_fleets1000[sets_n[1][0]].max().max()/100+2)*100
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets1000[sets_n[i][j]].mean(axis=0), label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets1000[sets_n[i][j]][int(k*nd*conf-1)]*mf[i][j], 
                    ch_fleets1000[sets_n[i][j]][int(k*nd*(1-conf))],
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_v3.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_v3.png'.format(fleet_size))


f,axs= plt.subplots(1,3, sharey=True)
#mf = [[1,1],[1,.98],[1,.95]]

ymax = 8
for i, ax in enumerate(axs):
    plt.sca(ax)
    for j in range(2):
        plt.plot(x, ch_fleets1000[sets_n[i][j]].mean(axis=0)/fleet_size, label=labels[j], color=colors[j],
                 linestyle=ls[j])
        plt.fill_between(x, ch_fleets1000[sets_n[i][j]][int(k*nd*conf-1)]*mf[i][j]/fleet_size, 
                    ch_fleets1000[sets_n[i][j]][int(k*nd*(1-conf))]/fleet_size,
                    alpha=0.1, color=colors[j], label='_90% range')
    plt.ylim(0,ymax)
    plt.xlabel('Time [h]')
    if i == 0:
        plt.ylabel('Power [kW/EV]')
#    plt.text(1,ymax*0.9, texts[i], fontweight='bold')
    plt.title(titles[i], y=-0.25)
    plt.xticks(xticks, (xticks-12)%24)
    plt.grid('--', alpha=0.5)
    plt.xlim(0,max(x))
    
f.set_size_inches(9,4)   
f.tight_layout()
f.legend(ncol=2, loc=9)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
    dy = 0.06
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])

plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_vpu.pdf'.format(fleet_size))
plt.savefig(r'c:\user\U546416\Pictures\PlugInModel\EVLoad\\' + '{}evs_3plots_tou_vpu.png'.format(fleet_size))
