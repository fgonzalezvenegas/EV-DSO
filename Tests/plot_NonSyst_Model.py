# -*- coding: utf-8 -*-

""" Plot non systematic charging behavior model
"""

import numpy as np
import matplotlib.pyplot as plt
# Case 1

#%%
def plot_one(ax=None, batt=40, kms=60, eff=0.2,  range_anx=1.5, 
             n=1, max_soc=100, deterministic=False,
             label='_', extralines=False, annotations=False, 
             **kwargs):
    """ Plot one line of 
    """
    step = 0.1
    soc = np.arange(0,100+step,step)
    
    eff = 0.2 #[kWh/km]
    energy_needed = kms * eff #kWh
        
    soc_needed = energy_needed / batt * 100
    
    soc_min_anx = soc_needed * range_anx
    
    prob_plugin = np.ones(len(soc)) 
    
    delta_soc = max_soc - soc_min_anx
    
    for k in range(len(soc)):
        if k * step > soc_min_anx:
            if deterministic:
                prob_plugin[k] = 0
            elif k * step <= max_soc:
                prob_plugin[k] = 1-((k*step-soc_min_anx)/delta_soc)**(n)
                
    if ax==None:
        f, ax = plt.subplots()
    ax.plot(soc, prob_plugin, label=label, **kwargs)
    if extralines in ['all', 'min', True]:
        ax.vlines(x=soc_needed, ymin=0, ymax=1.5, colors='k', linestyles='--')
    if extralines == 'all':
        ax.vlines(x=soc_min_anx, ymin=0, ymax=1.5, colors='k', linestyles='--')
    if annotations:
        ax.text('Minimum needed SOC', 
                    x=soc_needed+1, y=1.01)
#    if annotations == 'all':
#        ax.annotate("SOC needed\nfor next trip", 
#                xy=(soc_needed, 0.95), 
#                xytext=(soc_needed*range_anx+10, 0.95),
#                arrowprops=dict(arrowstyle="->",
#                                color='red'))

    #ax.vlines(x=max_soc, ymin=0, ymax=1.5, colors='k', linestyles='--')
    ax.set_xlim([0,100])
    ax.set_ylim([0,1.1])
    
    #plt.legend(loc=3)
    plt.xlabel('State of Charge [%]')
    plt.ylabel('Probability')
    ax.set_title('Plug-in probability')
    plt.legend()
    #plt.grid('on')
    

#%% Plot non syst for various n
n = [1/4,1,4]
f, ax = plt.subplots()
plot_one(ax, kms=60, deterministic=True, label='Deterministic', extralines=True)
plot_one(ax, kms=60, n=1, label='Probabilistic, n=1')
ax.text(s='Minimum needed SOC', x=47,y=1.02)
plt.legend(loc=3)
#%%
plot_one(ax, kms=60, n=1/4, label='Probabilistic, n=1/4')
plot_one(ax, kms=60, n=4, label='Probabilistic, n=4')

#%% Plot adjusted to GonzalezGarrido DTU 2019
# n=1.8
# Ranx = 1.5
f, ax = plt.subplots()
ds=[20,25,30,35,40,50,60,80,100]
colors = ['grey', 'mediumblue', 'purple', 'red', 'saddlebrown', 'orange', 'goldenrod', 'limegreen', 'dodgerblue']
for i in range(len(ds)):
    plot_one(ax=ax, kms=ds[i], n=1.8, label=str(ds[i])+ ' kms', color=colors[i])
ax.set_xticks(np.arange(0,100+100/12,100/12)) 
ax.set_xticklabels([0,'','',25,'','',50,'','',75,'','',100])   
ax.set_yticks(np.arange(0,1.1,0.1))

#%% Plot sigmoids
k = [0.089225628, 0.0903958, 0.091597073, 0.092830704, 0.0957316640, 
     0.10025716, 0.1139285910, 0.1612322750, 0.306341323]
soc_m = [67.5,70.8,74.2,77.5,80.0,85.8,88.3,92.5,95.0]
f, ax = plt.subplots()
ds=np.array([20,25,30,35,40,50,60,80,100])
x = np.arange(0,101,1)

batt = 40
eff = 0.2
neededsoc = ds * eff / batt
ranx = 1.5

colors = ['grey', 'mediumblue', 'purple', 'red', 'saddlebrown', 'orange', 'goldenrod', 'limegreen', 'dodgerblue']
for i in range(len(ds)):
    p = 1 - 1/(1+np.exp(-k[i] * (x-soc_m[i])))
    ax.plot(x, p, color=colors[i], label=str(ds[i]) + ' kms')
ax.set_xticks(np.arange(0,100+100/12,100/12)) 
ax.set_xticklabels([0,'','',25,'','',50,'','',75,'','',100])   
ax.set_yticks(np.arange(0,1.1,0.1))
plt.grid()
ax.set_xlim([0,100])
ax.set_ylim([0,1.1])

#plt.legend(loc=3)
plt.xlabel('State of Charge [%]')
plt.ylabel('Probability')
ax.set_title('Plug-in probability')
plt.legend()

#%% Plot 3 cases
f, ax = plt.subplots()
plt.plot([0,100],[1,1], 'b', label='Company')
plot_one(ax, n=5, kms=40, label='Commuter HP', color='r', linestyle='--')
plot_one(ax, n=1, kms=40, label='Commuter MP', color='g', linestyle='-.')
ax.vlines(x=40*0.2/40*1.5*100, ymin=0, ymax=1.5, colors='k', linestyles=':')
ax.text(s='Minimum SOC for \nnext trip (' + r'$\xi\cdot\rho$' + ')', x=40*0.2/40*1.5*100+1, y=1.05)
ax.set_ylim([0,1.2])