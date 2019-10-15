# -*- coding: utf-8 -*-

""" Plot non systematic charging behavior model
"""

import numpy as np
import matplotlib.pyplot as plt
# Case 1


def plot_non_syst(n, max_soc=100, min_soc=20):
    step = 0.1
    soc = np.arange(0,100+step,step)
    
    kms = 40  #km/day
    eff = 0.2 #[kWh/km]
    energy_needed = kms * eff #kWh
    
    batt = 40
    
    soc_needed = energy_needed / batt * 100
    
    range_anx = 1.5
    
    soc_min_anx = soc_needed * range_anx
    
    prob_plugin0 = np.ones(len(soc)) 
    prob_plugin1_n = np.ones(len(soc))
    prob_pluginn_n = np.ones(len(soc))
    prob_pluginn_1 = np.ones(len(soc))
    
    delta_soc = max_soc - soc_min_anx
    
    for k in range(len(soc)):
        if k * step > soc_min_anx:
            if k * step < max_soc:
                prob_plugin0[k] = 0
                prob_plugin1_n[k] = ((max_soc-k*step)/delta_soc)**(1/n)
                prob_pluginn_n[k]  = ((max_soc-k*step)/delta_soc)**1
                prob_pluginn_1[k]  = ((max_soc-k*step)/delta_soc)**n
            else:
                prob_plugin0[k] = 0
                prob_plugin1_n[k] = 0
                prob_pluginn_n[k]  = 0
                prob_pluginn_1[k]  = 0
                
    f, ax = plt.subplots()
    ax.plot(soc, prob_plugin0, label='Deterministic')
    ax.plot(soc, prob_plugin1_n, label='Probabilistic, n=0.25')
    ax.plot(soc, prob_pluginn_n, label='Probabilistic, n=1')
    ax.plot(soc, prob_pluginn_1, label='Probabilistic, n=4')
    #ax.vlines(x=soc_needed, ymin=0, ymax=1.5, colors='k', linestyles='--')
    ax.vlines(x=soc_min_anx, ymin=0, ymax=1.5, colors='k', linestyles='--')
    ax.vlines(x=max_soc, ymin=0, ymax=1.5, colors='k', linestyles='--')
    ax.set_xlim([0,100])
    ax.set_ylim([0,1.1])
    plt.text(soc_min_anx+1, 1.02, 'Minimun needed SOC')
    plt.text(max_soc+1, 1.02, 'Maximum SOC')
    plt.legend(loc=3)
    plt.xlabel('State of Charge [%]')
    plt.ylabel('Probability')
    ax.set_title('Plug-in probability')

#%% cases
plot_non_syst(n=4, max_soc=100)
plot_non_syst(n=4, max_soc=80)
plot_non_syst(n=100, max_soc=100)
plot_non_syst(n=100, max_soc=80)