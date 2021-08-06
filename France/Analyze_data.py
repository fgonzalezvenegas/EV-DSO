# -*- coding: utf-8 -*-

import pandas as pd
import util
import numpy as np
import matplotlib.pyplot as plt

# useful fx:

def compute_KPIs(global_data):
    """ Energy ratio = EV load over Total base load in Energy
        Peak ratio = EV peak load over Peak base load in Power
        Peak increase = Tot peak load over base peak load (i.e. how much does the peak increases)
        Coincident peak ratio = Increase of peak load over Max EV load
    """ 
    cols = ['energy_ratio', 'peak_ratio', 'peak_increase', 'coincident_peak_ratio']
    energy_ratio  = global_data.Tot_ev_charge_MWh / global_data.Base_load_MWh
    peak_ratio  = global_data.Max_ev_load_MW / global_data.Max_base_load_MW
    peak_increase = global_data.Max_load_MW / global_data.Max_base_load_MW
    coincident_peak_ratio = (global_data.Max_load_MW - global_data.Max_base_load_MW) / global_data.Max_ev_load_MW
    kpis = pd.concat([energy_ratio, peak_ratio, peak_increase, coincident_peak_ratio], axis=1)
    kpis.columns = cols
    return kpis

def plot_histogram(hs, bins, ax=None, labels=None, title=None, **bar_params):
    """ Plot histogram series of len n and bins of len n+1
    """
    # identify how many histogram series:
    if len(hs) == len(bins) - 1:
        nhs = 1
        hs = [hs]
    else:
        nhs = len(hs)
    if labels == None:
        labels = ['' for i in range(nhs)]
    width = (bins[1]-bins[0])/nhs
    x = np.array(bins[0:-1])
    if ax==None:
        f, ax = plt.subplots()
    for i in range(nhs):
        ax.bar(x + width * (i+0.5), hs[i], width=width, label=labels[i], **bar_params)
    if labels[0] != '':
        plt.legend()
    if title!=None:
        plt.title(title)
    return ax
    
    
    
#%%
print('Reading results files')
# results folder
folder = r'c:\user\U546416\Documents\PhD\Data\Simulations\Results'
# reading results:
# dumb
fd = r'\Dumb_EV05_W01'
global_dumb = pd.read_csv(folder + fd + r'\global_data.csv',
                      engine='python', index_col=0)
ev_dumb = pd.read_csv(folder + fd + r'\ev_data.csv',
                      engine='python', index_col=0)
#evload_day_dumb = pd.read_csv(folder + fd + r'\global_data.csv')
#evload_night_dumb = pd.read_csv(folder + fd + r'\global_data.csv')
# mod
fd = r'\Mod_EV05_W01'
global_mod = pd.read_csv(folder + fd + r'\global_data.csv',
                      engine='python', index_col=0)
ev_mod = pd.read_csv(folder + fd + r'\ev_data.csv',
                      engine='python', index_col=0)

# randstart
fd = r'\RandStart_EV05_W01'
global_rs = pd.read_csv(folder + fd + r'\global_data.csv',
                      engine='python', index_col=0)
ev_rs = pd.read_csv(folder + fd + r'\ev_data.csv',
                      engine='python', index_col=0)

# ToU - HC
fd = r'\dumb_ToU_EV05_W01'
global_tou = pd.read_csv(folder + fd + r'\global_data.csv',
                      engine='python', index_col=0)
ev_tou = pd.read_csv(folder + fd + r'\ev_data.csv',
                      engine='python', index_col=0)


# Do polygons
print('Reading Polygons')
#polygons_SS = util.load_polygons_SS()

print('Reading SS')
#folder_ssdata = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau'
#SS = pd.read_csv(folder_ssdata + r'\postes_source.csv',
#                                  engine='python', index_col=0)


#%% Compute basic data

kpis_dumb = compute_KPIs(global_dumb)
kpis_mod = compute_KPIs(global_mod)
kpis_rs = compute_KPIs(global_rs)
kpis_tou = compute_KPIs(global_tou)

nbins = 20
bins = {'energy_ratio' : np.linspace(0, 1, nbins+1),
        'peak_increase': np.linspace(1, 2, nbins+1),
        'peak_ratio' : np.linspace(0,1, nbins+1), 
        'coincident_peak_ratio' : np.linspace(0,1, nbins+1)}

hs_dumb = {k: np.histogram(kpis_dumb[k], bins[k])[0] for k in bins}
hs_mod = {k: np.histogram(kpis_mod[k], bins[k])[0] for k in bins}
hs_rs = {k: np.histogram(kpis_rs[k], bins[k])[0] for k in bins}
hs_tou = {k: np.histogram(kpis_tou[k], bins[k])[0] for k in bins}

#% Plot KPIs
labels=['Naturelle', 'Modulée', 'HC']
axs = {k: plot_histogram([hs_dumb[k], hs_mod[k], hs_tou[k]], bins[k], alpha=0.9, labels=labels, title=k) for k in bins.keys()}
#%% Plot Peak Load hist
bins = np.arange(0,1.5,0.075)
hdumb = np.histogram(global_dumb.Peak_ss_charge_pu, bins)[0]
hmod = np.histogram(global_mod.Peak_ss_charge_pu, bins)[0]
hrs = np.histogram(global_rs.Peak_ss_charge_pu, bins)[0]
htou = np.histogram(global_tou.Peak_ss_charge_pu, bins)[0]
plt.plot(bins[:-1],hdumb, label='Naturelle')
plt.plot(bins[:-1],hmod, label='Modulée')
plt.plot(bins[:-1],htou, label='HP/HC')
plt.legend()

#%% Number of Ovl and h of ovl
novld = (global_dumb.Peak_ss_charge_pu > 1).sum()
novlmod = (global_mod.Peak_ss_charge_pu > 1).sum()
novlrs = (global_rs.Peak_ss_charge_pu > 1).sum()
novltou = (global_tou.Peak_ss_charge_pu > 1).sum()