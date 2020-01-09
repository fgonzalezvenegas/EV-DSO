# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:38:55 2019

@author: U546416
"""

import numpy as np
import pandas as pd
import time
import mobility as mb
import util
import matplotlib.pyplot as plt
import scipy.stats as st

#%% Re-doing it in a more proper way

# Reading Tgeo - it has the information required for each commune
folder = 'c:/user/U546416/Documents/PhD/Data/Mobilité/'
folderMod = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\\'
data_file = 'data-flux-mob-dreal.txt'
geo_file = 'geoRefs.csv'
modal_file = 'ModeTransport.csv'

Tgeo = pd.read_csv(folder + geo_file, engine='python', sep=';', index_col=0)
flux = pd.read_csv(folder + data_file, engine='python')
Modal = pd.read_csv(folderMod + modal_file, engine='python', sep=';', index_col=0)


#%% Computing Histograms per commune (work and home)
hh_sc = {}
hh_dc = {}
hw_dc = {}

t0 = time.time()
print('Starting computing histograms considering modal ratio')
tlast = t0 
nflux = 0

MC = ['MC-AUCUN','MC-PIED','MC-2ROUES','MC-VOITURE','MC-TC']
DC = ['DC-AUCUN','DC-PIED','DC-2ROUES','DC-VOITURE','DC-TC']
modal_mc = Modal['MC-VOITURE']/Modal[MC].sum(axis=1)
modal_dc = Modal['DC-VOITURE']/Modal[DC].sum(axis=1)

bins = [i*2 for i in range(51)]
nl = flux.shape[0]

corse = ['2A', '2B', '96', '97']
etranger = ['AL', 'BE', 'DE', 'LU', 'IT', 'SU', 'ES', 'MO']
for i in flux.index:
    cinit = flux.CODINIT[i]
    if cinit[0:2] in corse:
        # hors Corse
        continue
    cend = flux.CODEND[i]
    if cinit == cend:
        f = int(np.ceil(flux.FLUX[i]) * modal_mc[cinit])
    else:
        f = int(np.ceil(flux.FLUX[i]) * modal_dc[cinit])
    d = flux.Distance[i]
    
    points = mb.distance2cities(f, Tgeo.Size[cinit], Tgeo.Size[cend], d)
    
    h = np.histogram(points, bins)[0]
    if cinit == cend:
        hh_sc[cinit] = h
    else:
        if cinit in hh_dc:
            hh_dc[cinit] = hh_dc[cinit] + h
        else:
            hh_dc[cinit] = h
        if not (cend[0:2] in etranger):
            if cend in hw_dc:
                hw_dc[cend] = hw_dc[cend] + h
            else:
                hw_dc[cend] = h
    nflux += f
    
    if i % 20000 == 0:
        tnew = time.time()
        print('Counter: ', i, 'Computed Flux', nflux, ', Remaining lines:', nl-i)
        print('Time :', round(tnew-t0,1), 's, deltaTime :', round(tnew - tlast,1), 's')
        tlast = tnew

hh_dc = pd.DataFrame(hh_dc).T
hh_sc = pd.DataFrame(hh_sc).T
hw_dc = pd.DataFrame(hw_dc).T

print('Finished computed histograms per commune')
print('Total rows = ', i, '; Total workers = ', nflux)
print('Total time = ', time.time()-t0, '; deltaTime =', time.time()-tlast)
print('Sum Thome: ', hh_dc.sum().sum() + hh_sc.sum().sum())
print('Sum Twork: ', hw_dc.sum().sum() + hh_sc.sum().sum())           

bin_n = ['bin{}_{}'.format(int(i*2), int((i+1)*2)) for i in range(50)]
hh_dc.columns = bin_n
hh_sc.columns = bin_n
hw_dc.columns = bin_n
# Save
hh_dc.to_csv(folder + 'hhome_diffcom_modal.csv')
hh_sc.to_csv(folder + 'hhome_samecom_modal.csv')
hw_dc.to_csv(folder + 'hwork_diffcom_modal.csv')
#%% Plotting some things
f, ax = plt.subplots()
ax.stackplot(bins[0:-1],[hh_dc.sum(axis=0)/1000, hh_sc.sum(axis=0)/1000], labels=['H/W Different Commune', 'H/W Same Commune'])
ax.set_xlabel('Distance [km]')
ax.set_ylabel('Commuters [thousands]')
ax.set_title('One-way Commuting Distance\nAggregated for Continental France')
plt.legend()