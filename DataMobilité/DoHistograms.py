# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:38:55 2019

@author: U546416
"""

import csv
import numpy as np
#import pandas as pd
import time
import mobility as mb


def saveDictHist(fold, out_fn, DictHist, GeoRefs, hist_headers):
    """ Saves an histogram dictionnary, 
    adding the key variables [3:] in GeoRefs Dict"""
    with open(fold + out_fn, 'w') as datafile:
        out_writer = csv.writer(datafile)
        out_writer.writerow(['CODE', 'ZE', 'Status', 'UU', 'Dep'] + hist_headers)
        for key in DictHist:
            out_writer.writerow([key] + GeoRefs[key][3:] + DictHist[key].tolist())


# Reading Tgeo - it has the information required for each commune
folder = 'c:/user/U546416/Documents/PhD/Data/Mobilité/'
folderMod = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\\'
data_file = 'data-flux-mob-dreal.txt'
geo_file = 'geoRefs.csv'
modal_file = 'ModeTransport.csv'

Tgeo = mb.readTgeo(folder, geo_file)
Modal = mb.readModal(folderMod, modal_file)

# Initializing Dict of histograms for communes:
hini = 0
hstep = 2
hend = 100
nbins = int((hend-hini) / hstep)

Thome = mb.initFrCommDict(Tgeo, nbins)    
Twork = mb.initFrCommDict(Tgeo, nbins)
Tdreal = mb.initFrCommDict(Tgeo, nbins)


# Computing Histograms per commune (work and home)
t0 = time.time()
print('Starting')
tlast = t0        
with open(folder + data_file, 'r') as datafile:
    data_reader = csv.reader(datafile, delimiter=',')
    headers = next(data_reader)
    print(headers)
    i = 0
    nflux = 0
    for rows in data_reader:
        # rows have [CommuneInit, CommuneEnd, Flux, distance]
        i += 1
        # points of 
        flux = int(np.ceil(float(rows[2])))

        points = mb.distance2cities(flux, Tgeo[rows[0]][2], Tgeo[rows[1]][2], float(rows[3]))
        h = mb.hist(points, hini, hstep, nbins)
        Thome[rows[0]] = np.add(Thome[rows[0]], h)
        if not(Tgeo[rows[1]][5] == '99'):
            Twork[rows[1]] = np.add(Twork[rows[1]], h)
        Tdreal[rows[0]][min(int(float(rows[3]) // hstep), nbins-1)] += flux
        nflux += flux
        if i % 100000 == 0:
            tnew = time.time()
            print('Counter: ', i, 'Computed Flux', nflux)
            print('Time :', tnew-t0, 'deltaTime :', tnew - tlast)
            tlast = tnew
            print 
            
print('Finished computed histograms per commune')
print('Total rows = ', i, '; Total workers = ', nflux)
print('Total time = ', time.time()-t0, '; deltaTime =', time.time()-tlast)
print('Sum Thome: ', sum(sum(Thome[key] for key in Thome)))
print('Sum Twork: ', sum(sum(Twork[key] for key in Twork)))
print('Sum Tdreal: ', sum(sum(Tdreal[key] for key in Tdreal)))

output_thome = 'HistHome.csv'       
output_twork = 'HistWork.csv'
output_tdreal = 'HistDreal.csv'

headers_bins = ['bin' + str(i*hstep) + '_' + str((i+1)*hstep) for i in range(nbins)]

print('Saving data')
#saveDictHist(folder, output_thome, Thome, Tgeo, headers_bins)
#saveDictHist(folder, output_twork, Twork, Tgeo, headers_bins)
#saveDictHist(folder, output_tdreal, Tdreal, Tgeo, headers_bins)      
print('Data Saved, finished')

#%% Modif considering modal transport       
Thome = mb.initFrCommDict(Tgeo, nbins)    
Twork = mb.initFrCommDict(Tgeo, nbins)
Tdreal = mb.initFrCommDict(Tgeo, nbins)      
        
# Computing Histograms per commune (work and home)
t0 = time.time()
print('Starting computing histograms considering modal ratio')
tlast = t0        
with open(folder + data_file, 'r') as datafile:
    data_reader = csv.reader(datafile, delimiter=',')
    headers = next(data_reader)
    print(headers)
    i = 0
    nflux = 0
    for rows in data_reader:
        # rows have [CommuneInit, CommuneEnd, Flux, distance]
        i += 1

        if rows[0] == rows[1]:
        # ie same commune  
            flux = int(np.ceil(float(rows[2]) * Modal[rows[0]][0]))
        else:
            flux = int(np.ceil(float(rows[2]) * Modal[rows[0]][1]))
            
        points = mb.distance2cities(flux, Tgeo[rows[0]][2], Tgeo[rows[1]][2], float(rows[3]))
        h = mb.hist(points, hini, hstep, nbins)
        Thome[rows[0]] = np.add(Thome[rows[0]], h)
        if not(Tgeo[rows[1]][5] == '99'):
            Twork[rows[1]] = np.add(Twork[rows[1]], h)
        Tdreal[rows[0]][min(int(float(rows[3]) // hstep), nbins-1)] += flux
        nflux += flux
        if i % 100000 == 0:
            tnew = time.time()
            print('Counter: ', i, 'Computed Flux', nflux)
            print('Time :', tnew-t0, 'deltaTime :', tnew - tlast)
            tlast = tnew
            print 
            
print('Finished computed histograms per commune')
print('Total rows = ', i, '; Total workers = ', nflux)
print('Total time = ', time.time()-t0, '; deltaTime =', time.time()-tlast)
print('Sum Thome: ', sum(sum(Thome[key] for key in Thome)))
print('Sum Twork: ', sum(sum(Twork[key] for key in Twork)))
print('Sum Tdreal: ', sum(sum(Tdreal[key] for key in Tdreal)))

#%%
output_thome = 'HistHomeModal.csv'       
output_twork = 'HistWorkModal.csv'
output_tdreal = 'HistDrealModal.csv'

headers_bins = ['bin' + str(i*hstep) + '_' + str((i+1)*hstep) for i in range(nbins)]

print('Saving data')
saveDictHist(folder, output_thome, Thome, Tgeo, headers_bins)
saveDictHist(folder, output_twork, Twork, Tgeo, headers_bins)
saveDictHist(folder, output_tdreal, Tdreal, Tgeo, headers_bins)      
print('Data Saved, finished')    
    

    