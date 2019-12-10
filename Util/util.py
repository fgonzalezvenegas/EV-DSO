# -*- coding: utf-8 -*-
""" Useful functions for pandas treatement, or pyplot plotting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe
#import assign_ss_modif as ass_ss

def plot_polygons(polys, ax='', **kwargs):
    """ Plot a list of polygons into the axis ax
    kwargs as arguments for PolygonCollection
    """
    if ax == '':
        f, ax = plt.subplots()
    collection = PatchCollection(polys, **kwargs)
    ax.add_collection(collection)
    ax.autoscale()
    return ax
    
def plot_segments(segments, ax='', **kwargs):
    if ax=='':
        f, ax = plt.subplots()
    for s in segments:
        ax.plot(np.array(s)[:,0],np.array(s)[:,1],**kwargs)
    return ax
    
    
def fix_wrong_encoding_str(pdSeries):
    """
    """
    # é, è, ê, ë, É, È
    out = pdSeries.apply(lambda x: x.replace('Ã©', 'é').replace('Ã¨', 'è').replace('Ãª', 'ê').replace('Ã‰', 'É').replace('Ã«','ë').replace('Ãˆ','È'))
    # Î, î, ï
    out = out.apply(lambda x: x.replace('ÃŽ', 'Î').replace('Ã®', 'î').replace('Ã¯', 'ï'))
    #ÿ
    out = out.apply(lambda x: x.replace('Ã½', 'ÿ'))
    # ç
    out = out.apply(lambda x: x.replace('Ã§', 'ç'))
    # ô
    out = out.apply(lambda x: x.replace('Ã´', 'ô'))
    # û
    out = out.apply(lambda x: x.replace('Ã»', 'û').replace('Ã¼', 'ü'))
    # â, à
    out = out.apply(lambda x: x.replace('Ã¢', 'â').replace('Ã\xa0', 'à'))
    
    return out

def do_polygons(df, plot=False):
    """ Do polygons from df or pdSeries
    """
    polygons = {c: [ptc.Polygon(p) for p in df.Polygon[c] if len(p) > 1] for c in df.index}
    if plot:
        plot_polygons([p for pp in polygons.values() for p in pp])
    return polygons

def compute_load_from_ss(energydata, profiledata, ss):
    """Returns the load profile for the substation ss, 
    where Substation data is stored in SS DataFrame (namely communes assigned) 
    and load data in load_profiles and load_by_comm
    """
    energy_types = ['Conso_RES', 'Conso_PRO', 'Conso_Agriculture', 'Conso_Industrie', 'Conso_Tertiaire']
    profiles = ['RES', 'PRO', 'Agriculture', 'Industrie', 'Tertiaire']
    
    # These factors are the total consumption for a year for all the components of energydata df 
    # associated to the substation ss
    factors = energydata[energydata.SS == ss][energy_types].sum()
    mwhy_to_mw = 1/8760 
    factors.index = profiles
    #print(factors)
    
    return (profiledata[profiles] * factors * mwhy_to_mw).sum(axis=1)

def aspect_carte_france(ax, title="", palette='',
                        ranges='', wbin_palette=15, label_middle='<d<', label_end='',
                        cns='France', latlons='', delta_cns=0.2):
    if palette=='':
        palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
    if ranges=='':
        ranges=[i for i in range(len(palette)+1)*wbin_palette]
    if cns=='France':
        cns = ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Bordeaux', 'Nantes', 'Lille', 'Rennes']
        latlons = [[2.3424567382662334, 48.859626443036575],
                 [5.3863214053108095, 43.300743046351528],
                 [4.8363757561790415, 45.771993345448962],
                 [1.4194663657069806, 43.58313856938689],
                 [-0.58251346635799961, 44.856834056176488],
                 [-1.5448118357936005, 47.227505954238453],
                 [2.98834511866088, 50.651686273910592],
                 [-1.6966383042920521, 48.083113659214533]]
    if cns == 'idf':
        cns = ['Paris', 'Versailles', 'Évry', 'Meaux', 'Bordeaux', 'Nemours', 'Cergy', 'Étampes', 'Provins']
        latlons = [[2.3424567382662334, 48.859626443036575],
                 [2.131139599462395, 48.813017693582793],
                 [2.4419262056107969, 48.632343164682723],
                 [2.9197438849044732, 48.952766254606502],
                 [-0.58251346635799961, 44.856834056176488],
                 [2.7070194793010449, 48.26850934641633],
                 [2.0698102422679203, 49.037687672577924],
                 [2.1386667699798143, 48.435164427848107],
                 [3.2930400953107446, 48.544799959855652]]
        delta_cns=0
        
    ax.set_title(title)
    ax.autoscale()
    ax.set_aspect('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Do labels
    for i in range(len(palette)):
        ax.plot(1,1,'s', color=palette[i], 
                label=str(ranges[i])+label_middle+str(ranges[i+1])+label_end)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.legend(loc=3)
    for i in range(len(cns)):
        ax.text(latlons[i][0],latlons[i][1]+delta_cns, cns[i], ha='center',
           path_effects=[pe.withStroke(linewidth=2, foreground='w')])
    