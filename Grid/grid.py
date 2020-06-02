# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:22:36 2020

@author: felip_001
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from time import time
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import Button


def dist_series_point(sx, sy, px, py):
    """ 
    """
    return (sx-px)*(sx-px) + (sy-py)*(sy-py)

def plot_lines(lines, ax=None, **plot_params):
    segs = []
    for i in lines.Shape:
        segs.append(i)
    if ax==None:
        f, ax = plt.subplots()
    if not 'color' in plot_params:
        plot_params['color'] = 'r'
    if not 'alpha' in plot_params:
        plot_params['alpha'] = 0.5 
    lc = LineCollection(segs, gid=list(lines.index), **plot_params)
    ax.add_collection(lc)
    ax.autoscale()
    return ax

def to_node(lines, node):
    """ Returns all lines connected to a given node"""
    return list(lines[((lines.node_i == node) | (lines.node_e == node))].index)

def unique_nodes(lines):
    """ Returns a list of unique nodes in a list of lines """
    return list(np.unique(list(lines.node_i.values) + list(lines.node_e.values)))

def new_nodes(lines, node):
    l = unique_nodes(lines)
    l.remove(node)
    return l

def connected(lines, node):
    """ Returns all lines directly or indirectly connected to a given departure node """
    nl = to_node(lines, node)
    if len(nl)>0:
        newnodes = new_nodes(lines.loc[nl], node)
        for nn in newnodes:
            nl += connected(lines.drop(nl), nn)
    return nl


def dist_to_node(lines, node, dist=None, maxd=80, name=None):
    """ Returns a pd.Series of with the min distance of nodes to a given Node
    """
    if dist is None:
        if name is None:
            name=node
        dist = pd.Series(index=unique_nodes(lines), dtype=float, name=name)
    if not dist[node] == dist[node]:
        dist[node] = 0 
    d0 = dist[node]
    # Lines coming to node 
    nl = to_node(lines, node) 
    if len(nl)>0:
        # Iterating over lines
        for l in nl:
            # Get other node and new tentative distance
            nout = lines.node_e[l] if lines.node_i[l] == node else lines.node_i[l]
            nd = lines.Length[l] + d0
            if nd < maxd*1000: #Max distance set at maxd km, useful to keep computing time reduced
                if not (dist[nout] <= nd):
                    # If tentative distance is less (or new), it updates the data
                    dist[nout] = nd
                    # Compute new distances from this new node, 
                    # this will modify the pd.Series dist inplace
                    dist_to_node(lines.drop(l), nout, dist=dist)
                    # dist = pd.DataFrame([dist,dist2]).T.min(axis=1)
    return dist

    
colors = ['r', 'g', 'b', 'gray', 'c', 'y', 'm', 'darkblue', 'purple', 'brown', 'maroon']


        
def get_ind_feeders(lines, n0, verbose=False):
    """ Returns a list of independent feeder connected to the Main SS (n0)"""
    # Lines connected to main node
    ls = to_node(lines, n0)
    # Sub-dataframe of lines, without the first lines connected to main node
    l_ = lines[lines.Connected].drop(ls)
    # New 'initial nodes'
    nn = new_nodes(lines.loc[ls], n0)
    feeder = pd.Series(index=lines.index, dtype=str)

    if verbose:
        print('Initial feeders = {}'.format(len(nn)))
    nfs = 0
    # For each initial feeder, comupte all lines connected to it
    for i, n in enumerate(nn):
        if verbose:
            print('\tFeeder {}'.format(i))
        # check if first line already is in one subset:
        l0 = to_node(l_, n)[0]
        if feeder[l0] == feeder[l0]:
            continue
        
        # Lines connected to node 'n'
        c = connected(l_, n)
        feeder[c] = 'F{:02d}'.format(nfs)
        nfs += 1
    # Add first lines to each feeder
    for l in ls:
        n = lines.node_i.loc[l] if lines.node_i.loc[l] != n0 else lines.node_e.loc[l]
        l2 = to_node(l_, n)[0]
        feeder[l] = feeder[l2]
            
    if verbose:
        print('Finished computing indepentent feeder, Total={}'.format(nfs))
    return feeder


def number_init_feeders(lines, n0):
     # Lines connected to main node
    ls = to_node(lines, n0)
    lsdf = lines.loc[ls]
    # List of feeders
    fs = lines.Feeder.unique()
    fs = np.sort([f for f in fs if f==f])
    print('Number of Initial feeders per connected feeder')
    for f in fs:
        print('{} : {}'.format(f, len(lsdf[lsdf.Feeder == f])))
#%%

class on_off_lines:
    
    def __init__(self, lines, n0, ax=None, ss=None, lv=None, geo=None, distances=None):
        self.lines = lines
        self.n0 = n0
        if (not ('Feeder' in lines.columns)):
            print('Computing independent feeders')
            self.lines['Feeder'] = get_ind_feeders(lines, n0, verbose=True)
        if ax==None:
            self.f, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.f = ax.figure
        self.ax.set_position([0.125,0.2,0.775,0.68])
        
        fs  = self.lines.Feeder.unique()
        self.feeders = np.sort([f for f in fs if f==f])
        self.ss = ss
        self.lv = lv
        self.geo = geo
        self.dist = distances
        
        
        # Add button to recompute feeders
        axbutton = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.buttonf = Button(axbutton, 'Recompute\nFeeders')
        self.buttonf.on_clicked(self.recompute)
        
        # Add button to (re)compute distances to nodes
        axbutton = plt.axes([0.25, 0.05, 0.2, 0.075])
        self.buttond = Button(axbutton, 'Recompute Distance\nTo Substations')
        self.buttond.on_clicked(self.recom_dist)
        
        # Add button to toggle aspect 
        axbutton = plt.axes([0.75, 0.905, 0.05, 0.05])
        self.buttona = Button(axbutton, 'Toggle\nAspect')
        self.buttona.on_clicked(self.aspect)
        
        # Add button to toggle on/off visibility of bt and geo:
        if lv is not None:
            axbuttonbt = plt.axes([0.55, 0.05, 0.1, 0.075])
            self.buttonlv = Button(axbuttonbt,'On/Off MV/LV')
            self.buttonlv.on_clicked(self.onoffbt)
        if geo is not None:
            axbuttongeo = plt.axes([0.4, 0.05, 0.1, 0.075])
            self.buttongeo = Button(axbuttongeo,'On/Off GeoShapes')
            self.buttongeo.on_clicked(self.onoffgeo)
            
        self.mainview = None
        self.currentview = None
        
        self.draw()
        
        self.cid = self.f.canvas.mpl_connect('pick_event', self.set_on_off)
        self.ncalls = 0
        self.lastevent = []
    
    def draw(self):
        for i, f in enumerate(self.feeders):
            plot_lines(self.lines[(self.lines.Feeder==f) & (self.lines.Connected)], self.ax, color=colors[int(i%len(colors))], picker=5, label=f) 
        # Not connected lines
        notconn = self.lines[self.lines.Connected == False]
        plot_lines(notconn, self.ax, color='k', linestyle=':', picker=5, label='Disconnected')
        # Connected but without feeder
        notassigned = self.lines[self.lines.Connected & self.lines.Feeder.isnull()]
        plot_lines(notassigned, self.ax, color='r', linestyle=':', picker=5, label='Without feeder')
        self.mainview = self.ax.axis()
        self.openlines = self.ax.add_collection(LineCollection(segments=[], gid=[], linestyle=':', color='k', picker=5))
        self.reconnected = self.ax.add_collection(LineCollection(segments=[], gid=[], linestyle='-.', color='darkgoldenrod', picker=5, label='Reconnected'))    
        
        if self.ss is not None:
            self.ax.plot(self.ss.x, self.ss.y, '*', color='purple', markersize=10, label='SS')
            for p in self.ss.index:
                self.ax.text(self.ss.x[p]+100, self.ss.y[p]+100, p)
        if self.lv is not None:
            self.lvartist = self.ax.plot(self.lv.x, self.lv.y, 'oy', markersize=5, label='MV/LV', alpha=0.5)[0]
        if self.geo is not None:
            axis = self.ax.axis()
            # plot polys
            
        
        self.f.legend()
        self.ax.set_title('Click on lines to switch on/off')
        if self.currentview is not None:
            self.ax.axis(self.currentview)
    
    def onoffbt(self, event=None):
        self.lvartist.set_visible(not self.lvartist.get_visible())
        
    def onoffgeo(self, event=None):
        pass
    
    def recompute(self, event=None):
        print('\nRecomputing independent feeders')
        #Getting currentview
        self.currentview = self.ax.axis()
        self.lines.Feeder = get_ind_feeders(self.lines, self.n0, verbose=True)
        fs  = self.lines.Feeder.unique()
        self.feeders = np.sort([f for f in fs if f==f])
        number_init_feeders(self.lines, self.n0)
        self.ax.clear()
        self.draw()
    
    def recom_dist(self, event=None):
        print('\nRecomputing distances to HV/MV substations')
        print('\tThis may take a while')
        df = pd.DataFrame()
        l = self.lines[self.lines.Connected][['Length','node_i','node_e']]
        for p in self.ss.index:
            print('\tComputing distance for {}'.format(p))
            df = pd.concat([df,dist_to_node(l, self.ss.node[p], name=p, maxd=50)], axis=1)
        self.dist = df
        print('Finished recomputing distances')
    
    def aspect(self, event=None):
        a = 'auto' if self.ax.get_aspect() == 'equal' else 'equal'
        self.ax.set_aspect(a)
        print('Aspect set to {}'.format(a))
        
    def draw_off(self, idline, artist):
        # Delete off line from current artist
        segs = artist.get_segments()
        gids = artist.get_gid()
        g = gids[idline]
        del segs[idline]
        del gids[idline]
        artist.set_segments(segs)
        artist.set_gid(gids)
        self.f.canvas.draw_idle()
        print('\nLine {}; Connected {}'.format(g, self.lines.Connected[g]))
        if not (self.dist is None):
            print('Line distances to Substations:')
            for p in self.dist:
                nodes = list(self.lines[['node_i', 'node_e']].loc[g])
                print('\t{}: {:.1f} km'.format(p, self.dist[p][nodes].mean()/1000))
    
    def draw_on(self, gid, artist):
        # Draw line from current artist
        segs = artist.get_segments()
        s = self.lines.Shape[gid]
        gids = artist.get_gid()
        segs.append(s)
        gids.append(gid)
        artist.set_segments(segs)
        artist.set_gid(gids)
        self.f.canvas.draw_idle()

        
    def set_on_off(self, event):
        if not self.lastevent==[]:
            if self.lastevent.mouseevent == event.mouseevent:
                return
        artist = event.artist
        ind = event.ind[0]
        line = artist.get_gid()[ind]
        onoff = not self.lines.Connected[line]
        self.lines.Connected.at[line] = onoff
        self.draw_off(ind, artist)
        art2 = self.reconnected if onoff else self.openlines
        self.draw_on(line, art2)
        self.lastevent = event

        

# off = on_off_lines(hta, n0, ss=ps, lv=bt)

