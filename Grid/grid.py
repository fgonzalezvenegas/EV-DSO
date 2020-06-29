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
import util
import datetime
import pandapower as pp
import networkx as nx

def dist_series_point(sx, sy, px, py):
    """ 
    """
    return (sx-px)*(sx-px) + (sy-py)*(sy-py)

def plot_lines(lines, col='Shape', ax=None, **plot_params):
    segs = []
    for i in lines[col]:
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

def dist_to_node_nx(lines, n0, name='name'):
    """ Returns a pd.Series of with the min distance of nodes to a given Node
    Uses networkx, djistra min dist, super fast!
    """
    # Creates nx graph
    g = nx.Graph()
    for l,t in lines.iterrows():
        g.add_edge(t.node_i, t.node_e, weight=t.Length)
    d = pd.Series(nx.single_source_dijkstra_path_length(g, n0), name=name)
    return d.sort_index()
        
    
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
        l0  = to_node(l_, n)[0]
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

def get_ind_feeders_nx(lines, n0, verbose=False):
    """ Returns a list of independent feeder connected to the Main SS (n0)"""
    # Create nx graph
    g = nx.Graph()
    for l,t in lines.iterrows():
        g.add_edge(t.node_i, t.node_e, weight=t.Length)
    # Remove initial node
    g.remove_node(n0)
    cc = list(nx.connected_components(g))
    # Putting it in pd.Series
    feeder = pd.Series(index=lines.index)
    nfs = 0
    for fs in cc:
        lls = lines[(lines.node_i.isin(fs)) | (lines.node_e.isin(fs))].index
        feeder[lls] = 'F{:02d}'.format(nfs)
        nfs += 1
    if verbose:
        print('Initial feeders = {}'.format(len(to_node(lines,n0))))      
        print('Finished computing indepentent feeder, Total={}'.format(len(cc)))
    return feeder

def number_init_feeders(lines, n0):
     # Lines connected to main node
    ls = to_node(lines, n0)
    lsdf = lines.loc[ls]
    # List of feeders
    fs = lines.Feeder.dropna().unique()
    fs.sort()
    print('Number of Initial feeders per connected feeder and total length [km]')
    print('Feeder:\t#Init\tLength [km]')
    for f in fs:
        print('{:6}:\t{:4}\t{:8.2f}'.format(f, len(lsdf[lsdf.Feeder == f]), lines[lines.Feeder == f].Length.sum()/1000))

def get_farther(point, points):
    idx = ((point.xGPS-points.xGPS)**2+(point.yGPS-points.yGPS)**2).idxmax()
    return points.xGPS[idx], points.yGPS[idx]

def get_coord_node(lines, node, GPS=True):
    u = lines[lines.node_i == node]
    if len(u) == 0:
        u = lines[lines.node_e == node]
    if len(u) == 0:
        return
    cols = ['xGPS', 'yGPS'] if GPS else ['x', 'y']
    return u[cols].iloc[0]

def rename_nodes(nodes, n0, lines, lv, ss):
    """ Rename the nodes according to Feeder appartenance and distance to main node
    New index is just ascending numeric
    Adds new column 'name' with a meaningful name
    Updates relations to lines node_i, node_e; lv and ss
    """
    nodes['d'] = ((nodes.xGPS[n0]-nodes.xGPS)**2+(nodes.yGPS[n0]-nodes.yGPS)**2)
    nodes['Feeder'] = ''

    for f in lines.Feeder.unique():
        nodes.Feeder[unique_nodes(lines[lines.Feeder == f])] = f
    nodes.Feeder[n0] = '0SS'
    # Sorting nodes
    nodes.sort_values(['Feeder', 'd'], inplace=True)
    # Creating new index
    nodes.reset_index(inplace=True)
    # Get relationship old index-new index
    old_new = pd.Series(index=nodes['index'], data=nodes.index)
    
    #Update relationship
    lines.node_i = old_new[lines.node_i].values
    lines.node_e = old_new[lines.node_e].values
    lv.node = old_new[lv.node].values
    ss.node = old_new[ss.node].values 
    
    #Adds new name
    nodes['name'] = ['N'+str(i)+'_'+nodes.Feeder[i] for i in nodes.index]
    nodes.drop('index', axis=1, inplace=True)
    
def rename_lines(lines, n0):
    """ Rename lines according to Feeder and distance to main node
    New index is just ascending numeric
    Adds new column 'name' with a meaningful name
    """
    # Computing distance to main node
    l = lines.loc[to_node(lines,n0)[0]]
    x0 = l.x1GPS if l.node_i == n0 else l.x2GPS
    y0 = l.y1GPS if l.node_i == n0 else l.y2GPS
    d1 = ((x0-lines.x1GPS)**2+(y0-lines.y1GPS)**2)
    d2 = ((x0-lines.x2GPS)**2+(y0-lines.y2GPS)**2)
    lines['d'] = pd.DataFrame([d1,d2]).min()
    # sorting lines
    lines.sort_values(['Feeder', 'd'], inplace=True, ascending=True)
    # Creating new index
    lines.reset_index(inplace=True, drop=True)
    # Renaming lines
    #Adds new name
    lines['name'] = ['L'+str(i)+'_'+lines.Feeder[i] for i in lines.index]
    
    
    
#%% Reducing number of nodes section
        
def join_lines(lines, l1, l2):
    """ Joins line l2 to l1
    """
    nl = dict(lines.loc[l1])
    ol = dict(lines.loc[l2])
    # id common node
    if nl['node_i'] in [ol['node_i'], ol['node_e']]:
        common_node = nl['node_i']
    elif nl['node_e'] in [ol['node_i'], ol['node_e']]:
        common_node = nl['node_e']
    else:
        print('no common node')
        return
    # id direction of join
    if nl['node_i'] == common_node:
        nl['x1'] = nl['x2']
        nl['y1'] = nl['y2']
        nl['x1GPS'] = nl['x2GPS']
        nl['y1GPS'] = nl['y2GPS']
        nl['node_i'] = nl['node_e']
        nl['Shape'] = nl['Shape'][::-1]
        nl['ShapeGPS'] = nl['ShapeGPS'][::-1]
    if ol['node_i'] == common_node:
        nl['x2'] = ol['x2']
        nl['y2'] = ol['y2']
        nl['x2GPS'] = ol['x2GPS']
        nl['y2GPS'] = ol['y2GPS']
        nl['node_e'] = ol['node_e']
        nl['Shape'] += ol['Shape']
        nl['ShapeGPS'] += ol['ShapeGPS']
    else:
        nl['x2'] = ol['x1']
        nl['y2'] = ol['y1']
        nl['x2GPS'] = ol['x1GPS']
        nl['y2GPS'] = ol['y1GPS']
        nl['node_e'] = ol['node_i']
        nl['Shape'] += ol['Shape'][::-1]
        nl['ShapeGPS'] += ol['ShapeGPS'][::-1]
    nl['x'] = np.mean([xy[0] for xy in nl['Shape']])
    nl['y'] = np.mean([xy[1] for xy in nl['Shape']])
    nl['xGPS'] = np.mean([xy[0] for xy in nl['ShapeGPS']])
    nl['yGPS'] = np.mean([xy[1] for xy in nl['ShapeGPS']])
    nl['Length'] += ol['Length']
    
    lines.loc[l1] = list(nl.values())
    lines.drop(l2, inplace=True)
    
def count_elements(lines, nodes, lv=None, ss=None):
    """ Returns the number of elements arriving at each node
    """
    oh = lines[lines.Type == 'Overhead']
    ug = lines[lines.Type == 'Underground']
    
    cols = ['nlines_oh', 'nlines_ug']
    if not lv is None:
        cols += ['nlv']
    if not ss is None:
        cols += ['nss']

    # Counting elements arriving at each node
    nelements = pd.DataFrame(index=nodes.index, columns=cols).fillna(0).sort_index()
    nelements.nlines_oh = nelements.nlines_oh.add(oh.node_i.value_counts(), fill_value=0).add(oh.node_e.value_counts(), fill_value=0)
    nelements.nlines_ug = nelements.nlines_ug.add(ug.node_i.value_counts(), fill_value=0).add(ug.node_e.value_counts(), fill_value=0)
    if not lv is None:
        nelements.nlv = nelements.nlv.add(lv.node.value_counts(), fill_value=0)
    if not ss is None:
        nelements.nss = nelements.nss.add(ss.node.value_counts(), fill_value=0)
    
    return nelements

def remove_paralell_lines(lines, nodes):
    """ Removes lines that start and end at the same node
    """
    i=0
    print('Checking for paralell lines')
    for n in nodes.index:
        ls = to_node(lines, n)
        other_nodes = unique_nodes(lines.loc[ls])
        if len(other_nodes) < 1 + len(ls):
            #get paralell lines
            for nn in other_nodes:
                if nn == n:
                    continue
                subfr = lines.loc[ls]
                nl_in = (subfr.node_i == nn).sum() + (subfr.node_e == nn).sum()
                if nl_in > 1:
                    dl = subfr[(subfr.node_i == nn) | (subfr.node_e == nn)].index[1:]
                    lines.drop(dl, inplace=True)
                    print('\tRemoved the lines connected to nodes {} and {}\n\t\t{}'.format(n, nn, list(dl)))
        i +=1
        if i%500==0:
            print('\t{}'.format(i))
        
def reduce_nodes(lines, nodes, lv, ss):
    """ It reduces the number of nodes in the grid, 
    removing the nodes that only link two lines of same type
    """
    # Eliminate paralell lines
    remove_paralell_lines(lines, nodes)
     # Counting elements arriving at each node
    nelements = count_elements(lines, nodes, lv, ss)
    # Selecting nodes to be reduced, those where there are no loads (lv) and only two lines of same kind arrive
    to_red = nelements[(((nelements.nlines_ug == 2) & (nelements.nlines_oh == 0)) |
                        ((nelements.nlines_ug == 0) & (nelements.nlines_oh == 2)) ) & 
                       (nelements.nlv == 0) & (nelements.nss == 0)]
    geocols = ['GeoPoint', 
               'Shape', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 
               'ShapeGPS', 'xGPS', 'yGPS', 'x1GPS', 'y1GPS', 'x2GPS', 'y2GPS']
    for c in geocols:
        if type(lines[c][0]) == str:
            lines[c] = lines[c].apply(eval)
    print('Starting to remove {} nodes'.format(len(to_red)))
    # Iterating over nodes to remove them from dataFrames
    for n in to_red.index:
        # Lines to join
        ls = to_node(lines, n)
        # join lines & remove old line
        join_lines(lines, ls[0], ls[1])        
    # remove old nodes
    nodes.drop(to_red.index)
    print('Done, removed {} out of {} nodes from data'.format(len(to_red), len(nelements)))    

#%% Assign load section

def assign_poly(point, geo, dt=0.05, notdt=None):
    """ Returns the geo.Polygon that contains the point
    It searches the polygons within +- dt North/South - Est/West
    If notdt, then it searches within the polygons within +-notdt (NS/EW), exluding the polygons within (+-dt)
    """
    geos = list(geo[(geo.xGPS < point.xGPS + dt) & (geo.xGPS > point.xGPS - dt) &
                    (geo.yGPS < point.yGPS + dt) & (geo.yGPS > point.yGPS - dt)].index)
    if not notdt is None:
        notgeos = list(geo[(geo.xGPS < point.xGPS + notdt) & (geo.xGPS > point.xGPS - notdt) &
                           (geo.yGPS < point.yGPS + notdt) & (geo.yGPS > point.yGPS - notdt)].index)
        for g in notgeos:
            geos.remove(g)
            
    for g in geos:
        for p in geo.Polygon.loc[g]:
            if p.contains_point((point.xGPS, point.yGPS)):
                return g
    
def assign_polys(lv, geo, dt=0.05):
    """ Looks which geo polygon containts each LV point
    geo needs to have a Polygon Series, which is [matplotlib.Patches.Polygon] type
    """
    geo_lv = {}
    i = 0
    print('Assigning Geographic zone to each point')
    for load in lv.index:
        point = lv[['xGPS', 'yGPS']].loc[load]
        g = assign_poly(point, geo, dt)
        if g is None:
            g = assign_poly(point, geo, dt*2, notdt=dt)
        geo_lv[load] = g
        i += 1
        if i%50 == 0:
            print('\t{}'.format(i))
    lv['Geo'] = pd.Series(geo_lv)
    if 'Name' in geo.columns:
        names = geo.Name.loc[lv.Geo]
        names.index = lv.index
        lv['Geo_Name'] = names
        
def equal_total_load(lv, consos, lv_per_geo=None):
    """ Assigns a repartition of PRO, RES, Ind, Tertiaire, Agri to each node
    Option 1: Equal for all LV nodes
    """
    cols = ['Conso_RES', 'Conso_PRO', 'Conso_Industrie', 'Conso_Tertiaire', 'Conso_Agriculture']
    if lv_per_geo is None:
        lv_per_geo = lv.Geo.value_counts()
    load = consos[cols].loc[lv.Geo].sum(axis=1)
    load.index = lv.index
    div = lv_per_geo[lv.Geo]
    div.index = lv.index
    
    return load/div

#%% Assign tech data

def compute_cum_load(lines, lv, n0, d0=None):
    """ Computes cumulative load at each line/node
    """
    # distance to initial node
    if d0 is None:
        print('Computing distances to initial node')
        d0 = dist_to_node_nx(lines, n0)  
        print('Done!')
    d0 = d0.sort_values(ascending=False)
    #Cumulative load at given node and line
    cumload_n = pd.Series(index=d0.index)        
    cumload_l = pd.Series(index=lines.index)
    # Iterates over nodes, from farthest to closest
    print('Iterating over nodes')
    for n in d0.index:
        load = lv[lv.node==n].Annual_Load_MWh.sum()
        # get lines connected to node
        ls = to_node(lines, n)
        for l in ls:
            # Check if line goes downstream or upstream
            n2 = lines.node_e[l] if lines.node_i[l] == n else lines.node_i[l]
            if d0[n2] > d0[n]:
                # line goes downstream. 
                # Compute cumulative load
                cumload_l[l] = cumload_n[n2]
                load += cumload_n[n2]            
        cumload_n[n] = load
    print('Done!')
    return cumload_n, cumload_l

def assign_tech_line(lines, lv, n0, tech, peak_factor=0.000207, d0=None, 
                     voltage=20, security_margin_power=1.5, 
                     tanphi=0.3, max_dv_per_feeder=0.04):
    """ Algorithm to assign the tech data to each line section
   It computes the peak current that transits per each line, given by:
       Pmax_line = Cumload_year * peak_factor [MW]
       Imax_line = Pmax_line / (voltage_LL * sqrt(3))
    The cable type should be the smallest that support the max current:
        cable[line] = argmin(Imax[cable])
        st. Imax[cable] >= Imax_line
    It computes also the maximum kilometric voltage drop per feeder as:
        As first approximation, this method assumes that the voltage profile is linear, 
            i.e. the voltage drop per km is limited to a given p.u. value all over the length of the feeder
        max_dV_per_feeder_km [pu/km] = max_dv_per_feeder [pu] / length_feeder [km]
        max_dV_per_line [pu/km] = max_dV_per_feeder
        Equivalent voltage drop per km per MW, per cable type. R,X in Ohm/km, V in kV
        dV_pu_cable [pu/(km.MW)] = (R + X*tan(phi)) / (V^2)
    The cable type should be the smallest (has the largest V drop) that allows the kilometric voltage drop to be under the limit
        cable[line] = argmax(dV_km[cable])
        st. dV_kmMW[cable] <= dV_km_feeder[line] / Pmax[line]
    """
    fs = lines.Feeder.dropna().unique()
    fs.sort()
    if d0 is None:
        print('Computing distances to initial node')
        d0 = dist_to_node_nx(lines, n0)  
        print('\tDone!')
    # max length per feeder
    length_feeder = pd.Series({f: d0[unique_nodes(lines[lines.Feeder==f])].max()/1000 for f in fs})
    # Peak loads per lines (Annual Energy)
    print('Computing cumulative load per line')
    _, annual_load = compute_cum_load(lines, lv, n0, d0=d0)
    peak_load = annual_load * peak_factor
    # Eauivalent peak current, considering security margin, per line
    i_max_security =  peak_load * (1/np.cos(np.arctan(tanphi))) / (voltage * np.sqrt(3)) * security_margin_power * 1e3
    
    # Equivalent voltage drop per cable type [pu/km.MW]
    dV_kmMW_cable  = (tech.R + tech.X * tanphi)/voltage**2
    # Max voltage drop per line per km, considering linear drop
    max_dV_km_per_line = max_dv_per_feeder / length_feeder[lines.Feeder] 
    max_dV_km_per_line.index = lines.index 
    max_dV_kmMW_per_line = max_dV_km_per_line / peak_load
    
    # Selecting cable based on the Imax_cable>Imax_line and dV_max_cable<dV_max_line
    cable = pd.Series(index=lines.index)
    # sort cable types from smallest to largest
    tech.sort_values('Section', inplace=True, ascending=True)
    for type_l in lines.Type.unique():
        for c in tech[tech.Type == type_l].index:
            # Lines to be tested for both conditions
            l = lines[(lines.Type == type_l) & (cable.isnull())].index
            if len(l)==0:
                break
            # lines that comply with I and V conditions
            accepted_cable = ((i_max_security[l] < tech.Imax[c]) & (max_dV_kmMW_per_line[l] > dV_kmMW_cable[c]))
            accepted_cable = accepted_cable[accepted_cable].index
            cable[accepted_cable] = c
        # If there is no conductor that meets both criteria, select biggest available
        # Lines to be tested for both conditions
        l = lines[(lines.Type == type_l) & (cable.isnull())].index
        if len(l) > 0:
            cable[l] = c
    return cable
import pandapower.topology as ppt
ppt.calc_distance_to_bus
            
def create_pp_grid(nodes, lines, tech, lv, n0, 
                   hv=True, ntrafos_hv=2, vn_kv=20,
                   tanphi=0.3, verbose=True):
    """
    """
    if verbose:
        print('Starting!')
    # 0- empty grid
    net = pp.create_empty_network()
    # 1- std_types
    if verbose:
        print('\tTech types')
    for i, t in tech.iterrows():
        # i is ID of tech, t is tech data
        data = dict(c_nf_per_km=t.C,
                    r_ohm_per_km=t.R,
                    x_ohm_per_km=t.X,
                    max_i_ka=t.Imax/1000,
                    q_mm=t.Section,
                    type='oh' if t.Type == 'Overhead' else 'cs')
        pp.create_std_type(net, name=i, data=data, element='line')
    # 2 - Create buses
    if verbose:
        print('\tBuses')
    for b, t in nodes.iterrows():
        pp.create_bus(net, vn_kv=vn_kv, 
                      name=t['name'], index=b, geodata=t.ShapeGPS, type='b',
                      zone=t.Feeder, in_service=True)
    # 3- Create lines
    if verbose:
        print('\tLines')
    for  l, t in lines.iterrows():
        pp.create_line(net, from_bus=t.node_i, to_bus=t.node_e,
                       length_km=t.Length/1000, std_type=t.Conductor, name=t['name'],
                       geodata=t.ShapeGPS)
    net.line['Feeder'] = lines.Feeder
    # 4- Create loads
    if verbose:
        print('\tLoads')
    for l, t in lv.iterrows():
        pp.create_load(net, bus=t.node,  p_mw=t.Pmax_MW, q_mvar=t.Pmax_MW * tanphi, name=t.Geo)

    # Adding external grid
    if verbose:
        print('\tExt Grid')
    if hv:
        # If HV, then add extra bus for HV and add trafo
        b0 = pp.create_bus(net, vn_kv=110, geodata=nodes.ShapeGPS[n0], name='HV_SS')
        # Adding HV-MV trafo (n x 40MW trafos)
        pp.create_transformer(net, hv_bus=b0, lv_bus=n0, 
                              std_type='40 MVA 110/20 kV', name='TrafoSS', parallel=ntrafos_hv)
    else:
        b0 = n0
    pp.create_ext_grid(net, bus=b0)
    
    if verbose:
        print('Finished!')
    return net
    

#%% main class to draw and do everything
colors = ['r', 'g', 'b', 'gray', 'c', 'y', 'm', 'darkblue', 'purple', 'brown', 
          'maroon', 'olive', 'deeppink', 'blueviolet', 'darkturquoise', 'darkorange']
colors_tech = {'Underground' : ['maroon', 'red', 'orangered', 'salmon', 'khaki'],
               'Overhead' : ['midnightblue', 'mediumblue', 'slateblue', 'cornflowerblue', 'skyblue']}        
class on_off_lines:
    
    def __init__(self, lines, n0, nodes=None, ax=None, ss=None, 
                 lv=None, geo=None, distances=None, GPS=False, tech=None,
                 profiles=None,
                 outputfolder=''):
        self.lines = lines
        self.n0 = n0
        if (not ('Feeder' in lines.columns)):
            print('Computing independent feeders')
            self.lines['Feeder'] = get_ind_feeders_nx(lines, n0, verbose=True)
        if ax==None:
            self.f, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.f = ax.figure
            # Setting figure as full screen, works in windows 
            #(might need to change plt manager https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python)

        self.ax.set_position([0.125,0.2,0.775,0.68])
        plt.get_current_fig_manager().window.showMaximized()
        
        fs  = self.lines.Feeder.unique()
        self.feeders = np.sort([f for f in fs if f==f])
        self.ss = ss
        self.lv = lv
        self.geo = geo
        self.dist = distances
        self.tech = tech        # Line types tech data
        self.outputfolder = outputfolder
        if not self.tech is None:
            self.tech.sort_values('Section', ascending=False, inplace=True)
            
        if GPS:
            self.col = 'ShapeGPS'
            self.x = 'xGPS'
            self.y = 'yGPS'
            self.dtext = 0.003
        else:
            self.col = 'Shape'
            self.x = 'x'
            self.y = 'y'
            self.dtext = 100
        
        if nodes is None:
            print('No nodes given, computing from lines')
            self.nodes = pd.DataFrame(index=unique_nodes(self.lines), columns=[self.x, self.y])
            for n in self.nodes.index:
                self.nodes.loc[n] = get_coord_node(self.lines, n, GPS=GPS)
            print('\tDone!')
        else: 
            self.nodes = nodes
            
        self.do_buttons()
        self.line_view = 'Feeder'
            
        
        self.mainview = None
        self.currentview = None
        
        self.draw()
        
        self.cid = self.f.canvas.mpl_connect('pick_event', self.set_on_off)
        self.ncalls = 0
        self.lastevent = []
    
    def do_buttons(self):
        # Add button to (re)compute distances to nodes
        axbutton = plt.axes([0.05, 0.05, 0.18, 0.075])
        self.buttond = Button(axbutton, 'Recompute Distance\nTo Substations')
        self.buttond.on_clicked(self.recom_dist)
        
        # Add button to recompute feeders
        axbutton = plt.axes([0.3, 0.05, 0.18, 0.075])
        self.buttonf = Button(axbutton, 'Recompute\nFeeders')
        self.buttonf.on_clicked(self.recompute_feeders)
        
        # Add button to reduce data
        axbutton = plt.axes([0.55, 0.05, 0.18, 0.075])
        self.buttonred = Button(axbutton, 'Reduce data and\n compute tech')
        self.buttonred.on_clicked(self.reduce_compute_tech)
        
        # Add button to save data
        axbutton = plt.axes([0.8, 0.05, 0.18, 0.075])
        self.buttonsave = Button(axbutton, 'Save current data')
        self.buttonsave.on_clicked(self.save_data)
        
        # Add button to toggle aspect 
        axbutton = plt.axes([0.75, 0.905, 0.05, 0.05])
        self.buttona = Button(axbutton, 'Toggle\nAspect')
        self.buttona.on_clicked(self.aspect)
        
        # Add button to toggle on/off visibility of bt and geo:
        if self.lv is not None:
            axbuttonbt = plt.axes([0.15, 0.905, 0.1, 0.05])
            self.buttonlv = Button(axbuttonbt,'On/Off MV/LV')
            self.buttonlv.on_clicked(self.onoffbt)
        if self.geo is not None:
            axbuttongeo = plt.axes([0.30, 0.905, 0.1, 0.05])
            self.buttongeo = Button(axbuttongeo,'On/Off GeoShapes')
            self.buttongeo.on_clicked(self.onoffgeo)
            
         # Add button to plot according to tech data
        axbutton = plt.axes([0.45, 0.905, 0.1, 0.05])
        self.buttonlt = Button(axbutton, 'Plot line type\nPlot feeder')
        self.buttonlt.on_clicked(self.plot_linetype)
    
    def redraw(self):
        """ Re open a new figure and draw everything
        """ 
        self.f, self.ax = plt.subplots()
        self.ax.set_position([0.125,0.2,0.775,0.68])
        plt.get_current_fig_manager().window.showMaximized()
        
        self.do_buttons()
        
        self.draw()
        
        
    def draw(self):
        """ Draw  the grid
        """        
        if self.line_view == 'Feeder':
            # Better way might be to just turn on/off these artists
            for i, f in enumerate(self.feeders):
                plot_lines(self.lines[(self.lines.Feeder==f) & (self.lines.Connected) & (self.lines.Type == 'Underground')], 
                                      col=self.col, ax=self.ax, 
                                      color=colors[int(i%len(colors))], picker=5, label=f, linewidth=2) 
                
                plot_lines(self.lines[(self.lines.Feeder==f) & (self.lines.Connected) & (self.lines.Type == 'Overhead')], 
                                      col=self.col, ax=self.ax, 
                                      color=colors[int(i%len(colors))], picker=5, linewidth=1)
                px, py = get_farther(self.ss[self.ss.node == self.n0].iloc[0], self.lines[self.lines.Feeder == f])
                self.ax.text(px + self.dtext, py + self.dtext, f)
                # Plotting lines according to tech data
        if self.line_view == 'LineType':
            for lt in colors_tech.keys():
                for i, c in enumerate(self.tech[self.tech.Type == lt].index):
                    plot_lines(self.lines[self.lines.Conductor == c], 
                               col=self.col, ax=self.ax, 
                               color=colors_tech[lt][int(i%len(colors_tech[lt]))], picker=5, label=c, linewidth=self.tech.Section[c]/50) 
        # Not connected lines
        notconn = self.lines[self.lines.Connected == False]
        plot_lines(notconn, col=self.col, ax=self.ax, 
                   color='k', linestyle=':', picker=5, label='Disconnected')
        # Connected but without feeder
        notassigned = self.lines[self.lines.Connected & self.lines.Feeder.isnull()]
        plot_lines(notassigned, col=self.col, ax=self.ax, 
                   color='r', linestyle=':', picker=5, label='Without feeder')
        self.mainview = self.ax.axis()
        self.openlines = self.ax.add_collection(LineCollection(segments=[], gid=[], linestyle=':', color='k', picker=5))
        self.reconnected = self.ax.add_collection(LineCollection(segments=[], gid=[], linestyle='-.', color='darkgoldenrod', picker=5, label='Reconnected'))    
        
        if self.ss is not None:
            self.ax.plot(self.ss[self.x], self.ss[self.y], '*', color='purple', markersize=10, label='SS')
            for p in self.ss.index:
                self.ax.text(self.ss[self.x][p]+self.dtext, self.ss[self.y][p]+self.dtext, p)
        if self.lv is not None:
            self.lvartist = self.ax.plot(self.lv[self.x], self.lv[self.y], 'oy', markersize=5, label='MV/LV', alpha=0.5)[0]
        if self.geo is not None:
            axis = self.ax.axis()
            self.geo = self.geo[(axis[0]<self.geo[self.x]) & (self.geo[self.x]<axis[1]) & 
                             (axis[2]<self.geo[self.y]) & (self.geo[self.y]<axis[3])]
            # plot polys
            collection = PatchCollection([p[0] for p in self.geo.Polygon], 
                                         facecolor='none', edgecolor='k', linestyle='--', linewidth=0.8, alpha=0.8)
            self.geoartist = self.ax.add_collection(collection)
            if 'Name' in self.geo:
                self.geonames = [self.ax.text(self.geo[self.x][g], self.geo[self.y][g], self.geo.Name[g],
                                              horizontalalignment='center') for g in self.geo.index]
            
        if len(self.f.legends) > 0:
            self.f.legends = [] # Remove existing legend
            plt.plot() # To force drawing of new legend
        self.f.legend()
        self.ax.set_title('Click on lines to switch on/off')
        if self.currentview is not None:
            self.ax.axis(self.currentview)
        self.ax.set_aspect('equal')

    
    def onoffbt(self, event=None):
        self.lvartist.set_visible(not self.lvartist.get_visible())
        
    def onoffgeo(self, event=None):
        self.geoartist.set_visible(not self.geoartist.get_visible())
        for gn in self.geonames:
            gn.set_visible(not gn.get_visible())
    
    def recompute_feeders(self, event=None):
        print('\nRecomputing independent feeders')
        #Getting currentview
        self.currentview = self.ax.axis()
        self.lines.Feeder = get_ind_feeders_nx(self.lines, self.n0, verbose=True)
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
            df = pd.concat([df,dist_to_node_nx(l, self.ss.node[p], name=p)], axis=1)
        self.dist = df
        print('Finished recomputing distances')
    
    def aspect(self, event=None):
        a = 'auto' if self.ax.get_aspect() == 'equal' else 'equal'
        self.ax.set_aspect(a)
        print('Aspect set to {}'.format(a))
    
    def plot_linetype(self, event=None):

        if (not ('Conductor' in self.lines.columns)) | (self.tech is None):
            print('No technical line data. Check input data')
            return
        
        self.line_view = 'LineType' if self.line_view == 'Feeder' else 'Feeder'
        self.ax.clear()
        self.draw()
    
    def reduce_compute_tech(self, event=None):
        """ Reduces data to only connected to main node
        Computes tech data (Conductor type) for each line
        """
        if (self.tech is None):
            print('No technical line data. Check input data')
            return
        # Reducing data to only lines and nodes connected to Main node (n0)
        self.lines = self.lines[~self.lines.Feeder.isnull()]
        ns = unique_nodes(self.lines)
        self.nodes = self.nodes.loc[ns]
        self.lv = self.lv[self.lv.node.isin(self.nodes.index)]
        self.ss = self.ss[self.ss.node.isin(self.nodes.index)]
        
        self.lines['Conductor'] = assign_tech_line(self.lines, self.lv, self.n0, self.tech)
        self.tech.sort_values('Section', inplace=True, ascending=False)
        
        self.line_view = 'LineType'
        self.ax.clear()
        self.draw()
        
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
        s = self.lines[self.col][gid]
        gids = artist.get_gid()
        segs.append(s)
        gids.append(gid)
        artist.set_segments(segs)
        artist.set_gid(gids)
        self.f.canvas.draw_idle()

        
    def set_on_off(self, event):
        # Macro to call functions to draw clicked lines as Connected/Disconnected lines
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
        
    def save_data(self, event=None):
        now = datetime.datetime.now()
        fn = r'Data_{}_{}'.format(self.ss.index[0], str(now.date()))
        if len(self.outputfolder) > 0:
            if not self.outputfolder[-2:] == r'\\':
                self.outputfolder += r'\\'
        util.create_folder(self.outputfolder + fn)
        print('Saving data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(self.lines), len(self.lv), len(self.nodes)))
        self.lv.to_csv(self.outputfolder + fn + r'\\' + 'MVLV.csv')
        self.ss.to_csv(self.outputfolder + fn + r'\\' + 'SS.csv')
        self.lines.to_csv(self.outputfolder + fn + r'\\' + 'MVLines.csv')
        self.nodes.to_csv(self.outputfolder + fn + r'\\' + 'Nodes.csv')
        print('Saved data in folder {}'.format(fn))

    def save_data_pp(self, event=None):
        now = datetime.datetime.now()
        fn = r'Data_{}_{}'.format(self.ss.index[0], str(now.date()))
        if len(self.outputfolder) > 0:
            if not self.outputfolder[-2:] == r'\\':
                self.outputfolder += r'\\'
        util.create_folder(self.outputfolder + fn)
        print('Saving data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(self.lines), len(self.lv), len(self.nodes)))
        self.lv.to_csv(self.outputfolder + fn + r'\\' + 'MVLV.csv')
        self.ss.to_csv(self.outputfolder + fn + r'\\' + 'SS.csv')
        self.lines.to_csv(self.outputfolder + fn + r'\\' + 'MVLines.csv')
        self.nodes.to_csv(self.outputfolder + fn + r'\\' + 'Nodes.csv')
        print('Saved data in folder {}'.format(fn))
    
    def create_pandapower(self, event=None):
        """ Creates pandapower net and runs(?)
        """


# off = on_off_lines(hta, n0, ss=ps, lv=bt)

