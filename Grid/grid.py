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

import pandapower.control as ppc


def plot_lines(lines, col='ShapeGPS', ax=None, **plot_params):
    """ Plots a DataFrame containing line coordinates in column 'col'
    """
    segs = []
    if not (col in lines):
        if 'Shape' in lines:
            col='Shape'
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
    
def plot_quick(lines, lv, ss, nodes=None, GPS=False):
    if GPS:
        sg = 'GPS'
        d = 0.005
    else:
        sg = ''
        d = 20
    col='Shape' + sg
    x = 'x' + sg
    y = 'y' + sg
    x1 = 'x1' + sg
    y1 = 'y1' + sg
    x2 = 'x2' + sg
    y2 = 'y2' + sg
    ax=plot_lines(lines, col=col, label='Lines')
    ax.plot(lv[x], lv[y], 'o', color='yellow', alpha=0.5, label='LV trafo')
    ax.plot(ss[x], ss[y], '*', color='purple', markersize=10, label='Substation')
    for s, t in ss.iterrows():
        ax.text(t[x]+d, t[y]+d, s)
    if nodes is None:
        ax.plot(lines[x1], lines[y1], '.', color='k', markersize=1, label='Nodes')
        ax.plot(lines[x2], lines[y2], '.', color='k', markersize=1, label='_')
    else:
        ax.plot(nodes[x], nodes[y], '.', color='k', markersize=1, label='Nodes')
        q = get_node_to_line_segments(nodes, lines, lv, GPS=GPS)
        plot_lines(q, linestyle='--', color='purple', col=col, ax=ax)
    plt.legend()
    return ax

def to_node(lines, node):
    """ Returns all lines connected to a given node"""
    return list(lines[((lines.node_i == node) | (lines.node_e == node))].index)

def unique_nodes(lines):
    """ Returns a list of unique nodes in a list of lines """
    return list(set(lines.node_i).union(set(lines.node_e)))

def new_nodes(lines, node):
    l = unique_nodes(lines)
    l.remove(node)
    return l

def connected(lines, node):
    """ Returns all lines directly or indirectly connected to a given departure node """
    # create nx graph
    g = nx.Graph()
    for l,t in lines.iterrows():
        g.add_edge(t.node_i, t.node_e)
    # compute connected elements
    cc = list(nx.connected_components(g))
    # check for connected elements to node
    for c in cc:
        if node in c:
            return list(lines[lines.node_i.isin(c) | lines.node_e.isin(c)].index)
    
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
    feeder_length = {}
    nfs = 0
    for fs in cc:
        lls = lines[(lines.node_i.isin(fs)) | (lines.node_e.isin(fs))].index
        # check if lines are connected to main node, otherwise skip
        if (n0 in lines.node_i[lls].values) or (n0 in lines.node_e[lls].values):
            feeder[lls] = nfs
            feeder_length[nfs] = lines.Length[lls].sum()
            nfs += 1
    # renaming feeders from shortest to longest
    feeder_length = pd.Series(feeder_length)
    feeder_length.sort_values(inplace=True)
    nfs = 0
    for fs in feeder_length.index:
        feeder[feeder == fs] = 'F{:02d}'.format(nfs)
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

def get_node_to_line_segments(nodes, lines, lv=None, GPS=True):
    """ returns a pd.DataFrame of segments that rely nodes to lines
    """
    if GPS:
        colpoint='xyGPS'
        col = 'ShapeGPS'
        colline1 = 'xy1GPS'
        colline2 = 'xy2GPS'
    else:
        col ='Shape'
        colpoint ='Shape'
        colline1 = 'xy1'
        colline2 = 'xy2'
    # getting node coordinates for each line extreme
    fi = nodes[colpoint][lines.node_i]
    fe = nodes[colpoint][lines.node_e]
    fi.index = lines.index
    fe.index = lines.index
    # reformating
    shape1 = lines[colline1]
    shape2 = lines[colline2]
    # defining segments as [(xnode, ynode), (xline(node), yline(node))]
    segi = fi.apply(lambda x: [x]) + shape1.apply(lambda x: [x]) 
    sege = fe.apply(lambda x: [x]) + shape2.apply(lambda x: [x]) 
    # removing segments of length null
    segi = segi[~(fi==shape1)]
    sege = sege[~(fe==shape2)]
    # appending to output
    segs = pd.concat([segi, sege], ignore_index=True)
    # doing the same for LV trafos
    if not (lv is None):
        fl = nodes[colpoint][lv.node]
        fl.index = lv.index
        segl = fl.apply(lambda x: [x]) + lv[colpoint].apply(lambda x: [x])
        segl = segl[~(fl==lv[colpoint])]
        segs = pd.concat([segs,segl], ignore_index=True)
    return pd.DataFrame(segs, columns=['Shape'])

def assign_feeder_to_node(nodes, n0, lines):
    nodes['Feeder'] = ''
    for f in lines.Feeder.unique():
        nodes.Feeder[unique_nodes(lines[lines.Feeder == f])] = f
    nodes.Feeder[n0] = '0SS'
    
def rename_nodes(nodes, n0, lines, lv, ss):
    """ Rename the nodes according to Feeder appartenance and distance to main node
    New index is just ascending numeric
    Adds new column 'name' with a meaningful name
    Updates relations to lines node_i, node_e; lv and ss
    """
    nodes['d'] = ((nodes.xGPS[n0]-nodes.xGPS)**2+(nodes.yGPS[n0]-nodes.yGPS)**2)
    assign_feeder_to_node(nodes, n0, lines)
    # Sorting nodes
    nodes.sort_values(['Feeder', 'd'], inplace=True)
    # Creating new index
    nodes.reset_index(inplace=True)
    #Renames nodes
    nodes.index = 'N'+ nodes.index.astype(str)
    # Get relationship old index-new index
    old_new = pd.Series(index=nodes['index'], data=nodes.index)
    
    #Update relationship to lines and trafos
    lines.node_i = old_new[lines.node_i].values
    lines.node_e = old_new[lines.node_e].values
    lv.node = old_new[lv.node].values
    ss.node = old_new[ss.node].values 
    # drops old index
    nodes.drop('index', axis=1, inplace=True)
    
def rename_lines(lines, n0):
    """ Rename lines according to Feeder and distance to main node
    Name (index) considers number and feeder meaningful name
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
    lines.index = ['L'+str(i)+'_'+lines.Feeder[i] for i in lines.index]
    
    
    
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
        g = assign_poly(point, geo, dt=dt, notdt=0)
        while g is None:
            dt = dt*2
            g = assign_poly(point, geo, dt=dt, notdt=dt/2)         
        geo_lv[load] = g
        i += 1
        if i%500 == 0:
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

def get_profiles_per_geo(lv, profiles_per_type, consos, MW=False):
    """ Returns a Dataframe of load profiles per IRIS
    if MW == True:
        returns the profiles in power (MW)
    if MW is False:
        returns the profiles with max == 1
    """
    cols = ['Conso_RES', 'Conso_PRO', 'Conso_Industrie', 'Conso_Tertiaire', 'Conso_Agriculture']
    profs = {}
    for geo in lv.Geo.unique():
        # omits nan assignmetns (avoid crashing the program)
        if geo == geo:
            profs[geo] = (profiles_per_type * consos[cols].loc[geo].values).sum(axis=1)
    profs = pd.DataFrame(profs)
    # correct null values to 0
    profs.fillna(0, inplace=True)
    if MW:
        return profs
    else:
        return profs / profs.max()
        
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
            
#%% Create pandapower grid
def add_extra(net_el, idxs, vals, param_net='Feeder'):
    if not (param_net in net_el):
        net_el[param_net] = None
    net_el[param_net][idxs] = vals
    
def add_tech_types(net, tech):
    """ Add std_type to an existing net
    """
    for i, t in tech.iterrows():
        # i is ID of tech, t is tech data
        data = dict(c_nf_per_km=t.C,
                    r_ohm_per_km=t.R,
                    x_ohm_per_km=t.X,
                    max_i_ka=t.Imax/1000,
                    q_mm=t.Section,
                    type='oh' if t.Type == 'Overhead' else 'cs')
        pp.create_std_type(net, name=i, data=data, element='line')      
            
def create_pp_grid(nodes, lines, tech, loads, n0, 
                   hv=True, ntrafos_hv=2, vn_kv=20,
                   tanphi=0.3, hv_trafo_controller=True, verbose=True):
    """
    """
    if verbose:
        print('Starting!')
    # 0- empty grid
    net = pp.create_empty_network()
    # 1- std_types
    if verbose:
        print('\tTech types')
    add_tech_types(net, tech)
    # 2 - Create buses
    if verbose:
        print('\tBuses')
    idxs = pp.create_buses(net, len(nodes), vn_kv=vn_kv, name=nodes.index, 
                           geodata=list(nodes.xyGPS), type='b', zone=nodes.Geo.values)
    if 'Feeder' in nodes:
        add_extra(net.bus, idxs, nodes.Feeder.values, 'Feeder')
    # 3- Create lines
    if verbose:
        print('\tLines')
    for linetype in lines.Conductor.unique():
        ls = lines[lines.Conductor == linetype]
        nis = pp.get_element_indices(net, "bus", ls.node_i)
        nes = pp.get_element_indices(net, "bus", ls.node_e)
        idxs = pp.create_lines(net, nis, nes, ls.Length.values/1000, 
                               std_type=linetype, 
                               name=ls.index, geodata=list(ls.ShapeGPS),
                               df=1., parallel=1, in_service=True)
        if 'Feeder' in lines:
            add_extra(net.line, idxs, ls.Feeder.values, 'Feeder')
    # 4- Create loads
    if verbose:
        print('\tLoads')
    nls = pp.get_element_indices(net, 'bus', loads.node)
    idxs = pp.create_loads(net, nls, name=loads.index, p_mw=loads.Pmax_MW.values, 
                    q_mvar=loads.Pmax_MW.values*tanphi)
    if 'type_load' in loads:
        add_extra(net.load, idxs, loads.type_load.values, 'type_load')
    else:
        add_extra(net.load, idxs, 'Base', 'type_load')
    if 'Geo' in loads:
        add_extra(net.load, idxs, loads.Geo, 'zone')
        
    # Adding external grid
    if verbose:
        print('\tExt Grid')
    if hv:
        # If HV, then add extra bus for HV and add trafo
        b0 = pp.create_bus(net, vn_kv=110, geodata=nodes.xyGPS[n0], name='HV_SS')
        # Adding HV-MV trafo (n x 40MW trafos)
        t = pp.create_transformer(net, hv_bus=b0, lv_bus=n0, 
                                  std_type='40 MVA 110/20 kV', 
                                  name='TrafoSS', parallel=ntrafos_hv) 
        if hv_trafo_controller:
            # Add tap changer controller at MV side of SS trafo
            ppc.DiscreteTapControl(net, t, 0.99, 1.01, side='lv')
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
                 lv=None, geo=None, GPS=True, tech=None,
                 profiles=None,
                 outputfolder=''):
        self.lines = lines
        self.n0 = n0
        self.p0 = ss[ss.node == n0].index[0]
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
        try:
            self.ax.set_position([0.125,0.2,0.775,0.68])
            plt.get_current_fig_manager().window.showMaximized()
        except:
            pass
        
        fs  = self.lines.Feeder.unique()
        self.feeders = np.sort([f for f in fs if f==f])
        self.ss = ss
        self.lv = lv
        self.geo = geo
        self.dist = None
        self.recompute_distance()
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
        
        self.drawn_nodes = False
        self.names = False
        
        self.cid = self.f.canvas.mpl_connect('pick_event', self.set_on_off)
        self.ncalls = 0
        self.lastevent = []
    
    def do_buttons(self):
        
        # Add button to separate substations
        axbutton = plt.axes([0.05, 0.05, 0.12, 0.075])
        self.buttonsepss = Button(axbutton, 'Separate substations\nservice areas')
        self.buttonsepss.on_clicked(self.auto_separation)
        
        # Add button to untangle feeders
        axbutton = plt.axes([0.24, 0.05, 0.12, 0.075])
        self.buttonuntang = Button(axbutton, 'Untangle\nfeeders')
        self.buttonuntang.on_clicked(self.auto_debouclage)
        
        # Add button to recompute feeders
        axbutton = plt.axes([0.43, 0.05, 0.12, 0.075])
        self.buttonf = Button(axbutton, 'Recompute\nFeeders')
        self.buttonf.on_clicked(self.recompute_feeders)
        
        # Add button to reduce data
        axbutton = plt.axes([0.62, 0.05, 0.12, 0.075])
        self.buttonred = Button(axbutton, 'Reduce data and\n compute tech')
        self.buttonred.on_clicked(self.reduce_compute_tech)
        
        # Add button to save data
        axbutton = plt.axes([0.81, 0.05, 0.12, 0.075])
        self.buttonsave = Button(axbutton, 'Save current data')
        self.buttonsave.on_clicked(self.save_data)
        
        # Add button to toggle aspect 
        axbutton = plt.axes([0.75, 0.905, 0.05, 0.05])
        self.buttona = Button(axbutton, 'Toggle\nAspect')
        self.buttona.on_clicked(self.aspect)
        
        # Add button to draw names
        axbutton = plt.axes([0.65,0.905,0.05,0.05])
        self.buttonms = Button(axbutton, 'Write\nnames')
        self.buttonms.on_clicked(self.onoff_names)
        
        # Add button to draw nodes
        axbutton = plt.axes([0.32,0.905,0.1,0.05])
        self.buttonnds = Button(axbutton, 'On/Off Nodes')
        self.buttonnds.on_clicked(self.onoff_nodes)
        
        # Add button to toggle on/off visibility of bt and geo:
        if self.lv is not None:
            axbuttonbt = plt.axes([0.06, 0.905, 0.1, 0.05])
            self.buttonlv = Button(axbuttonbt,'On/Off MV/LV')
            self.buttonlv.on_clicked(self.onoff_lv)
        if self.geo is not None:
            axbuttongeo = plt.axes([0.19, 0.905, 0.1, 0.05])
            self.buttongeo = Button(axbuttongeo,'On/Off GeoShapes')
            self.buttongeo.on_clicked(self.onoff_geo)
            
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
        self.draw_lines()
        self.draw_else()
        self.set_view()

    def remove_lines(self):
        for i in self.pltlines:
            i.remove()
    
    def set_view(self):
        if self.currentview is not None:
            self.ax.axis(self.currentview)
        self.ax.set_aspect('equal')
        
    def draw_lines(self):
        self.pltlines = []
        if self.line_view == 'Feeder':
            # Plotting each feeder a different color
            for i, f in enumerate(self.feeders):
                plot_lines(self.lines[(self.lines.Feeder==f) & (self.lines.Connected) & (self.lines.Type == 'Underground')], 
                                      col=self.col, ax=self.ax, 
                                      color=colors[int(i%len(colors))], picker=5, label=f, linewidth=2) 
                self.pltlines.append(self.ax.collections[-1])
                plot_lines(self.lines[(self.lines.Feeder==f) & (self.lines.Connected) & (self.lines.Type == 'Overhead')], 
                                      col=self.col, ax=self.ax, 
                                      color=colors[int(i%len(colors))], picker=5, linewidth=1)
                self.pltlines.append(self.ax.collections[-1])
                px, py = get_farther(self.ss[self.ss.node == self.n0].iloc[0], self.nodes.loc[unique_nodes(self.lines[self.lines.Feeder == f])])
                lt = self.ax.text(px + self.dtext, py + self.dtext, f)
                self.pltlines.append(lt)
        # Plotting lines according to tech data
        if self.line_view == 'LineType':
            for lt in colors_tech.keys():
                for i, c in enumerate(self.tech[self.tech.Type == lt].index):
                    plot_lines(self.lines[self.lines.Conductor == c], 
                               col=self.col, ax=self.ax, 
                               color=colors_tech[lt][int(i%len(colors_tech[lt]))], picker=5, label=c, linewidth=self.tech.Section[c]/50) 
                    self.pltlines.append(self.ax.collections[-1])
        # Not connected lines
        notconn = self.lines[self.lines.Connected == False]
        plot_lines(notconn, col=self.col, ax=self.ax, 
                   color='k', linestyle=':', picker=5, label='Disconnected')
        self.pltlines.append(self.ax.collections[-1])
        # Connected but without feeder
        notassigned = self.lines[self.lines.Connected & self.lines.Feeder.isnull()]
        plot_lines(notassigned, col=self.col, ax=self.ax, 
                   color='r', linestyle=':', picker=5, label='Without feeder')
        self.pltlines.append(self.ax.collections[-1])
        self.mainview = self.ax.axis()
        self.openlines = self.ax.add_collection(LineCollection(segments=[], gid=[], linestyle=':', color='k', picker=5))
        self.pltlines.append(self.ax.collections[-1])
        self.reconnected = self.ax.add_collection(LineCollection(segments=[], gid=[], linestyle='-.', color='darkgoldenrod', picker=5, label='Reconnected'))    
        self.pltlines.append(self.ax.collections[-1])
        if len(self.f.legends) > 0:
            self.f.legends = [] # Remove existing legend
            plt.plot() # To force drawing of new legend
        self.f.legend()
        
    def draw_else(self):
        if self.ss is not None:
            self.ax.plot(self.ss[self.x], self.ss[self.y], '*', color='purple', markersize=10, label='SS')
            for p in self.ss.index:
                self.ax.text(self.ss[self.x][p]+self.dtext, self.ss[self.y][p]+self.dtext, p)
        if self.lv is not None:
            self.lvartist = self.ax.plot(self.lv[self.x], self.lv[self.y], 'oy', markersize=5, label='MV/LV', alpha=0.5)[0]
        if self.geo is not None:
            # Reducing Geo shapes to plot to only those in the study zone
            axis = self.ax.axis()
            self.geo = self.geo[((axis[0]<self.geo[self.x]) & (self.geo[self.x]<axis[1]) & 
                             (axis[2]<self.geo[self.y]) & (self.geo[self.y]<axis[3])) | (self.geo.index.isin(self.lv.Geo))]
            # plot polys
            collection = PatchCollection([p[0] for p in self.geo.Polygon], 
                                         facecolor='none', edgecolor='k', linestyle='--', linewidth=0.8, alpha=0.8)
            self.geoartist = self.ax.add_collection(collection)
            if 'Name' in self.geo:
                self.geonames = [self.ax.text(self.geo[self.x][g], self.geo[self.y][g], self.geo.Name[g],
                                              horizontalalignment='center') for g in self.geo.index]
        self.ax.set_title('Click on lines to switch on/off')
        
    
    def draw_nodes(self):
        # drawing nodes
        self.node_artist = self.ax.plot(self.nodes[self.x], self.nodes[self.y], 
                                        'o', color='k', markersize=4, alpha=0.5, label='Nodes')[0]
        # drawing lines to nodes:
        ls = get_node_to_line_segments(self.nodes, self.lines, self.lv)
        plot_lines(ls, col=self.col, ax=self.ax, 
                   color='purple', linestyle='--', label='_', linewidth=1) 
        self.nodeline_artist = self.ax.collections[-1]        
        
    def plot_names(self):
        self.nodenames = [self.ax.text(t[self.x], t[self.y], n, horizontalalignment='center') 
                          for n, t in self.nodes.iterrows()]
        self.linenames = [self.ax.text(t[self.x], t[self.y], n, horizontalalignment='center') 
                          for n, t in self.lines.iterrows()]
        self.lvnames = [self.ax.text(t[self.x], t[self.y], n, horizontalalignment='center') 
                          for n, t in self.lv.iterrows()]
    
    def erase_names(self):
        for i in self.nodenames: i.remove()
        for i in self.linenames: i.remove()
        for i in self.lvnames: i.remove()
        
        
    def onoff_lv(self, event=None):
        self.lvartist.set_visible(not self.lvartist.get_visible())
    
    def onoff_geo(self, event=None):
        self.geoartist.set_visible(not self.geoartist.get_visible())
        for gn in self.geonames:
            gn.set_visible(not gn.get_visible())
    
    def onoff_nodes(self, event=None):
        if hasattr(self, 'node_artist'):
            self.node_artist.set_visible(not self.node_artist.get_visible())
            self.nodeline_artist.set_visible(not self.nodeline_artist.get_visible())
        else:
            self.draw_nodes()
        
    def onoff_names(self, event=None):
        self.names = not self.names
        if self.names:
            self.plot_names()
        else:
            self.erase_names()
            
    def recompute_feeders(self, event=None):
        print('\nRecomputing independent feeders')
        #Getting currentview
        self.currentview = self.ax.axis()
        self.lines.Feeder = get_ind_feeders_nx(self.lines[self.lines.Connected], self.n0, verbose=True)
        fs  = self.lines.Feeder.unique()
        self.feeders = np.sort([f for f in fs if f==f])
        number_init_feeders(self.lines, self.n0)
#        self.ax.clear()
        self.remove_lines()
        self.draw_lines()
        self.set_view()
        self.recompute_distance()
    
    def recompute_distance(self, event=None):
        print('\nRecomputing distances to HV/MV substations')
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
        self.remove_lines()
        self.draw_lines()
        self.set_view()
    
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
        # Renaming nodes
        rename_nodes(self.nodes, self.n0, self.lines, self.lv, self.ss)
        # get new main node
        self.n0 = self.ss.node[self.p0]
        # rename lines
        rename_lines(self.lines, self.n0)
        # Assigning tech data
        self.lines['Conductor'] = assign_tech_line(self.lines, self.lv, self.n0, self.tech)
        self.tech.sort_values('Section', inplace=True, ascending=False)
        
        self.line_view = 'LineType'
        self.ax.clear()
        self.draw()
        
    def auto_separation(self, event=None):
        """ Automatic separation of primary substation 
        service areas based on min distance
        """
        # assign each node to a SS
        areas = self.dist.idxmin(axis=1)
        # get nodes assigned to main SS
        ndS0 = areas[areas == self.p0].index
        # get nodes assigned to other SS
        ndSx = areas[~ (areas == self.p0)].index
        # get lines in the edge
        ledge = self.lines[(self.lines.node_i.isin(ndS0) & self.lines.node_e.isin(ndSx)) | 
                           (self.lines.node_i.isin(ndSx) & self.lines.node_e.isin(ndS0))].index
        self.lines.Connected[ledge] = False
        self.recompute_feeders()  

    def auto_debouclage(self, event=None):
        """ Automatic 'untangling' of feeders from one ss
        """
        # Primary nodes for each feeders
        lfs = to_node(self.lines, self.n0)
        # compute distance to each primary node
        df = pd.DataFrame()
        for p in lfs:
            other_lines = list(lfs)
            other_lines.remove(p)
            l = self.lines[self.lines.Connected][['Length','node_i','node_e']].drop(other_lines)
            print('\tComputing distance for {}'.format(p))
            df = pd.concat([df,dist_to_node_nx(l, self.n0, name=p)], axis=1)
        # Assign each node to a Primary feeder
        areas = df.idxmin(axis=1)
        areas[self.n0] = 'SS'
        # get lines in the edge
        ledge = self.lines[(~(areas[self.lines.node_i].values == areas[self.lines.node_e].values)) & 
                           (~(areas[self.lines.node_i].isnull().values & areas[self.lines.node_e].isnull().values))].index
        ledge = ledge.drop(lfs, errors='ignore')
        # assign lines in the edge as disconnected
        self.lines.Connected[ledge] = False
        self.recompute_feeders()           
        
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

