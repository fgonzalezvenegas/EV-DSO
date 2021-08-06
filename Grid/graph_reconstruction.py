# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:01:32 2020

Functions to reconstruct graph

@author: felip_001
"""
from grid import *
import pandas as pd
import numpy as np
from time import time
import coord_transform as ct
import util


#

def reading_files(folder, fnss='poste-source.xlsx', fnlv='poste-electrique.xlsx',
                  fnlinesair='reseau-hta.xlsx', fnlinesund='reseau-souterrain-hta.xlsx'):
    # Reading files
    ss = pd.read_excel(folder + fnss)
    air = pd.read_excel(folder + fnlinesair)
    und = pd.read_excel(folder + fnlinesund)
    lv = pd.read_excel(folder + fnlv)
    air['Type'] = 'Overhead'
    und['Type'] = 'Underground'
    lines = pd.concat((air, und), ignore_index=True)
    lines.index = ['L' + str(i) for i in range(len(lines))]
    lv.index = ['LV' + str(i) for i in lv.index]
    ss.index = ['SS' + str(i) for i in ss.index]
    return ss, lines, lv

def formating_files(ss, lines, lv):
    # Formatting
    for f in [lines, ss, lv]:
        if len(f.columns) == 2:
            f.columns = ['GeoPoint', 'Shape']
        else:        
            f.columns = ['GeoPoint', 'Shape', 'Type']
        f['x'] = f.GeoPoint.apply(lambda x: eval(x)[1])
        f['y'] = f.GeoPoint.apply(lambda x: eval(x)[0])
        f.Shape = f.Shape.apply(lambda x: eval(x)['coordinates'])
    # Adding columns for endpoints of Lines   
    lines['xy1'] = lines.Shape.apply(lambda x: x[0])
    lines['xy2'] = lines.Shape.apply(lambda x: x[-1])     
    df_split_point(lines, 'xy1', '1')
    df_split_point(lines, 'xy2', '2')
    return ss, lines, lv

def df_transform_point_gps(df, col='Shape', out='xyGPS'):
    df[out] = df[col].apply(lambda x: ct.point_LAMB93CC_WGS84(x, cc=8))

def df_transform_line_gps(df, col='Shape', out='ShapeGPS'):
    df[out] = df[col].apply(lambda x: ct.polygon_LAMB93CC_WGS84(x, cc=8))

def dist_LAMB(df, col='Shape', out='Length'):
    df[out] = df[col].apply(lambda x: dist_line(x))

def dist_GPS(df, col='ShapeGPS', out='Length'):
    df[out] = df[col].apply(lambda x: util.length_segment_WGS84(np.array(x)))

def df_split_point(df, col='Shape', out=''):
    df['x' + out] = df[col].apply(lambda x: x[0])
    df['y' + out] = df[col].apply(lambda x: x[1])

def lines_to_gps(lines):
    df_transform_line_gps(lines)
    dist_GPS(lines)
    df_transform_point_gps(lines, col='xy1', out='xy1GPS')
    df_split_point(lines, 'xy1GPS', '1GPS')
    df_transform_point_gps(lines, col='xy2', out='xy2GPS')
    df_split_point(lines, 'xy2GPS', '2GPS')
    lines['xyGPS'] = lines.apply(lambda x: ((x.x1GPS + x.x2GPS)/2, (x.y1GPS + x.y2GPS)/2), axis=1)
    df_split_point(lines, 'xyGPS', 'GPS')
    
def dfs_gps(ss, lines, lv, nodes):
    lines_to_gps(lines)
    df_transform_point_gps(lv)
    df_split_point(lv, 'xyGPS', 'GPS')
    df_transform_point_gps(ss)
    df_split_point(ss, 'xyGPS', 'GPS')
    df_transform_point_gps(nodes)
    df_split_point(nodes, 'xyGPS', 'GPS')

def dist_euclidean(sx, sy, px, py):
    """ Euclidean distance squared between 2 points
    """
    return np.sqrt((sx-px)*(sx-px) + (sy-py)*(sy-py))

def dist_line(segment):
    """ Euclidean distance of a line
    """
    n = len(segment)-1
    return sum([dist_euclidean(*segment[i],*segment[i+1]) for i in range(n)])

def dist_line_df(df, col='Shape', out='Length'):
    df[out] = df[col].apply(lambda x: dist_line(x))

def create_unconnected_nodes(lines, lv, ss):
    """ Returns a DF of all possible nodes coordinates between lines, transformers and substations
    """
    xy = ['x', 'y']
    ni = lines[['x1','y1']]
    nf = lines[['x2','y2']]
    nl = lv[['x', 'y']]
    ni.columns = xy
    nf.columns = xy
    nps = len(ss)
    nlv = len(lv)
    nlin = len(lines)
    nodes = pd.DataFrame(pd.concat((ss[xy], nl, ni, nf), ignore_index=True))
    nodes.index = ['N' + str(i) for i in range(len(nodes))]
    ss['node'] = nodes.index[:nps]
    lv['node'] = nodes.index[nps:nps+nlv]
    lines['node_i'] = nodes.index[nps+nlv:nps+nlv+nlin]
    lines['node_e'] = nodes.index[nps+nlv+nlin:]
    lines.index = ['L' + str(i) for i in range(len(lines))]
    return nodes 

def create_graph(nodes, eps1=3, eps2=7):
    """Create Graph by grouping closest nodes. 
    Thresholds (eps) for define 'closer' nodes are required
    """
    # Threshold using the Lambert projection measures (units in meters)
    eps = 3
    eps2 = 7
    t = [time()]
    i = 0
    # fnodes = final nodes, it will grow as new nodes are added
    fnodes = pd.DataFrame(columns=nodes.columns)
    assignment = pd.Series(index=nodes.index, dtype=str)
    # grouping nodes!
    print('Starting grouping nodes: Initial nodes = {}'.format(len(nodes)))
    # First group all nodes that are at d==0
    j = 0
    print('Grouping nodes at d==0')
    while True:
        if j >= len(nodes):
            break
        # Select first node in queue
        px = nodes.x.iloc[j]
        py = nodes.y.iloc[j]
        # Compute distance to other nodes
        d = dist_euclidean(nodes.x, nodes.y, px, py)
        # select nodes that are at distance 0
        n = nodes[d==0]
        # Assign new node if there is at least 2 nodes at d==0
        # keep track of old nodes to new nodes
        if len(n)>=2:
            fnodes = fnodes.append(n.mean(axis=0), ignore_index=True)
            assignment[n.index] = i
            # Drop these nodes from index
            nodes = nodes.drop(n.index)
            # Add one new node i
            i += 1
        else:
            #If there is no other node at d==0, skip this node
            j += 1
        # print some info
        if (i+j)%200 == 0:
            print(i+j)
    print('Finished grouping nodes at d==0, remaining nodes = {}'.format(len(nodes)))
    print('Grouping remaining nodes at 0<d<epsilon')
    while True:
        # Select first node in queue
        px = nodes.x.iloc[0]
        py = nodes.y.iloc[0]
        # Compute distance to other nodes
        d = dist_euclidean(nodes.x, nodes.y, px, py)
        # Get min distance to other nodes
        dmin = d.nsmallest(2)
        dmin = dmin[-1]
        n = nodes[d<eps]
        if eps < dmin < eps2:
            # If there is no other close node (at eps distance), check with a bigger tolerance
            n = nodes[d<eps2]
        # Assign new node, keep track of old nodes to new nodes
        fnodes = fnodes.append(n.mean(axis=0), ignore_index=True)
        assignment[n.index] = i
        nodes = nodes.drop(n.index)
        i +=1
        if len(nodes) == 0:
            break
        t.append(time())
    print('Finished grouping nodes\nNew nodes = {}'.format(len(fnodes)))
    print('Elapsed time {:.2f} s'.format(t[1]-t[0]))
    
    fnodes.index = ['N'+ str(i) for i in fnodes.index]
    fnodes['Shape'] = fnodes.apply(lambda x: (x['x'], x['y']), axis=1)
    assignment = 'N' + assignment.astype(str)
    
    return fnodes, assignment
    
def correct_node_assignment(lines, lv, ss, assignment):
      
    lines.node_i = assignment[lines.node_i].values
    lines.node_e = assignment[lines.node_e].values
    lv.node = assignment[lv.node].values
    ss.node = assignment[ss.node].values
    
def create_node_graph(lines, lv, ss):
    nodes = create_unconnected_nodes(lines, lv, ss)
    nodes, assignment = create_graph(nodes)
    correct_node_assignment(lines, lv, ss, assignment)
    return nodes

def clean_graph(lines, lv, ss, nodes, n0=None, plot=True, inplace=False):
    """ Drops non connected data, removes lines connected to themselves, and parallel lines
    """
    if n0 is None:
        n0 = ss.node[0]
    c0 = connected(lines, n0)
    cnodes = unique_nodes(lines.loc[c0])
    if plot:
        ax = plot_quick(lines.loc[c0], lv[lv.node.isin(cnodes)], ss[ss.node.isin(cnodes)], nodes.loc[cnodes])
        plot_lines(lines.drop(c0), color='grey', linestyle='--', ax=ax, col='Shape')
        ax.set_title('Connected and not-connected data (wrt {})'.format(ss[ss.node==n0].index[0]))
    if inplace:
        # drop non connected data
        lines = lines.loc[c0]
        lv =  lv[lv.node.isin(cnodes)]
        ss = ss[ss.node.isin(cnodes)]
        nodes = nodes.loc[cnodes]
        # drop lines connected to themselves
        lines = lines[lines.node_i != lines.node_e]
        return
    return lines.loc[c0], lv[lv.node.isin(cnodes)], ss[ss.node.isin(cnodes)], nodes.loc[cnodes]

def save_grid(lines, lv, ss, nodes, folder=''):
    util.create_folder(folder + r'ProcessedData')
    print('Saving Full data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(lines), len(lv), len(unique_nodes(lines))))
    lv.to_csv(folder + r'ProcessedData\\' +  'MVLV_full.csv')
    ss.to_csv(folder + r'ProcessedData\\' +  'SS_full.csv')
    lines.to_csv(folder + r'ProcessedData\\' +  'MVLines_full.csv')
    nodes.to_csv(folder + r'ProcessedData\\' +  'Nodes_full.csv')
    
def run_graph(folder, folder_output=''):
    # Reading files
    print('Reading files at ' + folder)
    ss, lines, lv = reading_files(folder)
    ss, lines, lv = formating_files(ss, lines, lv)
    ax=plot_quick(lines, lv, ss)
    ax.set_title('Initial data')
    nodes = create_node_graph(lines, lv, ss)
    clean_graph(lines, lv, ss, nodes, inplace=True, plot=True)
    print('Transforming data to GPS (WGS84)')
    dfs_gps(ss, lines, lv, nodes)
    print('Saving')
    save_grid(lines, lv, ss, nodes, folder=folder_output)
    return lines, lv, ss, nodes


