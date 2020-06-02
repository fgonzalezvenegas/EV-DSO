# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:01:32 2020

@author: felip_001
"""
from grid import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from time import time
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import coord_transform as ct
import util

#%% Reading data
folder = r'C:\Users\felip_001\Downloads\Boriette\\'

fnps = 'poste-source.xlsx'
fnhta = 'reseau-hta.xlsx'
fnhtasout = 'reseau-souterrain-hta.xlsx'
fnbt = 'poste-electrique.xlsx'

ps = pd.read_excel(folder + fnps)
air = pd.read_excel(folder + fnhta)
sout = pd.read_excel(folder + fnhtasout)
bt = pd.read_excel(folder + fnbt)

air['Type'] = 'Overhead'
sout['Type'] = 'Underground'
hta = pd.concat((air, sout), ignore_index=True)
hta.index = ['L' + str(i) for i in range(len(hta))]
bt.index = ['BT' + str(i) for i in bt.index]
ps.index = ['PS' + str(i) for i in ps.index]

#%% Format data
for f in [ps, hta, bt]:
    if len(f.columns) == 2:
        f.columns = ['GeoPoint', 'Shape']
    else:        
        f.columns = ['GeoPoint', 'Shape', 'Type']
    f['x'] = f.GeoPoint.apply(lambda x: eval(x)[1])
    f['y'] = f.GeoPoint.apply(lambda x: eval(x)[0])
    f.Shape = f.Shape.apply(lambda x: eval(x)['coordinates'])

# Adding columns for endpoints of Lines
x1 = []
x2 = []
y1 = []
y2 = []
for i in hta.Shape:
    px1, py1 = i[0]
    px2, py2 = i[-1]
    x1.append(px1)
    x2.append(px2)
    y1.append(py1)
    y2.append(py2)
    
hta['x1'] = x1
hta['y1'] = y1
hta['x2'] = x2
hta['y2'] = y2

# TODO: Transform to GPS coordinates
# Transforming the data from Lambert 93 projection to GPS
# NOTE: Enedis data is not exactly in LAMB93, but in Lambert Conic Conformal Zone 8 (RGF93CC49), corresponding to the Paris latitude projection
for f in [ps, hta, bt]:
    xyGPS = f[['x','y']].apply(lambda x: ct.point_LAMB93CC_WGS84((x[0], x[1]), cc=8), axis=1)
    f['xGPS'] = xyGPS.apply(lambda x: x[0])
    f['yGPS'] = xyGPS.apply(lambda x: x[1])
    
hta['ShapeGPS'] = hta.Shape.apply(lambda x: ct.polygon_LAMB93CC_WGS84(x, cc=8))
hta['Length'] = hta.ShapeGPS.apply(lambda x: util.length_segment_WGS84(np.array(x)))
xy1GPS = hta[['x1','y1']].apply(lambda x: ct.point_LAMB93CC_WGS84((x[0], x[1]), cc=8), axis=1)
xy2GPS = hta[['x2','y2']].apply(lambda x: ct.point_LAMB93CC_WGS84((x[0], x[1]), cc=8), axis=1)
hta['x1GPS'] = xy1GPS.apply(lambda x: x[0])
hta['y1GPS'] = xy1GPS.apply(lambda x: x[1])
hta['x2GPS'] = xy2GPS.apply(lambda x: x[0])
hta['y2GPS'] = xy2GPS.apply(lambda x: x[1])

# # drop too short lines that might cause problems
# d = ((hta.x1-hta.x2)*(hta.x1-hta.x2)+(hta.y1-hta.y2)*(hta.y1-hta.y2))
# hta = hta[d>5]

#%% Plot grid
#
ax = plot_lines(hta)
ax.plot(hta.x1,hta.y1,'*g', markersize=5, alpha=0.5, label='Line Start')
ax.plot(hta.x2,hta.y2,'<b', markersize=3, label='Line End')
ax.plot(bt.x, bt.y, 'oy', markersize=5, label='MV/LV SS', alpha=0.5)
ax.plot(ps.x, ps.y, '*', color='purple', markersize=12, alpha=1, label='HV/MV SS')
ax.set_title('Initial grid')
delta = 100
for p in ps.index:
    ax.text(ps.x[p] + delta, ps.y[p] + delta, p)
    
plt.legend()


#%% Defines origin node and sorts lines according to distance to that node
ps0 = 'PS3'
n0x, n0y = (ps.x[ps0], ps.y[ps0])
d1 = dist_series_point(hta.x1, hta.y1, n0x, n0y)
d2 = dist_series_point(hta.x2, hta.y2, n0x, n0y)

d = pd.concat([d1,d2], axis=1).min(axis=1)

hta['d0'] = d
hta = hta.sort_values('d0')
hta.index = ['L' + str(i) for i in range(len(hta))]


#%% Creates nodes dataframe with coordinates of all possible nodes
xy = ['x', 'y']
ni = hta[['x1','y1']]
nf = hta[['x2','y2']]
ni.columns = xy
nf.columns = xy
nps = len(ps)
nhta = len(hta)
nodes = pd.DataFrame(pd.concat((ps[xy], ni,nf), ignore_index=True))
nodes.index = ['N' + str(i) for i in range(len(nodes))]
hta['node_i'] = nodes.index[nps:nps+nhta]
hta['node_e'] = nodes.index[nps+nhta:]

#%% Create Graph by grouping closest nodes. Some thresholds for define 'closer' nodes are required
# Threshold using the Lambert projection measures (units in meters)
eps = 5
eps2 = 40
t = [time()]
#ids = pd.DataFrame(nodes.index, index=nodes.index)
i = 0
# fnodes = final nodes, it will grow as new nodes are added
fnodes = pd.DataFrame(columns=nodes.columns)
assignment = pd.Series(index=nodes.index, dtype=str)
doubts = []
d0 = []
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
    d = dist_series_point(nodes.x, nodes.y, px, py)
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
    d = dist_series_point(nodes.x, nodes.y, px, py)
    # select nodes that are sufficiently closer (eps tolerance)
    n = nodes[d<eps]
    # Get min distance to other nodes
    dmin = d.nsmallest(2)
    idmin = dmin.index[-1]
    dmin = dmin[-1]
    # If there is no other close node (at eps distance), check with a bigger tolerance
    if len(n) == 1:
        if dmin < eps2:
            n = nodes[d<eps2]
    if dmin > 0.5:
        n = nodes[d<eps2]
    # Assign new node, keep track of old nodes to new nodes
    fnodes = fnodes.append(n.mean(axis=0), ignore_index=True)
    assignment[n.index] = i
    #print some info
    if dmin == 0:
        d0.append([n, len(n)])
    if dmin > eps2:
        if dmin < eps2*2:
            print(i, 'min_d: {:.2f}'.format(dmin), '#nodes:', len(n))
            print('\t oldnodes {}'.format(len(nodes)))
            doubts.append([px,py, nodes.x[idmin], nodes.y[idmin]])
    nodes = nodes.drop(n.index)
    i +=1
    if len(nodes) == 0:
        break
#    if i >100: 
#        break 
t.append(time())
print('Finished grouping nodes\nNew nodes = {}'.format(len(fnodes)))
print('Elapsed time {:.2f} s'.format(t[1]-t[0]))

fnodes.index = ['N'+ str(i) for i in fnodes.index]

hta.node_i = assignment[hta.node_i].values
hta.node_e = assignment[hta.node_e].values
hta.node_i = hta.node_i.apply(lambda x: 'N' + str(int(x)))
hta.node_e = hta.node_e.apply(lambda x: 'N' + str(int(x)))

#%% Assign MT/LT transfo to node
node_bt = []
for i in bt.index:
    d = dist_series_point(fnodes.x, fnodes.y, bt.x[i], bt.y[i])
    node_bt.append(d.idxmin())
bt['node'] = node_bt

# Assign SS to node:
node_ps = []
for i in ps.index:
    d = dist_series_point(fnodes.x, fnodes.y, ps.x[i], ps.y[i])
    node_ps.append(d.idxmin())
ps['node'] = node_ps

#%% Compute non connected data
 
# This is the list of lines connected to node of SS
n0 = ps.node[ps0]
c0 = connected(hta, n0)

#%% Plot connected and disconnected data
ax =  plot_lines(hta.loc[c0], label='Connected')
# disconnected = hta.drop(c0)
plot_lines(disconnected, ax=ax, color='k', linestyle='--', label='Non Connected')
ax.plot(ps.x, ps.y, '*', color='purple', markersize=15, label='SS')
ax.set_title('Connected and not-connected data (wrt {})'.format(ps0))
plt.legend()
delta = 100
for p in ps.index:
    ax.text(ps.x[p] + delta, ps.y[p] + delta, p)
#%% Drop non connected data
hta = hta.loc[c0]
ns = unique_nodes(hta)
fnodes = fnodes.loc[ns]
bt = bt[bt.node.isin(fnodes.index)]

# drop lines connected to themselves
hta = hta[hta.node_i != hta.node_e]

# Compute segments from node to lines
fi = fnodes[xy].loc[hta.node_i]
fe = fnodes[xy].loc[hta.node_e]
fi.index = hta.index
fe.index = hta.index
n =  len(fi)
segsi = pd.concat([hta[['x1', 'y1']], fi], axis=1).values.reshape(n,2,2)
segse = pd.concat([hta[['x2', 'y2']], fe], axis=1).values.reshape(n,2,2)

#%% Plot connected data
ax = plot_lines(hta, label='Lines')
# plot nodes
ax.plot(fnodes.x, fnodes.y, 'b.', markersize=3, label='Nodes')
# Plot BT trafo
ax.plot(bt.x, bt.y, 'yo', markersize=5, alpha=0.5, label='MV/LV tr')
# Plot SS
ax.plot(ps.x, ps.y, '*', color='purple', markersize=12, alpha=1, label='Main SS')
ax.set_title('Connected data to Main substation')
plt.legend()
for p in ps.index:
    ax.text(ps.x[p] + delta, ps.y[p] + delta, p)

#% Plot links of lines to nodes

lci = LineCollection(segsi, color='violet')
lce = LineCollection(segse, color='violet')
ax.add_collection(lci)
ax.add_collection(lce)

#%% SAVING DATA

util.create_folder(folder + r'ProcessedData')
print('Saving Full data:\n\tLines: {}\n\tML/LV: {}\n\tUnique Nodes:{}'.format(len(hta), len(bt), len(unique_nodes(hta))))
bt.to_csv(folder + r'ProcessedData\\' +  'MVLV_full.csv')
ps.to_csv(folder + r'ProcessedData\\' +  'SS_full.csv')
hta.to_csv(folder + r'ProcessedData\\' +  'MVLines_full.csv')
fnodes.to_csv(folder + r'ProcessedData\\' +  'FullNodes.csv')

