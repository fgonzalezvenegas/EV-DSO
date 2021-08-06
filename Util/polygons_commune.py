# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:29:29 2020

@author: U546416
"""
import pandas as pd
import util
polygons_comms = pd.read_json(r'c:\user\U546416\Documents\PhD\Data\DataGeo\communes-20190101.json')

#%%
df = {}
for i in polygons_comms.features:
    comm = i['properties']['insee']
    name = i['properties']['nom']
    surf = i['properties']['surf_ha'] / 100
    ptype = i['geometry']['type']
    polygon = i['geometry']['coordinates']
    if ptype == 'MultiPolygon':
        polygon = [p[0] for p in polygon]
    if not (comm[0:2] in ['2A', '2B', '96', '97']):
        df[int(comm)] = {'COMM_NAME':name,
          'SURF_KM2' : surf,
          'Polygon':  polygon,
          'Polygon_Type' : ptype}
    
df = pd.DataFrame(df).T

df.COMM_NAME = util.fix_wrong_encoding_str(df.COMM_NAME)

polygons_c = util.do_polygons(df)

df.to_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\COMM_all_geo_2019')