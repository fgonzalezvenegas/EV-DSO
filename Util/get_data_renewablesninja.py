# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:51:16 2021

@author: U546416
"""

import numpy as np
import matplotlib.pyplot as plt

import requests
import pandas as pd
import json

token = '71b12e7f11995e0d16afb74901b926524e98dfae'
api_base = 'https://www.renewables.ninja/api/'

s = requests.session()
# Send token header with each request
s.headers = {'Authorization': 'Token ' + token}


##
# PV example
##

url = api_base + 'data/pv'
year = 2018
args = {
    'lat': 45.1328,
    'lon': 1.5216,
    'date_from': '{}-01-01'.format(year),
    'date_to': '{}-12-31'.format(year),
    'dataset': 'merra2',
    'capacity': 1.0,
    'system_loss': 0.1,
    'tracking': 0,
    'tilt': 35,
    'azim': 180,
    'format': 'json',
    'local_time': 'false' # this is weird, it doesnt provide correct local time
}

proxies = {   "http"  : 'http://U546416:c4m3l14s@http.ntlm.internetpsa.inetpsa.com:8080', 
              "https" : 'https://U546416:c4m3l14s@http.ntlm.internetpsa.inetpsa.com:8080'}

print('doing request')
r = s.get(url, params=args, proxies=proxies)

# Parse JSON to get a pandas.DataFrame of data and dict of metadata
print('parsing request')
parsed_response = json.loads(r.text)

data = pd.read_json(json.dumps(parsed_response['data']), orient='index')

metadata = parsed_response['metadata']
print('saving')
#data.to_csv(r'C:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles\PV_{}.csv'.format(year))
#(data*0.9).to_csv(r'C:\user\U546416\Documents\PhD\Data\MVGrids\Boriette\Profiles\PVrooftop_{}.csv'.format(year))
#%%