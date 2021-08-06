# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:31:37 2020

Cleaning Installed capacity by department file

@author: U546416
"""

import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt


folder = r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau'

#iris_prod = pd.read_csv(folder + r'\production-electrique-par-filiere-a-la-maille-iris.csv',
#                        engine='python', sep=';')
dep_parc_prod = pd.read_csv(folder + r'\parc-des-installations-de-production-raccordees-par-departement.csv',
                            engine='python', sep=';')

dep_parc_prod.columns = ['REG_CODE', 'REG_NAME', 'DEP_CODE', 'DEP_NAME',
       'CODE_TYPE_PROD', 'TYPE_PROD', 'ID_TYPE_INJ', 'TYPE_INJ',
       'TRANCHE', 'FIN_TRIMESTRE', 'NB_INSTALL', 'P_MW',
       'GEO_POINT', 'GEO_SHAPE']

dep_parc_prod.drop(['GEO_POINT', 'GEO_SHAPE'], axis=1, inplace=True)
dep_parc_prod.REG_NAME = util.fix_wrong_encoding_str(dep_parc_prod.REG_NAME)
dep_parc_prod.DEP_NAME = util.fix_wrong_encoding_str(dep_parc_prod.DEP_NAME)
tranche_type = {'a-]0;36]' : 'home_rooftop',
                'b-]36;100]': 'commercial_rooftop',
                'c-]100;250]': 'commercial_rooftop',
                'd-]250;...[': 'solar_farm'}

dep_parc_prod['TYPE_PV'] = dep_parc_prod.TRANCHE.apply(lambda x: tranche_type[x]) 

dep_parc_prod.to_csv(folder + r'\parc-pv-departement.csv')
