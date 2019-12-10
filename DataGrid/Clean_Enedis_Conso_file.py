# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:29:20 2019
Cleaning Enedis file of demand by IRIS
@author: U546416
"""

import pandas as pd
import numpy as np
#
#['AnnÃ©e', 'Nom IRIS', 'Type IRIS', 'Nom commune', 'Code commune',
#       'Nom EPCI', 'Code EPCI', 'Type EPCI', 'Nom dÃ©partement',
#       'Code dÃ©partement', 'Nom rÃ©gion', 'Code rÃ©gion',
#       'Nb sites RÃ©sidentiel', 'Conso totale RÃ©sidentiel (MWh)',
#       'Conso moyenne RÃ©sidentiel (MWh)',
#       'Conso totale RÃ©sidentiel usages thermosensibles (MWh)',
#       'Conso totale RÃ©sidentiel usages non thermosensibles (MWh)',
#       'Conso moyenne RÃ©sidentiel usages thermosensibles (MWh)',
#       'Conso moyenne RÃ©sidentiel usages non thermosensibles (MWh)',
#       'Part thermosensible RÃ©sidentiel (%)',
#       'ThermosensibilitÃ© totale RÃ©sidentiel (kWh/DJU)',
#       'ThermosensibilitÃ© moyenne RÃ©sidentiel (kWh/DJU)',
#       'Conso totale corrigÃ©e de l'alÃ©a climatique RÃ©sidentiel usages thermosensibles (MWh)',
#       'Conso moyenne corrigÃ©e de l'alÃ©a climatique RÃ©sidentiel usages thermosensibles (MWh)',
#       'Nb sites Professionnel', 'Conso totale Professionnel (MWh)',
#       'Conso moyenne Professionnel (MWh)', 'DJU', 'Nb sites Agriculture',
#       'Conso totale Agriculture (MWh)', 'Nb sites Industrie',
#       'Conso totale Industrie (MWh)', 'Nb sites Tertiaire',
#       'Conso totale Tertiaire (MWh)', 'Nb sites Autres',
#       'Conso totale Autres (MWh)', 'Nombre d'habitants',
#       'Taux de logements collectifs', 'Taux de rÃ©sidences principales',
#       'Superficie des logements < 30 m2',
#       'Superficie des logements 30 Ã  40 m2',
#       'Superficie des logements 40 Ã  60 m2',
#       'Superficie des logements 60 Ã  80 m2',
#       'Superficie des logements 80 Ã  100 m2',
#       'Superficie des logements > 100 m2',
#       'RÃ©sidences principales avant 1919',
#       'RÃ©sidences principales de 1919 Ã  1945',
#       'RÃ©sidences principales de 1946 Ã  1970',
#       'RÃ©sidences principales de 1971 Ã  1990',
#       'RÃ©sidences principales de 1991 Ã  2005',
#       'RÃ©sidences principales de 2006 Ã  2010',
#       'RÃ©sidences principales aprÃ¨s 2011', 'Taux de chauffage Ã©lectrique']

iris_enedis = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\Conso\consommation-electrique-par-secteur-dactivite-iris.csv', 
                          engine='python', sep=';')
iris_enedis = iris_enedis.drop(['geom', 'geo_point_2d'], axis=1)
iris_enedis = iris_enedis.set_index('Code IRIS')
cols = ['Annee', 'IRIS_Name', 'IRIS_TYPE', 'COMM_NAME', 'COMM_CODE',
       'EPCI_NAME', 'EPCI_CODE', 'EPCI_TYPE', 'DEP_NAME',
       'Departement', 'REGION_NAME', 'REGION_CODE',
       'Nb_RES', 'Conso_RES',
       'Conso_moyenne_RES',
       'Conso_totale_RES_theromosensible',
       'Conso_totale_RES_non_theromosensible',
       'Conso_moyenne_RES_theromosensible',
       'Conso_moyenne_RES_non_theromosensible',
       'Part_thermosensible_RES',
       'Thermosensibilite_tot_RES_kWh_DJU',
       'Thermosensibilite_moyenneRES_kWh_DJU',
       'Conso_tot_corrigee_alea_climatique',
       'Conso_moy_corrigee_alea_climatique',
       'Nb_PRO', 'Conso_PRO',
       'Conso_moyenne_PRO', 'DJU', 'Nb_Agriculture',
       'Conso_Agriculture', 'Nb_Industrie',
       'Conso_Industrie', 'Nb_Tertiaire',
       'Conso_Tertiaire', 'Nb_Autres',
       'Conso_Autres', 'Habitants',
       'Taux_logements_collectifs', 'Taux_residences_principales',
       'Logements_inf_30m2','Logements_30_40m2',
       'Logements_40_60m2','Logements_60_80m2',
       'Logements_80_100m2','Logements_sup_100m2',
       'Residences_principales_1919',
       'Residences_principales_1919_1945',
       'Residences_principales_1946_1970',
       'Residences_principales_1971_1990',
       'Residences_principales_1991_2005',
       'Residences_principales_2006_2010',
       'Residences_principales_2011', 'Taux_chauffage_elec']
iris_enedis.columns = cols

#% correct communes code and name for Paris, lyon and Marseille

cs = {'Paris' : 75100, 'Lyon': 69380, 'Marseille':13200}
for c in cs:
    iris = pd.Series(iris_enedis[iris_enedis.COMM_NAME == c].index, index=iris_enedis[iris_enedis.COMM_NAME == c].index)
    arr_code = iris.apply(lambda x: int(x[0:5]))
    name = arr_code.apply(lambda x: c + ' ' + str(x%cs[c]))
    iris_enedis.loc[iris, ['COMM_CODE']] = arr_code
    iris_enedis.loc[iris, ['COMM_NAME']] = name

#75056	France	Paris // 751xx
#13055	France	Marseille // 132xx
#69123	France	Lyon // 6938x

consos = ['Conso_RES', 'Conso_PRO', 'Conso_Industrie', 'Conso_Agriculture', 'Conso_Tertiaire', 'Conso_Autres']
iris_enedis[consos] = iris_enedis[consos].replace(np.nan, 0)


c_tot = iris_enedis[consos].sum()
naff = iris_enedis[iris_enedis.IRIS_Name == 'Non affectÃ©']
c_naff = naff[consos].sum()



#%% add non affectés to the other IRIS
diffs = {}
naff = naff[naff[consos].sum(axis=1) > 2000]
test = iris_enedis[consos]
for iris in naff.index:
    na = naff.loc[iris, consos]
    comm =  naff.COMM_CODE[iris]
    aff = iris_enedis[(iris_enedis.COMM_CODE == comm) & (iris_enedis.IRIS_Name != 'Non affectÃ©')][consos]
    ratios = (aff/aff.sum()).replace(np.nan, 0)
    extra_load = (ratios * na)
    new_load = aff + extra_load
    iris_enedis.loc[aff.index, consos] = new_load

#%% Check diffs:
cs = {}
for comm in naff.COMM_CODE:
    na = naff[(naff.COMM_CODE == comm)][consos].sum().sum()
    aff = iris_enedis[(iris_enedis.COMM_CODE == comm) & (iris_enedis.IRIS_Name != 'Non affectÃ©')][consos]
    orig = test.loc[aff.index, consos].sum().sum()
    new = aff.sum().sum()
    if np.abs(new - (orig + na)) > 1:
        cs[comm] = np.abs(new - (orig + na))
        print(comm, np.abs(new - (orig + na)))
        
#%% save
# drop non affectés
iris_enedis = iris_enedis.drop(iris_enedis[iris_enedis.IRIS_Name == 'Non affectÃ©'].index)
iris_enedis['Load'] = iris_enedis[consos].sum(axis=1)/1000
iris_enedis.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\IRIS_enedis_2017.csv')
