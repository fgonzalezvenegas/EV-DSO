# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:29:20 2019
Cleaning Enedis file of demand by IRIS
@author: U546416
"""

import pandas as pd
import numpy as np
import util
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
cols = ['Annee', 'IRIS_NAME', 'IRIS_TYPE', 'COMM_NAME', 'COMM_CODE',
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
naff = iris_enedis[iris_enedis.IRIS_NAME == 'Non affectÃ©']
iris_aff = iris_enedis[iris_enedis.IRIS_NAME != 'Non affectÃ©']
iris_aff.index = iris_aff.index.astype(int)
c_naff = naff[consos].sum()

print('Conso totale:', c_tot.sum() ,'\n', c_tot)
print('Conso naff:', c_naff.sum() ,'\n', c_naff)

#%% Correcting wrong encodings:
names = ['IRIS_NAME', 'COMM_NAME', 'DEP_NAME', 'EPCI_NAME', 'REGION_NAME']
for n in names:
    iris_aff[n] = util.fix_wrong_encoding_str(iris_aff[n])

#%% Add IRIS that are not included in the Enedis file, but should be. 
# They will have 0 conso (helps for mapping)

iris_poly = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\IRIS_all_geo_2016.csv',
                        engine='python', index_col=0)
print('Finished reading')
polygons = util.do_polygons(iris_poly)
print('Adding missing IRIS')
extra_iris = iris_poly[(iris_poly.GRD == 'Enedis') & (iris_poly.index.isin(iris_aff.index) == False)][['COMM_CODE', 'COMM_NAME', 'IRIS_NAME', 'IRIS_TYPE']]

iris_aff = iris_aff.append(extra_iris).replace(np.nan,0)

#%% Correcting population data, from INSEE 20xx Recensement de la population

print('Correcting population data')
population = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\DataGeo\base-ic-evol-struct-pop-2014.csv',
                           engine='python', index_col=0, sep=';', decimal=',')
# Correct Habitants
iris_aff.Habitants = population.P14_POP[iris_aff.index]

#%% Adding non affected RES conso to other IRIS, based on population
res = naff[naff.Conso_RES > 0]

for i in res.index:
    comm = res.COMM_CODE[i]
    irises = iris_aff[iris_aff.COMM_CODE==comm]
    
    cres = irises.Conso_RES.sum() + res.Conso_RES[i]            # conso total
    c_hab = cres / irises.Habitants.sum()                       # conso per hab global
#    c_hab_i = irises.Conso_RES / Habitants[irises.index]# conso per hab per iris
    proy_conso = c_hab * irises.Habitants                       # proyected conso per iris at avg conso at commune
    delta_conso = (proy_conso - irises.Conso_RES).clip_lower(0) # extra conso that is missing, clipped at 0
    ratios = delta_conso/delta_conso.sum()                      # share of conso at each IRIS
    add_conso = res.Conso_RES[i] * ratios                       # RES conso that will be added to each IRIS
    add_Nb = res.Nb_RES[i] * ratios                             # RES PDLs that will be added to eahc IRIS
    iris_aff.loc[irises.index, 'Conso_RES'] = irises.Conso_RES + add_conso
    iris_aff.loc[irises.index, 'Nb_RES'] = irises.Nb_RES + add_Nb
    #print(c_hab_i, c_hab)


#%% add Industry, Agriculture, Tertiary and PRO non affectés to the other IRIS, based on already existing conso
#diffs = {}
consos = util.consos[1:]
naff = naff[naff[consos].sum(axis=1) > 1000]
test = iris_enedis[consos]
for iris in naff.index:
    na = naff.loc[iris, consos]
    comm =  naff.COMM_CODE[iris]
    irises = iris_aff[iris_aff.COMM_CODE == comm][consos]
    ratios = (irises/irises.sum()).replace(np.nan, 0)
    extra_load = (ratios * na)
    new_load = irises + extra_load
    iris_aff.loc[irises.index, consos] = new_load

#%% Adding useful info
# Total load in GWh
iris_aff['Load_GWh'] = iris_aff[util.consos].sum(axis=1)/1000

# Add latitude and longitude
iris_aff['Lat'] = iris_poly.Lat[iris_aff.index]
iris_aff['Lon'] = iris_poly.Lon[iris_aff.index]

#%% Add usefull info:
# Proportion of workers and residents in each IRIS
# Proportion of Habitants of commune in each iris 
print('Adding Hab and workers per comm')
hab_com = iris_aff[['COMM_CODE','Habitants']].groupby('COMM_CODE').sum().squeeze()
hab_com  = hab_com[iris_aff.COMM_CODE]
hab_com.index = iris_aff.index
irishabpu = iris_aff.Habitants / hab_com
iris_aff['Hab_pu'] =  irishabpu

# Proportion of workers
consos = ['Conso_PRO', 'Conso_Industrie', 'Conso_Agriculture', 'Conso_Tertiaire']
ratios_w_conso = pd.Series(data=[7,4.5,0.3,13], index=consos)
ratios_w_conso = ratios_w_conso * 1e6 / iris_aff[consos].sum() # This gives a ratio of Workers/MWh for each type of conso
w_com = (iris_aff[['COMM_CODE'] + consos].groupby('COMM_CODE').sum() * ratios_w_conso).sum(axis=1).squeeze()
w_com = w_com[iris_aff.COMM_CODE] + 0.0001
w_com.index = iris_aff.index
irisworkerpu = ((iris_aff[consos] * ratios_w_conso).sum(axis=1).squeeze() + 0.0001)/ w_com

iris_aff['Work_pu'] = irisworkerpu

# %%Number of cars per IRIS - Using census data INSEE, "millesime 2014, with IRIS zoning from 2016"
print('Adding number of parkings, vehicles and residences')
logs = pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Base\base-ic-logement-2014.csv', engine='python', sep=';', decimal=',' , index_col=0)
##%% Add Number of cars, Ratio of Parking per residence and Number of main residences
nvoit = logs.P14_RP_VOIT1.loc[iris_aff.index] + logs.P14_RP_VOIT2P.loc[iris_aff.index]*2
ratio_parking = logs.P14_RP_GARL.loc[iris_aff.index] / (logs.P14_RP_VOIT1P.loc[iris_aff.index] + 0.1) #+0.1 to avoid inf values
res_princ = logs.P14_RP.loc[iris_aff.index]

iris_aff['N_VOIT'] = nvoit
iris_aff['RATIO_PARKING'] = ratio_parking
iris_aff['RES_PRINC'] = res_princ
print('Total Cars', nvoit.sum())
print('Avg Ratio_Parking', (nvoit * ratio_parking).sum()/nvoit.sum())
print('Avg Cars per Main residence', nvoit.sum()/res_princ.sum())

#%% Save Conso Enedis 2017, avec la geographie 2016
print('Saving')
iris_aff.to_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\IRIS_enedis_2017.csv')