# -*- coding: utf-8 -*-

import pandas as pd

file = r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\ProfilesRES.xlsx'
sheet = 'conso-resPU-AugRES1'
#load residential p.u. profiles
profiles = pd.read_excel(file, sheetname=sheet)
profiles.columns = ['horodate', 'RES1', 'RES11', 'RES2', 'RES2WE', 'RES3', 'RES4']
profiles.index = profiles.horodate
#define study cases
cases = {'low_elec': {'RES1': 0.8, 'RES11':0.2, 'RES2':0},
         'medium_elec': {'RES1': 0.4, 'RES11':0.2, 'RES2':0.4},
         'high_elec': {'RES1': 0, 'RES11':0.2, 'RES2':0.8}}
#Create synthetic profiles
new_profiles = pd.DataFrame({i: sum([profiles[j] * cases[i][j] for j in cases[i]]) for i in cases})
#Get high demand day and plot
d = new_profiles.index.get_loc(new_profiles.idxmax()['low_elec'])//(48)
ax = new_profiles.iloc[d*48:(d+1)*48,:].plot()
ax.set_title('Profiles in the day of max demand')


#%%save! but in p.u. with respect to max demand in low electrification case
maxd = new_profiles.low_elec.max()
new_profiles = new_profiles / maxd
for i in new_profiles:
    new_profiles[i].to_csv(r"C:\Users\u546416\AnacondaProjects\OpenDSS\LoadProfiles\puProfiles\\" + i + '.csv', 
                header=False, index=False)

    

