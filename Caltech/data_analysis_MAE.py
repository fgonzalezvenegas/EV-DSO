# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:10:41 2020
Analysis of data of My Electric Avenue project

Using it to estimate parameters of plug-in behavior model

@author: U546416
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import util

folder_data = r'c:\user\U546416\Documents\PhD\Data\MyElectricAvenue\\'
charge_data = pd.read_csv(folder_data + 'EVChargeData.csv',
                          engine='python')
charge_data.columns = ['ParticipantID', 'BatteryChargeStartDate', 'BatteryChargeStopDate',
       'StartingSOC', 'EndingSOC']
trip_data = pd.read_csv(folder_data + 'EVTripData.csv',
                          engine='python')
trip_data.columns = ['ParticipantID', 'TripStartDate', 'TripStopDate',
       'TripDistance_m', 'PowerCons_Wh',
       'Odometer_at_start_km']

participants = pd.read_csv(folder_data + 'Participants.csv',
                          engine='python', index_col=0)

# setting datetime columns
charge_data.BatteryChargeStartDate = pd.to_datetime(charge_data.BatteryChargeStartDate)
charge_data.BatteryChargeStopDate = pd.to_datetime(charge_data.BatteryChargeStopDate)

trip_data.TripStartDate = pd.to_datetime(trip_data.TripStartDate)
trip_data.TripStopDate = pd.to_datetime(trip_data.TripStopDate)

# Sorting and resetting index

charge_data.sort_values(['ParticipantID', 'BatteryChargeStartDate'], inplace=True)
charge_data.reset_index(drop=True, inplace=True)
trip_data.sort_values(['ParticipantID', 'TripStartDate'], inplace=True)
trip_data.reset_index(drop=True, inplace=True)

# Dropping users where there are data is too small
small_data = 50
count_trip = trip_data.ParticipantID.value_counts()
users = count_trip[count_trip<small_data].index
count_charge = charge_data.ParticipantID.value_counts()
users = users.append(count_trip[count_trip<small_data].index).unique()
trip_data = trip_data[~trip_data.ParticipantID.isin(users)]
charge_data = charge_data[~charge_data.ParticipantID.isin(users)]

## Charging measurements (I, V)
#icb_data = pd.read_csv(folder_data + 'ICBData.csv',
#                          engine='python')
## feeder measurement
#mc_data = pd.read_csv(folder_data + 'MCData.csv',
#                          engine='python')
## control (curtailment) measurements
#switch_states = pd.read_csv(folder_data + 'SwitchStates.csv',
#                          engine='python')


#%% Analyse consumption [kWh/km]
reliable_trip_data = trip_data[(trip_data.TripDistance_m > 1000) & (trip_data.PowerCons_Wh > 1)]
cons = reliable_trip_data.PowerCons_Wh/reliable_trip_data.TripDistance_m
plt.figure()
cons.hist(bins=np.arange(0,1,0.05))
plt.title('Consumption histogram [kWh/km]')
plt.figure()
plt.plot(reliable_trip_data.TripDistance_m/1000, cons, '.')
plt.xlim([0,150])
plt.ylim((0,1))
plt.xlabel('Trip distance [km]')
plt.ylabel('Consumtion [kWh/km]')

#%% separate conso per season:
seasons = {'winter' : [12,1,2],
           'spring': [3,4,5],
           'summer': [6,7,8],
           'fall': [9,10,11]}
cons_s = {}
f, ax = plt.subplots()
for s, m in seasons.items():
    data = reliable_trip_data[reliable_trip_data.TripStartDate.dt.month.isin(m)]
    cons_s[s] =  data.PowerCons_Wh/data.TripDistance_m
    # Plot histogram
    plt.figure()
    cons_s[s].hist(bins=np.arange(0,0.5,0.02))
    mean_conso = cons_s[s].mean()
    plt.axvline(mean_conso, color='r', linestyle='--')
    plt.text(x=mean_conso + 0.05, y=2000, s='Mean Conso = {:.3f}'.format(mean_conso))
    plt.title('Histogram of consumption [kWh/km] in {}'.format(s))
    plt.xlabel('Consumption [kWh/km]')
    plt.ylabel('# trips')
    # Plot data points 
    ax.plot(data.TripDistance_m/1000, cons_s[s], '.', label=s, alpha=(12-min(m)/12))
    ax.set_xlim([0,150])
    ax.set_ylim((0,1))
    ax.set_xlabel('Trip distance [km]')
    ax.set_ylabel('Consumtion [kWh/km]')
    ax.legend()

#% conso per participant

data_participants = reliable_trip_data.groupby('ParticipantID')[['TripDistance_m', 'PowerCons_Wh']].sum()    
data_participants['Conso_kWh_km'] = data_participants.PowerCons_Wh / data_participants.TripDistance_m
plt.figure()
data_participants.Conso_kWh_km.hist(bins=np.arange(0.1,0.3,0.01))
plt.title('Histogram of mean consumption [km/kWh] per EV user')
plt.xlabel('Consumption [kWh/km]')
plt.ylabel('# of EV users')
# TODO: analyze hour of connection (wd/saturday/sunday)
# TODO: Analyze day of connection
# TODO: Macro analysis of # charging/EV/day.week

#%% Analysis on Workplace chargers
yh = participants[participants.index.str.contains('YH')].index
yh_data = charge_data[charge_data.ParticipantID.isin(yh)]
init_ch = yh_data.BatteryChargeStartDate.dt.hour + yh_data.BatteryChargeStartDate.dt.minute/60
stop_ch = yh_data.BatteryChargeStopDate.dt.hour + yh_data.BatteryChargeStopDate.dt.minute/60

# Histogram of arrival/departure
plt.figure()
hist, _, _ = np.histogram2d(init_ch, stop_ch, bins=np.arange(0,24.5,0.5))
plt.imshow(hist.T, origin='bottom', imlim=((0,24),(0,24)) )
plt.title('Start and End of charging sessions')

# Histogram of charging session
plt.figure()
dt_ch = (stop_ch - init_ch)%24
dt_ch.hist(bins=np.arange(0,12,0.25))
plt.title('Duration of charging session')

#%% Analysis on other chargers
not_yh = charge_data[~charge_data.ParticipantID.isin(yh)]
init_ch = not_yh.BatteryChargeStartDate.dt.hour + not_yh.BatteryChargeStartDate.dt.minute/60
stop_ch = not_yh.BatteryChargeStopDate.dt.hour + not_yh.BatteryChargeStopDate.dt.minute/60


# Histogram of arrival/departure
plt.figure()
hist, _, _ = np.histogram2d(init_ch, stop_ch, bins=np.arange(0,24.5,0.5))
plt.imshow(hist.T, origin='bottom', imlim=((0,24),(0,24)) )
plt.title('Start and End of charging sessions')

# Histogram of charging session
plt.figure()
dt_ch = (stop_ch - init_ch)%24
dt_ch.hist(bins=np.arange(0,12,0.25))
plt.title('Duration of charging session')


#%% Create activity DF: it contains the driving and charging IDs for each ev

activity = pd.DataFrame(columns=('ParticipantID', 'Activity', 'EventID', 
                                 'StartTime', 'EndTime', 
                                 'SOC_ini', 'SOC_end', 'delta_SOC', 'dkm'))
print('Creating activity DF')
print('Adding Charge events')
df = []
#charge_data.sort_values(['ParticipantID', 'BatteryChargeStartDate'], inplace=True)
#charge_data.reset_index(drop=True, inplace=True)
#trip_data.sort_values(['ParticipantID', 'TripStartDate'], inplace=True)
#charge_data.reset_index(drop=True, inplace=True)

for i,t in charge_data.iterrows():
    if i%10000 == 0:
        print('\t{} out of {}'.format(i, charge_data.shape[0]))
    df.append(dict(ParticipantID=t.ParticipantID, 
                         Activity='Charge', 
                         EventID=i, 
                         StartTime=t.BatteryChargeStartDate, 
                         EndTime=t.BatteryChargeStopDate, 
                         SOC_ini=t.StartingSOC/12, 
                         SOC_end=t.EndingSOC/12, 
                         delta_SOC=(t.EndingSOC-t.StartingSOC)/12, 
                         dkm=0))
df = pd.DataFrame(df, columns=('ParticipantID', 'Activity', 'EventID', 
                                 'StartTime', 'EndTime', 
                                 'SOC_ini', 'SOC_end', 'delta_SOC', 'dkm'))
activity = activity.append(df, ignore_index=True)
idp = 0
exp_odo = 0
print('Adding driving events')
df = []
for i,t in trip_data.iterrows():
    if i%10000 == 0:
        print('\t{} out of {}'.format(i, trip_data.shape[0]))
    if t.PowerCons_Wh < 25:
        pass
    # Check if same participant than previous row    
    if t.ParticipantID == idp:
        # Check if there are no trips missing
        if t.Odometer_at_start_km > exp_odo + 2: #2km of margin for error
            df.append(dict(ParticipantID=t.ParticipantID, 
                         Activity='MissingTrip', 
                         EventID=i, 
                         StartTime=activity[(activity.EndTime < t.TripStartDate) & 
                                            (activity.ParticipantID == idp)].EndTime.max(), 
                         EndTime=t.TripStartDate, 
                         SOC_ini=np.nan, 
                         SOC_end=np.nan, 
                         delta_SOC=np.nan, 
                         dkm=t.Odometer_at_start_km - exp_odo))
#            if (df[-1]['StartTime'] == df[-1]['StartTime']) == False:
                
    if t.TripDistance_m < 1000:
            df.append(dict(ParticipantID=t.ParticipantID, 
                             Activity='OtherAct', 
                             EventID=i, 
                             StartTime=t.TripStartDate, 
                             EndTime=t.TripStopDate, 
                             SOC_ini=np.nan, 
                             SOC_end=np.nan, 
                             delta_SOC=-t.PowerCons_Wh/24000, 
                             dkm=0))
    else:
        df.append(dict(ParticipantID=t.ParticipantID, 
                         Activity='Trip', 
                         EventID=i, 
                         StartTime=t.TripStartDate, 
                         EndTime=t.TripStopDate, 
                         SOC_ini=np.nan, 
                         SOC_end=np.nan, 
                         delta_SOC=-t.PowerCons_Wh/24000, 
                         dkm=t.TripDistance_m/1000))
    if idp != t.ParticipantID:    
        exp_odo = t.Odometer_at_start_km + t.TripDistance_m/1000
        idp = t.ParticipantID
    else:
        exp_odo = max(t.Odometer_at_start_km, exp_odo) + t.TripDistance_m/1000
    
df = pd.DataFrame(df, columns=('ParticipantID', 'Activity', 'EventID', 
                                 'StartTime', 'EndTime', 
                                 'SOC_ini', 'SOC_end', 'delta_SOC', 'dkm'))
activity = activity.append(df, ignore_index=True)
activity.sort_values(['ParticipantID','StartTime'], inplace=True)
activity.reset_index(inplace=True, drop=True)

#%% Cleaning activity DF:

# Correct NaT from Missing trips and delta_SOC


month_to_season = {1:'winter',2:'winter',3:'spring', 
                   4:'spring', 5:'spring', 6:'summer', 
                   7:'summer', 8:'summer', 9:'fall',
                   10:'fall', 11:'fall', 12:'winter'}
cons_season = dict()
for s, v in cons_s.items():
    cons_season[s] = v.mean()

for i, t in activity[activity.StartTime.isnull()].iterrows():
    activity.StartTime[i] = activity[(activity.EndTime < t.EndTime) & 
                                     (activity.ParticipantID == t.ParticipantID)].EndTime.max()
for i, t in activity[activity.Activity == 'MissingTrip'].iterrows():
    activity.delta_SOC[i] = -t.dkm * cons_season[month_to_season[t.StartTime.month]]/24

# droping trips or charges before there are real charges (or charges before trips?)

   
# Recreating SOC after activities 
idp = 0
SOC_end = 1
for i, t in activity.iterrows():
    if i % 10000 == 0:
        print('\t{} out of {}'.format(i, activity.shape[0]))
    if idp == t.ParticipantID:
        if t.Activity in ['MissingTrip', 'OtherAct', 'Trip']:
            activity.SOC_ini[i] = SOC_end
            SOC_end = max(0, SOC_end - t.delta_SOC)
            activity.SOC_end[i] = SOC_end
    else: 
        idp = t.ParticipantID
        SOC_end = t.SOC_end

# Adding useful data:
activity['Weekday'] = activity.StartTime.dt.weekday #Monday 0- Sunday 6
activity['InitTime'] = activity.StartTime.dt.hour + activity.StartTime.dt.minute/60
activity['ActDuration'] = activity.EndTime - activity.StartTime 
activity['Time_since_last_event'] = activity.StartTime.diff()   

#%% Define available dates for EVS:
# for each ev: get Init date - end Date
# Eliminate periods where there are big gaps: ex. hollidays
# Compute avg params for those dates


#%% General analysis on trip and charging events

plt.figure()
charge_data.BatteryChargeStartDate.dt.weekday.hist(bins=np.arange(0,8,1))
plt.title('Histograms of charging events per week day')
plt.xticks(np.arange(0.5,7,1), util.dsnms)

plt.figure()
trip_data[trip_data.TripDistance_m > 1000].TripStartDate.dt.weekday.hist(bins=np.arange(0,8,1))
plt.title('Histograms of trip events per week day (>1km)')
plt.xticks(np.arange(0.5,7,1), util.dsnms)