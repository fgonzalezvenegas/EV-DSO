# -*- coding: utf-8 -*-
""" Useful functions for pandas treatement, or pyplot plotting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe
#import assign_ss_modif as ass_ss
import scipy.stats as stats
import datetime as dt
import os
import importlib
#import util

# PARAMS

# Constants
k = 1e3
M = 1e6

daysnames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dsnms = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
monthnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# For plotting
deps_idf = [75, 77, 78, 91, 92, 93,  94, 95]
cns_idf = ['Paris', 'Versailles', 'Évry', 'Meaux', 'Nemours', 'Cergy', 'Étampes', 'Provins']
latlons_idf = [[2.3424567382662334, 48.859626443036575],
         [2.131139599462395, 48.813017693582793],
         [2.4419262056107969, 48.632343164682723],
         [2.9197438849044732, 48.952766254606502],
         [2.7070194793010449, 48.26850934641633],
         [2.0698102422679203, 49.037687672577924],
         [2.1386667699798143, 48.435164427848107],
         [3.2930400953107446, 48.544799959855652]]
cns_fr = ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Bordeaux', 'Nantes', 'Lille', 'Rennes', 'Strasbourg']
latlons_fr = [[2.3424567382662334, 48.859626443036575],
         [5.3863214053108095, 43.300743046351528],
         [4.8363757561790415, 45.771993345448962],
         [1.4194663657069806, 43.58313856938689],
         [-0.58251346635799961, 44.856834056176488],
         [-1.5448118357936005, 47.227505954238453],
         [2.98834511866088, 50.651686273910592],
         [-1.6966383042920521, 48.083113659214533],
         [7.748341, 48.584826]]

# Global params
consos = ['Conso_RES', 'Conso_PRO', 'Conso_Agriculture', 'Conso_Industrie', 'Conso_Tertiaire']
nb_pdl = ['Nb_RES', 'Nb_PRO', 'Nb_Agriculture', 'Nb_Industrie', 'Nb_Tertiaire']

iris_cols = ['Annee', 'IRIS_NAME', 'IRIS_TYPE', 'COMM_NAME', 'COMM_CODE',
       'EPCI_NAME', 'EPCI_CODE', 'EPCI_TYPE', 'DEP_NAME', 'Departement',
       'REGION_NAME', 'REGION_CODE', 'Nb_RES', 'Conso_RES',
       'Conso_moyenne_RES', 'Conso_totale_RES_theromosensible',
       'Conso_totale_RES_non_theromosensible',
       'Conso_moyenne_RES_theromosensible',
       'Conso_moyenne_RES_non_theromosensible', 'Part_thermosensible_RES',
       'Thermosensibilite_tot_RES_kWh_DJU',
       'Thermosensibilite_moyenneRES_kWh_DJU',
       'Conso_tot_corrigee_alea_climatique',
       'Conso_moy_corrigee_alea_climatique', 'Nb_PRO', 'Conso_PRO',
       'Conso_moyenne_PRO', 'DJU', 'Nb_Agriculture', 'Conso_Agriculture',
       'Nb_Industrie', 'Conso_Industrie', 'Nb_Tertiaire', 'Conso_Tertiaire',
       'Nb_Autres', 'Conso_Autres', 'Habitants', 'Taux_logements_collectifs',
       'Taux_residences_principales', 'Logements_inf_30m2',
       'Logements_30_40m2', 'Logements_40_60m2', 'Logements_60_80m2',
       'Logements_80_100m2', 'Logements_sup_100m2',
       'Residences_principales_1919', 'Residences_principales_1919_1945',
       'Residences_principales_1946_1970', 'Residences_principales_1971_1990',
       'Residences_principales_1991_2005', 'Residences_principales_2006_2010',
       'Residences_principales_2011', 'Taux_chauffage_elec', 'Lat', 'Lon',
       'Load_GWh', 'SS', 'hab_pu', 'w_pu', 'N_VOIT', 'RATIO_PARKING', 'RES_PRINC']

# FUNCTIONS

def plot_polygons(polys, ax='', **kwargs):
    """ Plot a list of polygons into the axis ax
    kwargs as arguments for PolygonCollection
    """
    if ax == '':
        f, ax = plt.subplots()
    collection = PatchCollection(polys, **kwargs)
    ax.add_collection(collection)
    ax.autoscale()
    return ax
    
def plot_segments(segments, ax='', loop=True, ends=False, **kwargs):
    if ax=='':
        f, ax = plt.subplots()
    for s in segments:
        if loop:
            s.append(s[0]) 
        x = np.array(s)[:,0]
        y = np.array(s)[:,1]
        ax.plot(x,y,**kwargs)
        if ends:
            ax.plot([x[0],x[-1]], [y[0],y[-1]], 'r*')
    return ax

def plot_arr_dep_hist(hist, binsh=np.arange(0,24.5,0.5), ftitle=''):
    """ Plots arrival and departure histogram
    """
    f, (ax, ax2) = plt.subplots(1,2)
    i = ax.imshow(hist.T/hist.sum().sum(), origin='lower', extent=(0,24,0,24))
    ax.set_title('Distribution of sessions')
    ax.set_xlabel('Start of charging sessions')
    ax.set_ylabel('End of charging sessions')
    ax.set_xticks(np.arange(0,25,2))
    ax.set_yticks(np.arange(0,25,2))
    ax.set_xticklabels(np.arange(0,25,2))
    ax.set_yticklabels(np.arange(0,25,2))
    plt.colorbar(i, ax=ax)
    
    ax2.bar((binsh[:-1]+binsh[1:])/2, hist.sum(axis=1)/hist.sum().sum(), width=0.5, label='Arrivals')
    ax2.bar((binsh[:-1]+binsh[1:])/2, -hist.sum(axis=0)/hist.sum().sum(), width=0.5, label='Departures')
    ax2.set_xlim(0,24)
    ax2.set_xticks(np.arange(0,25,2))
    ax2.set_xticklabels(np.arange(0,25,2))
    ax2.set_title('Arrival and departure distribution')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Distribution')
    ax2.legend()
    ax2.grid()
    f.suptitle(ftitle)
    f.set_size_inches(11.92,4.43)

def length_segment_WGS84(segment, unit='m'):
    """ Returns the length in meters of a segment
    of points in GPS coordinates [(lon, lat)_i, ....] """
    k = dict(km=1,
             m=1000)[unit]
    l = 0
    p0 = segment[0]
    for p in segment[1:]:
        l += computeDist(p0, p)
        p0 = p
    return l * k
    
def fix_wrong_encoding_str(pdSeries):
    """
    """
    # é, è, ê, ë, É, È
    out = pdSeries.apply(lambda x: x.replace('Ã©', 'é').replace('Ã¨', 'è').replace('Ãª', 'ê').replace('Ã‰', 'É').replace('Ã«','ë').replace('Ãˆ','È'))
    # Î, î, ï
    out = out.apply(lambda x: x.replace('ÃŽ', 'Î').replace('Ã®', 'î').replace('Ã¯', 'ï'))
    #ÿ
    out = out.apply(lambda x: x.replace('Ã½', 'ÿ'))
    # ç
    out = out.apply(lambda x: x.replace('Ã§', 'ç'))
    # ô
    out = out.apply(lambda x: x.replace('Ã´', 'ô'))
    # û
    out = out.apply(lambda x: x.replace('Ã»', 'û').replace('Ã¼', 'ü'))
    # â, à
    out = out.apply(lambda x: x.replace('Ã¢', 'â').replace('Ã\xa0', 'à'))
    
    return out



def load_polygons_iris(year=2016, folder='', file=''):
    if folder =='':
        folder = r'c:\user\U546416\Documents\PhD\Data\DataGeo\\'
    if file=='':
        file = 'IRIS_all_geo_'+str(year)+'.csv'
    iris_poly = pd.read_csv(folder+file,
                        engine='python', index_col=0)
    return do_polygons(iris_poly, plot=True)

def load_polygons_SS(year=2016, folder='', file=''):
    if folder == '':
        folder = r'c:/user/U546416/Documents/PhD/Data/Mobilité/Data_Traitee/Reseau//'
    if file == '':
        file = 'postes_source_polygons.csv'
    SS_poly = pd.read_csv(folder+file,
                        engine='python', index_col=0)
    return do_polygons(SS_poly, plot=True)

def load_data(data='iris'):
    """ data=iris; BT; etc
    """
    if data in ['iris']:
        return pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Conso\IRIS_enedis_2017.csv', 
                    engine='python', index_col=0)
    if data in ['ss', 'SS']:
        return pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau\postes_source.csv', 
                    engine='python', index_col=0)
    if data in ['BT', 'bt']:
        return pd.read_csv(r'C:\user\U546416\Documents\PhD\Data\Mobilité\Data_Traitee\Reseau\postes_BT.csv',
                           engine='python', index_col=0)
    if data in ['Nb_BT', 'nb_bt']:
        return pd.read_csv(r'c:\user\U546416\Documents\PhD\Data\Conso-Reseau\Réseau\nombre_bts_IRIS2017.csv',
                           engine='python', index_col=0)


def list_polygons(polygons, index):
    return [p for i in index for p in polygons[i]]

def do_polygons(df, plot=True):
    """ Do polygons from df or pdSeries
    """
    if type(df.Polygon.iloc[0]) == str:
        df.Polygon = df.Polygon.apply(lambda x: eval(x))
    #print(type(df.Polygon.iloc[0]))
    polygons = {c: [ptc.Polygon(p) for p in df.Polygon[c] if len(p) > 1] for c in df.index}
    if plot:
        plot_polygons([p for pp in polygons.values() for p in pp])
    return polygons

def compute_load_from_ss(energydata, profiledata, ss):
    """Returns the load profile for the substation ss, 
    where Substation data is stored in SS DataFrame (namely communes assigned) 
    and load data in load_profiles and load_by_comm
    """
    energy_types = ['Conso_RES', 'Conso_PRO', 'Conso_Agriculture', 'Conso_Industrie', 'Conso_Tertiaire']
    profiles = ['RES', 'PRO', 'Agriculture', 'Industrie', 'Tertiaire']
    
    # These factors are the total consumption for a year for all the components of energydata df 
    # associated to the substation ss
    factors = energydata[energydata.SS == ss][energy_types].sum()
    mwhy_to_mw = 1/8760 
    factors.index = profiles
    #print(factors)
    
    return (profiledata[profiles] * factors * mwhy_to_mw).sum(axis=1)

def aspect_carte_france(ax, title='', palette=None,
                       labels=None,
                        cns='France', latlons='', delta_cns=0.2):
#    if palette==None:
#        palette = ['b','lightgreen', 'forestgreen', 'khaki', 'gold', 'orange', 'r']
#    if labels ==None:
#        wbin = 15
#        labels=[str(i * wbin) + '<d<' + str((i+1)*wbin) for i in range(len(palette))]
    if cns=='France':
        cns = cns_fr
        latlons = latlons_fr
    if cns == 'idf':
        cns = cns_idf
        latlons = latlons_idf
        delta_cns=0
        
    ax.set_title(title)
    ax.autoscale()
    a = ax.axis()
    ax.set_aspect(compute_aspect_carte(*a))
    # Do labels
    if labels != None:
        do_labels(labels, palette, ax)
    # Write the name of some cities
    for i in range(len(cns)):
        ax.text(latlons[i][0],latlons[i][1]+delta_cns, cns[i], ha='center',
           path_effects=[pe.withStroke(linewidth=2, foreground='w')])

def do_labels(labels, palette, ax, f=None):
    a = ax.axis()
    for i in range(len(palette)):
            ax.plot(0,0,'s', color=palette[i], label=labels[i])
    a=ax.axis(a)
    if f is None:
        ax.legend(loc=3)
    else:
        ax.figure.legend(loc=5)


def compute_lognorm_cdf(hist, bins='', params=False, plot=False, ax=None):
    """ Returns a fitted CDF of a lognormal distribution, from a given histogram of distances
    If params=True, returns the parameters of the fitted lognorm
    """
    if bins == '':
        bins = [i*2 for i in range(len(hist)+1)]
    db = bins[1]-bins[0]
    points = [bins[i] + db/2 for i in range(len(hist)) for j in range(int(np.round(hist[i],0)))]
    #return points
    s, loc, scale = stats.lognorm.fit(points)
    if plot:
        if ax==None:
            f,ax = plt.subplots()
        pdf = stats.lognorm.pdf(bins, s, loc, scale)
        ax.plot(bins, pdf / sum(pdf))
        ax.bar(bins[:-1], hist/hist.sum())
    if params:
        return {'s':s, 'loc':loc, 'scale':scale}
    cdf = stats.lognorm.cdf(bins, s, loc, scale)
    return cdf/cdf[-1], bins
    
    
def get_max_load_week(load, step=30, buffer_before=0, buffer_after=0, extra_t=0):
    """ Returns the week of max load. It adds buffer days before and after
    load needs to be a pandas Series
    """
    if type(load) == pd.core.frame.DataFrame:
        load = load.squeeze()
    if type(load.index[0]) == str:
        fmtdt = '%Y-%m-%d %H:%M:%S%z'
        #parse!
        load.index = load.index.map(lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt))
    idmax = load.idxmax()
    dwmax = idmax.weekday()
    dini = idmax - dt.timedelta(days=dwmax+buffer_before, hours=idmax.hour, minutes=idmax.minute)
    dend = dini + dt.timedelta(days=7+buffer_after+buffer_before) - dt.timedelta(minutes=(1-extra_t)*step)
    return load.loc[dini:dend]

def period_to_year(period, dt_ini=0, step=30):
    """ repeats the period vector to a full year, returning it with dt_ini days of delay
    """
    days_p = len(period) / (24 * 60/step)
    if days_p % 1 > 0:
        raise ValueError('Invalid length of period to repeat. Needs to be a full day(s)')
    year = np.tile(period, int(np.ceil(365/days_p)+1))
    return year[int(dt_ini * 24 * 60/step):int((dt_ini + 365) * 24 * 60/step)]

def hist_ovl(load, max_load, h_nsteps=4):
    """ Returns an histogram of the lengths of overloads
    """
    ovl = load > max_load
    k = 0
    len_ovl = []
    for j in ovl:
        if j:
            k+=1
        elif k>0:
            len_ovl.append(k)
            k=0
    return np.histogram(len_ovl, bins=[i for i in range(0, h_nsteps+1)])
    
def evaluate_max_load(base_load, ev_load, max_load, step=30):
    """ Evaluates the impact of a x days ev load for the full year.
    Returns: max load, hours overload at 80,90,100%, and 
    histograms of duration of overload at 100% by 1,2,3,4+ steps 
    """
    if type(base_load) in [pd.core.frame.DataFrame, pd.core.frame.Series]:
        base_load = base_load.values.squeeze()
    load = base_load + ev_load
    peak_load = load.max()
    h_ovl = [(load > (max_load * 0.8)).sum() * step / 60,
             (load > (max_load * 0.9)).sum() * step / 60,
             (load > (max_load * 1.0)).sum() * step / 60]
    return peak_load, h_ovl, hist_ovl(load, max_load), load

def interpolate(data, step=15, **kwargs):
    """ Returns the data with a greater time resolution, by interpolating it
    """
    if type(data.index[0]) == str:
        fmtdt = '%Y-%m-%d %H:%M:%S%z'
        #parse!
        data.index = data.index.map(lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt))
    if (data.index[1]-data.index[0])/dt.timedelta(minutes=step) % 1>0:
        raise ValueError('Invalid step, it should be a divisor of data step')
    return data.asfreq(freq=dt.timedelta(minutes=step)).interpolate(**kwargs)
    
def computeDist(latlon1, latlon2):
    """Computes pythagorean distance between 2 points (need to be np.arrays)
    """
    if type(latlon1) == list:
        latlon1 = np.array(latlon1)
        latlon2 = np.array(latlon2)
    radius=6371
    latlon1 = latlon1 * np.pi/180
    latlon2 = latlon2 * np.pi/180
    deltaLatLon = (latlon2-latlon1)
    x = deltaLatLon[1] * np.cos((latlon1[0]+latlon2[0])/2)
    y = deltaLatLon[0]
    return radius*np.sqrt(x*x + y*y)

def sec_to_time(s):
    """ Returns the hours, minutes and seconds of a given time in secs
    """
    return (int(s//3600), int((s//60)%60), (s%60))
    
def compute_aspect_carte(lon1, lon2, lat1, lat2):
    """ Sets the ratio of height/width for WGPS-based maps
    """   
    lat0, lon0 = (lat1+lat2)/2, (lon1+lon2)/2
    km_per_lat = computeDist([lat1, lon0], [lat2, lon0]) / abs(lat1-lat2)
    km_per_lon = computeDist([lat0, lon1], [lat0, lon2]) / abs(lon1-lon2)
    return km_per_lat / km_per_lon

def create_folder(path, *folders):
    """ Creates folder in given path
    """
    newpath = path + ''.join(map(str, [r'\\' + f for f in folders])) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def self_reload(module=None):
    """ Reloads a module. Useful for debugging and developing
    """
    if module == None:
        importlib.reload(util)
    else:
        importlib.reload(module)

def input_y_n(message):
    while True:
        v =  input(message + ' (Y/N)')
        if v in ['Y', 'y', 'N', 'n', True, False]:
            return v
def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in diff_segments(p)))

def diff_segments(p):
    return zip(p, p[1:] + [p[0]])