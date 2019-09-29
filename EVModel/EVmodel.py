F# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:07:14 2019

EV Model:
    Creates Class GRID and Class EV
        
    GRID Class:
        A Set of EVs exists
        computes and plots general data
    EV Class:
        A single EV with its behaviour:
            

@author: U546416
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt

bins_dist = np.linspace(0, 100, num=51)
dist_function = np.sin(bins_dist[:-1]/ 100 * np.pi * 2) + 1
dist_function[10:15] = [0, 0, 0 , 0 , 0]
pdfunc = (dist_function/sum(dist_function)).cumsum()

bins_hours = np.arange(0,24,0.15)


def random_from_pdf(pdf, bins):
    """Returns a random bin value given a pdf.
    pdf has n values, and bins n+1, delimiting initial and final boundaries for each bin
    """
    r = np.random.rand(1)
    x = int(np.digitize(r, pdf))
    if pdf.max() > 1 or pdf.min() < 0:
        raise ValueError('pdf is not a valid probability distribution function')
    return bins[x] + np.random.rand(1) * (bins[x+1] - bins[x])


def load_conso_ss_data(folder = 'c:/user/U546416/Documents/PhD/Data/Mobilité/',
                       folder_load = 'Data_Traitee/Conso/',
                       folder_grid = 'Data_Traitee/Reseau/',
                       file_load_comm = 'consommation-electrique-par-secteur-dactivite-commune-red.csv',
                       file_load_profile = 'conso_all_pu.csv',
                       file_ss = 'postes_source.csv'):
    """ load data for conso and substations
    """ 
    # Load load by commune data
    load_by_comm = pd.read_csv(folder + folder_load + file_load_comm, 
                               engine='python', delimiter=';', index_col=0)
    load_by_comm.index = load_by_comm.index.astype(str)
    load_by_comm.index = load_by_comm.index.map(lambda x: x if len(x) == 5 else '0' + x)
    # Load load profiles data (in pu (power, not energy))
    load_profiles = pd.read_csv(folder + folder_load + file_load_profile, 
                               engine='python', delimiter=',', index_col=0)
    # drop ENT profile that's not useful
    load_profiles = load_profiles.drop('ENT', axis=1)
    # Load Trafo data
    SS = pd.read_csv(folder + folder_grid + file_ss, 
                               engine='python', delimiter=',', index_col=0)
    # parse communes
    SS.Communes = SS.Communes.apply(lambda x: eval(x) if x==x else [])
    return load_by_comm, load_profiles, SS


def load_hist_data(folder=r'c:/user/U546416/Documents/PhD/Data/Mobilité/', 
                   folder_hist='',
                   file_hist_home='HistHome.csv',
                   file_hist_work='HistWork.csv'):
    """ load histogram data for conso and substations
    """
    
    hist_home = pd.read_csv(folder + folder_hist+ file_hist_home, 
                               engine='python', delimiter=',', index_col=0)
    hist_work = pd.read_csv(folder + folder_hist+ file_hist_work, 
                               engine='python', delimiter=',', index_col=0)
    return hist_home, hist_work


def extract_hist(hist, comms):
    """ returns histograms for communes
    """
    labels = ['UU', 'ZE', 'Status', 'Dep']
    cols_h = hist.columns.drop(labels)
    return pd.DataFrame({c: np.asarray(hist.loc[c,cols_h]) for c in comms}).transpose()
    

def compute_load_from_ss(load_by_comm, load_profiles, SS, ss):
    """Returns the load profile for the substation ss, 
    where Substation data is stored in SS DataFrame (namely communes assigned) 
    and load data in load_profiles and load_by_comm
    """
    if not ss in SS.Communes:
        raise ValueError('Invalid Substation %s' %ss)
    comms = SS.Communes[ss]
    try: 
        factors = load_by_comm.loc[comms, load_profiles.columns].sum() / (8760)
    except:
        factors = pd.DataFrame({key: 0 for key in load_profiles.columns})
    return load_profiles * factors


def get_max_load_week(load, step=30, extradays=0):
    """ Returns the week of max load
    """
    if type(load.index[0]) == str:
        fmtdt = '%Y-%m-%d %H:%M:%S%z'
        #parse!
        load.index = load.index.map(lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt))
    idmax = load.sum(axis=1).idxmax()
    dwmax = idmax.weekday()
    dini = idmax - dt.timedelta(days=dwmax, hours=idmax.hour, minutes=idmax.minute)
    dend = dini + dt.timedelta(days=7+extradays) #- dt.timedelta(minutes=30)
    return load.loc[dini:dend,:]


def interpolate(data, step=15):
    """ Returns the data with a greater time resolution, by interpolating it
    """
    if type(data.index[0]) == str:
        fmtdt = '%Y-%m-%d %H:%M:%S%z'
        #parse!
        data.index = data.index.map(lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt))
    if (data.index[1]-data.index[0])/dt.timedelta(minutes=step) % 1>0:
        raise ValueError('Invalid step, it should be a divisor of data step')
    return data.asfreq(freq=dt.timedelta(minutes=step)).interpolate()

    
class Grid:
    def __init__(self, 
                 ndays=7, step=30, init_day=0, 
                 name='def', load=0, ss_pmax=20, verbose=True,
                 buses = []):
        """Instantiates the grid object:
            Creates vectors of ev load, conso load
            for time horizon ndays = default 7, one week
            with steps of 30 min
            ev_data is a dict of ev types where each entry has a dict with:
                type : 'tou/dumb'
                n_ev : # of evs
                other : (not needed), dict with other params
                **ev_global_params are general params passed to all types of evs
        """
        self.verbose = verbose
        if verbose:
            print('Instantiating Grid {}'.format(name))
        self.weekends = [5, 6]
        if 60 % step > 0:
            raise ValueError('Steps should be a divisor of an 60 minutes \
                             (ex. 30-15-5min), given value: %d' % step)
        
        # General params
        self.ndays = ndays
        self.periods = int(ndays * 24 * 60 / step)
        self.periods_day = int(24 * 60 / step)
        self.periods_hour = int(60/step)
        self.period_dur = step / 60         #in hours
        self.day = 0
        self.days = [(i + init_day)%7 for i in range(ndays + 1)]
        # times is an array that contains of len=nperiods, where for period i:
        # times[i] = [day, hour, #ofweekday]
        self.times = [[i, j, (i+init_day)%7] 
                        for i in range(ndays) 
                        for j in np.arange(0,24,self.period_dur)]
        
        #Grid params
        self.ss_pmax = ss_pmax              #in MW
        self.name = name
        
        # Init global vectors           
        self.init_load_vector(load)
        # TODO: load as dataframe, adjusting interpolation and days to given params
        self.buses = []
        
        # Empty arrays for EVs
        self.types_evs = []
        self.evs = {}
        print('Grid instantiated')
        
        
    def add_evs(self, nameset, n_evs, ev_type, **ev_params):
        """ Initiates EVs give by the dict ev_data and other ev global params
        """
        ev_types = {'dumb' : EV,
                    'mod': EV_Modulated,
                    'randstart': EV_RandStart,
                    'reverse': EV_DumbReverse}
        ev_fx = ev_types[ev_type]
       
        evset = []
        if self.verbose:
            print('Creating EV set {} containing {} {} EVs'.format(
                    nameset, n_evs, ev_type))
        for i in range(n_evs):
            evset.append(ev_fx(self, name=nameset+str(i), **ev_params))
        
        self.types_evs.append(nameset)
        self.evs[nameset] = evset
        self.init_ev_vectors(nameset)
                
    def init_load_vector(self, load):
        """ Creates empty array for global variables"""
        self.ev_load = {'Total': np.zeros(self.periods)}
        self.ev_potential = {'Total': np.zeros(self.periods)}
        self.ev_off_peak_potential = {'Total': np.zeros(self.periods)}
        self.ev_up_flex  = {'Total': np.zeros(self.periods)}
        self.ev_dn_flex = {'Total': np.zeros(self.periods)}
        self.ev_mean_flex = {'Total': np.zeros(self.periods)}
        self.ev_batt = {'Total': np.zeros(self.periods)}
        if type(load) == int:
            #TODO: load as DataFrame?
            #no base load given
            self.base_load = np.zeros(self.periods)
        else:
            self.base_load = load
   
    def init_ev_vectors(self, nameset):
        """ Creates empty array for global EV variables per set of EV"""
        self.ev_load[nameset] = np.zeros(self.periods)
        self.ev_potential[nameset] = np.zeros(self.periods)
        self.ev_off_peak_potential[nameset] = np.zeros(self.periods)
        self.ev_up_flex[nameset] = np.zeros(self.periods)
        self.ev_dn_flex[nameset] = np.zeros(self.periods)
        self.ev_mean_flex[nameset] = np.zeros(self.periods)
        self.ev_batt[nameset] = np.zeros(self.periods)
        
    
    def assign_ev_bus(self, evtype, buses, ev_per_bus):
        """ Asign a bus for a group of evs (evtype), in a random fashion,
        limited to a maximum number of evs per bus
        """
        available_buses = [buses[i] for i in range(len(buses)) for j in range(ev_per_bus[i])]
        np.random.shuffle(available_buses)
        ev = self.evs[evtype]
        if len(ev) > len(available_buses):
            strg = ('Not sufficient slots in buses for the number of EVs\n'+
                '# slots {}, # EVs {}'.format(len(available_buses), len(ev)))
            raise ValueError(strg)
        for i in range(len(ev)):
            ev[i].bus = available_buses[i] 
            
    def new_day(self):
        """ Iterates over evs to compute new day 
        """
        for types in self.evs:
            for ev in self.evs[types]:
                ev.new_day(self)
    
    def compute_per_bus_data(self):
        """ Computes aggregated ev load per bus and ev type
        """
        load_ev = {}
        print(load_ev)
        for types in self.evs:
            for ev in self.evs[types]:
                if (types, ev.bus) in load_ev:
                    load_ev[types, ev.bus] += ev.charging
                else:
                    load_ev[types, ev.bus] = ev.charging *1
        return load_ev
    
    def compute_agg_data(self) :    
        """ Computes aggregated charging per type of EV and then total for the grid 
        """
        if self.verbose:
            print('Grid {}: Computing aggregated data'.format(self.name))
        for types in self.evs:
            for ev in self.evs[types]:
                self.ev_potential[types] += ev.potential
                self.ev_load[types] += ev.charging
                self.ev_off_peak_potential[types] += ev.off_peak_potential
                self.ev_up_flex[types] += ev.up_flex
                self.ev_dn_flex[types] += ev.dn_flex
                self.ev_mean_flex[types] += ev.mean_flex_traj
                self.ev_batt[types] += ev.soc * ev.batt_size
        self.ev_potential['Total'] = sum([self.ev_potential[types] for types in self.evs])
        self.ev_load['Total'] = sum([self.ev_load[types] for types in self.evs])
        self.ev_off_peak_potential['Total'] = sum([self.ev_off_peak_potential[types] for types in self.evs])
        self.ev_up_flex['Total'] = sum([self.ev_up_flex[types] for types in self.evs])
        self.ev_dn_flex['Total'] = sum([self.ev_dn_flex[types] for types in self.evs])
        self.ev_mean_flex['Total'] = sum([self.ev_mean_flex[types] for types in self.evs])
        self.ev_batt['Total'] = sum([self.ev_batt[types] for types in self.evs])
        
    def do_days(self, agg_data=True):
        """Iterates over days to compute charging 
        """
        for d in range(self.ndays):
            if self.verbose:
                print('Grid {}: Computing day {}'.format(self.name, self.day))
            self.new_day()
            self.day += 1
        if agg_data:
            self.compute_agg_data()
        
    def plot_evload(self, **plot_params):
        """ Stacked plot of EV charging load
        """
        load = [self.ev_load[types] for types in self.types_evs]
        tot = 'Total'
        x = [t[0] * 24 + t[1] for t in self.times]
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        ax.stackplot(x, load, labels=self.types_evs)
        ax.plot(x, self.ev_potential[tot], label='EV potential')
        if not (self.ev_potential[tot] == self.ev_off_peak_potential[tot]).all():
            ax.plot(x, self.ev_off_peak_potential[tot], label='EV off-peak potential')
        ax.legend(loc=1)
        ax.set_ylabel('Power [kW]')
        ax.set_xlabel('Time [h]')
        if 'title' in plot_params:
            ax.set_title(plot_params['title'])
        if 'ylim' in plot_params:
            ax.set_ylim(top=plot_params['ylim'])
        ax.grid(axis='x')
        ax.set_xticks(np.arange(self.ndays) * 24)
        ax.set_xlim([0, self.ndays * 24])
   
     
    def plot_tot_load(self, **plot_params):
        """ Stacked plot of EV charging load + base load
        """
        tot = 'Total'
        x = [t[0] * 24 + t[1] for t in self.times]
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        ax.stackplot(x, [self.base_load, self.ev_load[tot]/1000], labels=['Base Load', 'EV Load'])
        ax.set_ylabel('Power [MW]')
        ax.set_xlabel('Time [h]')
        if self.ss_pmax > 0:
            ax.axhline(self.ss_pmax, label='Pmax', linestyle='--', color='red')
        if 'title' in plot_params:
            ax.set_title(plot_params['title'])
        if 'ylim' in plot_params:
            ax.set_ylim(top=plot_params['ylim'])
        ax.grid(axis='x')
        ax.set_xticks(np.arange(self.ndays) * 24)
        ax.set_xlim([0, self.ndays * 24])
        ax.legend(loc=1)
        
    def plot_flex_pot(self, trajectory=False, **plot_params):
        """ Plot of aggregated flex
        """
        tot = 'Total'
        x = [t[0] * 24 + t[1] for t in self.times]
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        ax.plot(x, self.ev_up_flex[tot], label='Upper storage limit')
        ax.plot(x, self.ev_dn_flex[tot], label='Lower storage limit')
        if trajectory:
            ax.plot(x, self.ev_batt[tot], label='Real trajectory')
        else:
            ax.plot(x, self.ev_mean_flex[tot], label='Mean flexible trajectory', linestyle='--')
        ax.set_ylabel('EV energy storage [kWh]')
        ax.set_xlabel('Time [h]')
        if 'title' in plot_params:
            ax.set_title(plot_params['title'])
        if 'ylim' in plot_params:
            ax.set_ylim(top=plot_params['ylim'])
        ax.set_ylim(bottom=0)
        ax.grid(axis='x')
        ax.set_xticks(np.arange(self.ndays) * 24)
        ax.set_xlim([0, self.ndays * 24])
        ax.legend(loc=1)
        
    def get_global_data(self):
        """ Some global info
        """
        total_ev_charge = self.ev_load['Total'].sum() * self.period_dur
        flex_pot = sum(ev.off_peak_potential.sum() * self.period_dur
                        for key in self.evs
                        for ev in self.evs[key])
        ev_flex_ratio = 1-total_ev_charge / flex_pot
        max_ev_load = self.ev_load['Total'].max()
        max_load = (self.ev_load['Total'] + self.base_load).max()
        max_base_load = self.base_load.max()
        peak_charge = max_load / self.ss_pmax
        h_overload = ((self.ev_load['Total'] + self.base_load) > self.ss_pmax).sum() * self.period_dur
        return {'Tot_ev_charge' : total_ev_charge,
                'Flex_ratio' : ev_flex_ratio,
                'Max_ev_load' : max_ev_load,
                'Max_base_load' : max_base_load,
                'Max_load' : max_load,
                'Peak_ss_charge' : peak_charge,
                'Hours_overload' : h_overload
                }
        
        
    def do_dist_hist(self, weekday = True, **plot_params):
        """ Do histogram of distances
        """
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        d = np.asarray([(ev.dist_wd if weekday else ev.dist_we) for types in self.evs
                                for ev in self.evs[types]])
        ax.hist(d, bins=np.arange(0,100,2))
        ax.set_xlim([0,100])
        ax.set_title('Histogram of trip distances')
        ax.set_xlabel('km')
        ax.set_ylabel('Frequency')
    
    
    def do_ncharging_hist(self, **plot_params):
        """ Do histogram of number of charging sessions
        """
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        d = np.asarray([ev.ch_status.sum() for types in self.evs
                                for ev in self.evs[types]])
        ax.hist(d, bins=np.arange(0, self.ndays + 2, 1))
        ax.set_xlim([0, self.ndays+1])
        if not 'title' in plot_params:
            ax.set_title('Histogram of charging sessions')
        else:
            ax.set_title(plot_params['title'])
        ax.set_xlabel('# charging sessions')
        ax.set_ylabel('Frequency')
    
    def get_ev(self):
        """ Returns random ev
        """
        return np.random.choice(self.evs[np.random.choice([key for key in self.evs])])
    
    def get_evs(self):
        """ Returns list of evs
        """
        return [ev for key in self.evs for ev in self.evs[key]]
    
    def export_ev_data(self, atts=''):
        """ returns a dict with ev data
        atts : attributes to export
        """
        if atts == '':
            atts = ['name', 'bus', 'dist_wd', 'dist_we']
        ev_data = {}
        for types in self.evs:
            ev_data[types] = []
            for ev in self.evs[types]:
                ev_data[types].append({att : getattr(ev, att) for att in atts})
        return ev_data
    
    def import_ev_data(self, ev_data):
        """ sets ev data
        ev_data is a dict
        {types0: [{ev0}, {ev1}, ...],
         types1: [{ev0}, {ev1}, ...]}
        and evi
        evi = {att, val}, with attribute and value
        """
        for types in ev_data:
            ev_set = self.evs[types]
            for i in range(len(ev_data[types])):
                ev = ev_data[types][i]
                for att in ev:
                    setattr(ev_set[i], att, ev[att])
    
    def evs_per_bus(self):
        """Returns the list of buses and the number of evs per bus 
        """
        busev = []
        for ev in self.get_evs():
            busev.append(ev.bus)
        busev = np.array(busev)
        buslist = np.unique(busev)
        evs_bus = []
        for b in buslist:
            evs_bus.append((busev == b).sum())
        return buslist, evs_bus
        
class EV:
    """ Basic EV model with dumb charging
    """
    bins_dist = np.linspace(0, 100, num=51)
    def __init__(self, model, name, 
                 cdf_dist_wd=0,
                 cdf_dist_we=0, 
                 charging_power=3.6, 
                 charging_eff = 0.95,
                 discharging_eff = 0.95,
                 charging_type='all_days',
                 tou_ini = 0,
                 tou_end = 0,
                 tou_we = False,
                 driving_eff=0.2, 
                 batt_size=40,
                 range_anx_factor = 1.5,
                 extra_trip_proba = 0,
                 arrival_departure_data_wd = {},
                 arrival_departure_data_we = {'mu_arr':16, 'mu_dep':8,
                                              'std_arr':2, 'std_dep':2},
                 bus=''):
        """Instantiates EV object:
           name id
           sets home-work distance [km]
           sets weekend distance [km]
           charging power [kW] = default 3.6kW
           efficiency [kWh/km] = default 0.2 kWh/km
           battery size [kWh] = default 40 kWh
           charging_type = 'all_days' // others
           
        """
        self.name = name
        # PARAMS
        # Sets distance for weekday and weekend one-way trips
        self.dist_wd = self.set_dist(cdf_dist_wd)
        self.dist_we = self.set_dist(cdf_dist_we)
        # Discrete random distribution (non correlated) for battery & charging power
        if type(charging_power) is int or type(charging_power) is float:
            self.charging_power = charging_power
        elif len(charging_power) == 2:
            self.set_discrete_random_data(charging_power[0], charging_power[1])
        else:
            ValueError('Invalid charging_power value')     
        if type(batt_size) is int or type(batt_size) is float:
            self.batt_size = batt_size
        elif len(batt_size) == 2:
            self.set_discrete_random_data(batt_size[0], batt_size[1])
        else:
            ValueError('Invalid charging_eff value')               
        self.charging_eff = charging_eff                # Charging efficiency, in pu
        self.discharging_eff = discharging_eff          # Discharging efficiency, in pu
        self.driving_eff = driving_eff                  # Driving efficiency kWh / km
        self.min_soc = 0.2                              # Min SOC of battery
        self.n_trips = 2                                # Number of trips per day (Go and come back)
        self.extra_trip_proba = extra_trip_proba        # probability of extra trip
        if not charging_type in ['if_needed', 'if_needed_sunday', 'all_days']:
            ValueError('Invalid charging type %s' %charging_type)
        self.charging_type = charging_type              # Charging behavior (every day or not)
        self.range_anx_factor = range_anx_factor        # Range anxiety factor for "if needed" charging
        self.tou_ini = tou_ini                          # Time of Use (low tariff) start time (default=0) 
        self.tou_end = tou_end                          # Time of Use (low tariff) end time (default=0)
        self.tou_we = tou_we                            # Time of Use for weekend
        self.arrival_departure_data_wd = arrival_departure_data_wd
        self.arrival_departure_data_we = arrival_departure_data_we
        self.eff_per_period = model.period_dur * self.charging_eff 
        self.soc_eff_per_period = self.eff_per_period / self.batt_size
        self.soc_v2geff_per_period = model.period_dur / self.batt_size / self.discharging_eff
        
        # Grid Params
        self.bus = ''

        
        # RESULTS/VARIABLES
        self.soc_ini = np.zeros(model.ndays)        #list of SOC ini at each day (of charging session)
        self.soc_end = np.zeros(model.ndays)        #list of SOC end at each day (of charging session)
        self.energy_trip = np.zeros(model.ndays)    #list of energy consumed per day in trips
        self.charged_energy = np.zeros(model.ndays) # charged energy into the battery
        self.extra_charge = np.zeros(model.ndays)   # extra charge needed during the day (bcs too long trips, not enough batt!)
        self.ch_status = np.zeros(model.ndays)      # Charging status for each day (connected or not connected)

        self.set_ch_vector(model)
        self.set_off_peak(model)
        
    def set_dist(self, cdf_dist):
        """Returns one-way distance given by a cumulative distribution function
        The distance is limited to 120km
        """
        if type(cdf_dist) == int:
            #means no pdf has been given, using default values lognormal distribution
            # Based on O.Borne thesis (ch.3), avg trip +-19km
            d = np.random.lognormal(2.75, 0.736, 1)
            #check that distance is under dmax = 120km, so it can be done with one charge
            while d > 120:
                d = np.random.lognormal(2.75, 0.736, 1)
            return d
        else:  
            return random_from_pdf(cdf_dist, bins_dist)
        
    def set_discrete_random_data(self, data_values, values_prob):
        """ Returns a random value from a set data_values, according to the probability vector values_prob
        """
        np.random.choice(data_values, p=values_prob)
        
    def set_arrival_departure(self, pdf_arr = 0, pdf_dep = 0, 
                              mu_arr = 18, mu_dep = 8, 
                              std_arr = 2, std_dep = 2):
        """ Sets arrival and departure times
        """
        if type(pdf_arr) == int:
            self.arrival = np.random.randn(1) * std_arr + mu_arr

        else:
            self.arrival = random_from_pdf(pdf_arr, bins_hours)
        if type(pdf_dep) == int:
            self.departure = np.random.randn(1) * std_dep + mu_dep
        else:
            self.departure = random_from_pdf(pdf_dep, bins_hours)
        self.dt = (self.departure - self.arrival if self.departure > self.arrival
                       else self.departure + 24 - self.arrival)
        
    def set_ch_vector(self, model):
        # Grid view
        self.charging = np.zeros(model.periods) # Charging power at time t 
        self.off_peak_potential = np.zeros(model.periods) # Connected and chargeable power (only off-peak period)
        self.potential = np.zeros(model.periods) # Connected power at time t
        self.up_flex = np.zeros(model.periods) # Battery flex capacity, upper bound
        self.dn_flex = np.zeros(model.periods) # Battery flex capacity, lower bound (assumes bidir charger)
        self.mean_flex_traj = np.zeros(model.periods) # Mean trajectory to be used to comupte up & dn flex
        self.soc = np.zeros(model.periods) # SOC at time t
   
    def set_off_peak(self, model):
        """ Sets vector for off-peak period (EV will charge only during this period)
        """
        # TODO: expand to different off-peak hours during the weekend
        self.off_peak = np.ones(model.periods)
        if self.tou_ini < self.tou_end:
            for i in range(model.periods):
                if not (self.tou_we and model.times[i][2] in model.weekends):
                    # This checks that there is no special ToU in weekends, and that it is not the weekend
                    if model.times[i][1] < self.tou_ini or model.times[i][1] >= self.tou_end:
                        self.off_peak[i] = 0
        elif self.tou_ini > self.tou_end:
            for i in range(model.periods):
                if not (self.tou_we and model.times[i][2] in model.weekends):
                    if self.tou_end <= model.times[i][1] < self.tou_ini:
                        self.off_peak[i] = 0
                    
    def compute_energy_trip(self, model):
        """ Computes the energy used during the current day trips and to be charged
        in the current session.
        """
        # TODO: extend to add stochasticity
        if (self.dist_wd if model.days[model.day] < 5 else self.dist_we) * self.n_trips * self.driving_eff > self.batt_size:
            #This means that home-work trip is too long to do it without extra charge, 
            # so forced work charging (i.e one trip)
            self.energy_trip[model.day] = (self.dist_wd if model.days[model.day] < 5 else self.dist_we) * self.driving_eff
        else:
            extra_trip = 0 
            if np.random.rand(1) < self.extra_trip_proba:
                # Extra trip probability, normal distribution around 5km +- 1.5
                # TODO: better way to add extra trip
                extra_trip = np.random.randn(1) * 1.5 + 5
            self.energy_trip[model.day] = ((self.dist_wd if model.days[model.day] < 5 else self.dist_we) 
                                            * self.n_trips + extra_trip) * self.driving_eff 
            
    
    def compute_soc_ini(self, model):
        """ Computes soc at the start of the session based on 
        the energy used during current day
        """
        
        if model.day == 0:
            self.soc_ini[model.day] = 1 - self.energy_trip[model.day] / self.batt_size
        else:
            self.soc_ini[model.day] = self.soc_end[model.day-1] - self.energy_trip[model.day] / self.batt_size
        if self.soc_ini[model.day] < 0.05:                          # To correct some negative SOCs
            self.extra_charge[model.day] = 0.05 - self.soc_ini[model.day]
            self.soc_ini[model.day] = 0.05
            
    def define_charging(self, model):
        """ Defines charging status for the session. 
        True means it will charge this session
        """
        # TODO: How to compute next_trip?
        next_trip_energy = ((self.dist_wd if model.days[model.day + 1] < 5 
                                            else self.dist_we) * 
                            self.n_trips * self.driving_eff)
        # TODO: Other types of charging ?
        if self.charging_type == 'all_days':
            return True
        if self.charging_type == 'weekdays':
            if not model.days[model.day] in model.weekends:
                return True
            return False
#        if self.charging_type == 'weekends':
#            # TODO: Complete if charging is needed
#            if model.days[model.day] in model.weekends:
#                return True
#            return False
        if self.charging_type in ['if_needed', 'if_needed_sunday']:
        # Enough kWh in batt to do next trip?
            if self.charging_type == 'if_needed' and model.days[model.day] in model.weekends:
                #Force charging on weekend
                return True
            if self.charging_type == 'if_needed_sunday' and model.days[model.day] == 6:
                #Force charging only on sundays
                return True
            if (self.soc_ini[model.day] * self.batt_size < next_trip_energy * self.range_anx_factor 
                    or self.soc_ini[model.day] < self.min_soc):
                # Charging because it is needed for expected next day
                return True
            
            return False
    

    def do_charging(self, model):
        """ Computes charging potential and calls charging function
        """
        delta = model.day * model.periods_day
        tini = int(self.arrival * model.periods_hour)
        # Computes index for charging session
        if self.departure < self.arrival:
            tend = int((self.departure + 24) * model.periods_hour)    
        else:
            tend = int(self.departure *  model.periods_hour)
        idx_tini = delta + tini
        idx_tend = min([delta + tend, model.periods-1])
        
        if idx_tini >= idx_tend:
            return
        # Potential charging vector
        potential = np.ones(idx_tend+1-idx_tini) * self.charging_power
        # Correct for arrival period 
        potential[0] = ((model.period_dur - self.arrival % model.period_dur ) / 
                 model.period_dur * self.charging_power)
        # And correct for departure period
        potential[-1] = (self.departure % model.period_dur / 
                 model.period_dur * self.charging_power)
        
        self.potential[idx_tini:idx_tend+1] = potential
        self.off_peak_potential[idx_tini:idx_tend+1] = (
                potential * self.off_peak[idx_tini:idx_tend+1])
        
        # calls functions that update things
        self.compute_up_dn_flex(model, idx_tini, idx_tend)
        self.compute_charge(model, idx_tini, idx_tend)
        self.compute_soc_end(model, idx_tend)
    
    def compute_charge(self, model, idx_tini, idx_tend):
        """ do default charge: dumb
        This function should be extended by classes to do smart charging algos
        """
        # off peak potential in "per unit" of charging power 
        opp = self.off_peak_potential[idx_tini:idx_tend+1]
        
        # SOC is computed as the cumulative sum of charged energy  (Potential[pu] * Ch_Power*Efficiency*dt / batt_size) 
        soc = (opp.cumsum() * self.soc_eff_per_period + self.soc_ini[model.day]).clip(0,1)
        # charging power
        power = (soc - np.concatenate([[self.soc_ini[model.day]], soc[:-1]])) / self.soc_eff_per_period 
        
        # chssarged_energy
        self.charged_energy[model.day] = (soc[-1]-soc[0]) * self.batt_size
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power
    
    
    def compute_up_dn_flex(self, model, idx_tini, idx_tend):
        """ Computes up and down flex in terms of battery storage capacity [kWh]
        """
        # considers flex only with SOC end the highest possible, during all the connected period
        # Potential kw
        pu_pot = np.ones(idx_tend+1-idx_tini) 
        pu_pot[0] = 1 - (self.arrival % model.period_dur) / model.period_dur
        pu_pot[-1] = (self.departure % model.period_dur) / model.period_dur
        potential = pu_pot * self.charging_power
        #Soc up and down
        m_soc = min(self.soc_ini[model.day], self.min_soc)
        
        soc_up = (potential.cumsum() * self.soc_eff_per_period  + self.soc_ini[model.day]).clip(0,1)
        soc_dn = (-potential.cumsum() * self.soc_v2geff_per_period + self.soc_ini[model.day]).clip(m_soc,1)
        
        soc_end = soc_up[-1]
        downwards_soc = np.concatenate([[soc_end], soc_end - potential[:-1][::-1].cumsum() * self.soc_eff_per_period])[::-1]
        soc_dn = np.maximum(soc_dn, downwards_soc)
        
        avg_ch_pu = (soc_end - self.soc_ini[model.day]) / (self.dt / model.period_dur)
        
        self.up_flex[idx_tini:idx_tend+1] = soc_up * self.batt_size
        self.dn_flex[idx_tini:idx_tend+1] = soc_dn * self.batt_size
        self.mean_flex_traj[idx_tini:idx_tend+1] = (self.soc_ini[model.day] + pu_pot.cumsum() * avg_ch_pu) * self.batt_size         
            
    def compute_soc_end(self, model, idx_tend=False):
        """ Calculates SOC at the end of the charging session
        """
        if self.ch_status[model.day]:
            self.soc_end[model.day] = self.soc[idx_tend]
        else:
            self.soc_end[model.day] = self.soc_ini[model.day]
            
    def new_day(self, model):
        # Update for a new day:
        # Compute arrivals and departures
        if model.days[model.day] in model.weekends:
                self.set_arrival_departure(**self.arrival_departure_data_we)                    
        else:
            self.set_arrival_departure(**self.arrival_departure_data_wd)
        # Computes initial soc based on past trips
        self.compute_energy_trip(model)
        self.compute_soc_ini(model)
        # Defines if charging is needed
        self.ch_status[model.day] = self.define_charging(model)
        if self.ch_status[model.day]:
            self.do_charging(model)
        else:
            self.compute_soc_end(model)


class EV_Modulated(EV):
    """ Class of EV that does a flat charge over the whole period of connection (considering ToU constraints)
    """
    def __init__(self, model, name, pmin_charger=0.6, **params):
        super().__init__(model, name, **params)
        self.pmin_charger = pmin_charger                    # in pu of max charging power
        
    def compute_charge(self, model, idx_tini, idx_tend):
        """ compute Modulated charge
        """

        # off peak potential in "per unit" of time step
        opp = self.off_peak_potential[idx_tini:idx_tend+1] / self.charging_power  
        needed_soc = 1 - self.soc_ini[model.day]
        available_dt = opp.sum()                                        # in time_steps
        if available_dt == 0:
            return                                                    
        avg_ch_soc = needed_soc / available_dt                          # charged soc per unit of timestep
        max_ch_soc = self.charging_power * self.soc_eff_per_period      # Avg charging cannot be more than charging power
        min_ch_soc = max_ch_soc * self.pmin_charger                     # avg charging cannot be less than min charging power
        
        avg_ch_soc = min(max_ch_soc, max(min_ch_soc, avg_ch_soc))       #
        
        soc = (opp.cumsum() * avg_ch_soc + self.soc_ini[model.day]).clip(0,1)
        # charging power
        power = (soc - np.concatenate([[self.soc_ini[model.day]], soc[:-1]])) / self.soc_eff_per_period 
        
        # charged_energy
        self.charged_energy[model.day] = (soc[-1]-soc[0]) * self.batt_size
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power
        
class EV_RandStart(EV):
    """ Class of EV that charges "dumb", but starting at a random hour within 
    the time-charge constraints
    Ex: An EV needs to charge 22kWh with a 11kW charger, 
    arrives at 8:00am, leaves at 13:00.
    Dumb (as soon as plugged): Charge at 11kW between 8-10h
    DumbReverse : Charge at 11kW between 11-13h (departure time)
    Modulated : Charge at 4,4kW between 8-13h (22kWh/5h)
    RandStart : Charges for 2h at 11kW starting at random between 8-11h
    """
    def __init__(self, model, name, **params):
        super().__init__(model, name, **params)
        
    def compute_charge(self, model, idx_tini, idx_tend):
        """ compute random start charge
        """
        opp = self.off_peak_potential[idx_tini:idx_tend+1]
        needed_soc = 1 - self.soc_ini[model.day]
        needed_time = needed_soc / (
                self.soc_eff_per_period * self.charging_power)      # Number of periods that need charging                                                                    
        session_time = opp.sum() / self.charging_power              # Number of periods for charging
        
        if session_time > needed_time:                              # Number of periods of delay for charging
            randstart = np.random.random() * (session_time - needed_time)
        else:
            randstart = 0
        # SOC is computed as the cumulative sum of charged energy  (Potential[pu] * Ch_Power*Efficiency*dt / batt_size) 
        # but with a delay of randstart number of periods
        soc = ((opp.cumsum() - randstart * self.charging_power).clip(min=0) * 
               self.soc_eff_per_period + self.soc_ini[model.day]).clip(0,1)
        # charging power
        power = (soc - np.concatenate([[self.soc_ini[model.day]], soc[:-1]])) / self.soc_eff_per_period 
        
        # chssarged_energy
        self.charged_energy[model.day] = (soc[-1]-soc[0]) * self.batt_size
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power
        
class EV_DumbReverse(EV):
    """ Class of EV that charges "dumb" to be ready at departure hour
    Ex: An EV needs to charge 22kWh with a 11kW charger, 
    arrives at 8:00am, leaves at 13:00.
    Dumb (as soon as plugged): Charge at 11kW between 8-10h
    DumbReverse : Charge at 11kW between 11-13h (departure time)
    Modulated : Charge at 4,4kW between 8-13h (22kWh/5h)
    RandStart : Charges for 2h at 11kW starting at random between 8-11h
    
    """
    def __init__(self, model, name, **params):
        super().__init__(model, name, **params)
        
    def compute_charge(self, model, idx_tini, idx_tend):
        """ compute Reverse dumb charge
        """

        # off peak potential in "per unit" of time step
        opp = self.off_peak_potential[idx_tini:idx_tend+1]  
        
        
        # reverse charging
        rev_ch = (opp[::-1].cumsum()[::-1] * self.soc_eff_per_period).clip(
                            max=1-self.soc_ini[model.day])
        soc = self.soc_ini[model.day] + (rev_ch.max() - rev_ch)
        # charging power
        power = (soc - np.concatenate([[self.soc_ini[model.day]], soc[:-1]])) / self.soc_eff_per_period 
        
        # charged_energy
        self.charged_energy[model.day] = (soc[-1]-soc[0]) * self.batt_size
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power
        