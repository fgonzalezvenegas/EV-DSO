20# -*- coding: utf-8 -*-
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
import util
import scipy.stats as stats
import cvxopt
import time


cvxopt.solvers.options['show_progress'] = False
bins_dist = np.linspace(0, 100, num=51)
dist_function = np.sin(bins_dist[:-1]/ 100 * np.pi * 2) + 1
dist_function[10:15] = [0, 0, 0 , 0 , 0]
pdfunc = (dist_function/sum(dist_function)).cumsum()

bins_hours = np.linspace(0,24,num=25)

def random_from_cdf(cdf, bins):
    """Returns a random bin value given a cdf.
    cdf has n values, and bins n+1, delimiting initial and final boundaries for each bin
    """
    if cdf.max() > 1.0001 or cdf.min() < 0:
        raise ValueError('CDF is not a valid cumulative distribution function')
    r = np.random.rand(1)
    x = int(np.digitize(r, cdf))
    return bins[x] + np.random.rand(1) * (bins[x+1] - bins[x])

def random_from_2d_pdf(pdf, bins):
    """ Returns a 2-d random value from a joint PDF.
    It assumes a square np.matrix as cdf. Sum of all values in the cdf is 1.
    """
    val1 = random_from_cdf(pdf.sum(axis=1).cumsum(), bins)
    x = np.digitize(val1, bins)
    val2 = random_from_cdf(pdf[x-1].cumsum() / pdf[x-1].sum(), bins)
    return val1, val2

def discrete_random_data(self, data_values, values_prob):
    """ Returns a random value from a set data_values, according to the probability vector values_prob
    """
    return np.random.choice(data_values, p=values_prob)

def set_dist(data_dist):
        """Returns one-way distance given by a cumulative distribution function
        The distance is limited to 120km
        """
        # Default values for 
        # Based on O.Borne thesis (ch.3), avg trip +-19km
        s=0.736
        scale=np.exp(2.75)
        loc=0
        if type(data_dist) in [int, float]:
            return data_dist
        if type(data_dist) == dict:
            if 's' in data_dist:
                # Data as scipy.stats.lognorm params
                s = data_dist['s']
                loc = data_dist['loc']
                scale = data_dist['scale']
            if 'cdf' in data_dist:
                # data as a cdf, containts cdf and bins values
                cdf = data_dist['cdf']
                if 'bins' in data_dist:
                    bins_dist = data_dist['bins']
                else:
                    bins_dist = np.linspace(0, 100, num=51)
                return random_from_cdf(cdf, bins_dist)
            
        d = stats.lognorm.rvs(s, loc, scale, 1)
        #check that distance is under dmax = 120km, so it can be done with one charge
        while d > 120:
            d = stats.lognorm.rvs(s, loc, scale, 1)
        return d        

#def load_conso_ss_data(folder = 'c:/user/U546416/Documents/PhD/Data/Mobilité/',
#                       folder_load = 'Data_Traitee/Conso/',
#                       folder_grid = 'Data_Traitee/Reseau/',
#                       file_load_comm = 'consommation-electrique-par-secteur-dactivite-commune-red.csv',
#                       file_load_profile = 'conso_all_pu.csv',
#                       file_ss = 'postes_source.csv'):
#    """ load data for conso and substations
#    """ 
#    # Load load by commune data
#    load_by_comm = pd.read_csv(folder + folder_load + file_load_comm, 
#                               engine='python', delimiter=';', index_col=0)
#    load_by_comm.index = load_by_comm.index.astype(str)
#    load_by_comm.index = load_by_comm.index.map(lambda x: x if len(x) == 5 else '0' + x)
#    # Load load profiles data (in pu (power, not energy))
#    load_profiles = pd.read_csv(folder + folder_load + file_load_profile, 
#                               engine='python', delimiter=',', index_col=0)
#    # drop ENT profile that's not useful
#    load_profiles = load_profiles.drop('ENT', axis=1)
#    # Load Trafo data
#    SS = pd.read_csv(folder + folder_grid + file_ss, 
#                               engine='python', delimiter=',', index_col=0)
#    return load_by_comm, load_profiles, SS


#def load_hist_data(folder=r'c:/user/U546416/Documents/PhD/Data/Mobilité/', 
#                   folder_hist='',
#                   file_hist_home='HistHome.csv',
#                   file_hist_work='HistWork.csv'):
#    """ load histogram data for conso and substations
#    """
#    
#    hist_home = pd.read_csv(folder + folder_hist+ file_hist_home, 
#                               engine='python', delimiter=',', index_col=0)
#    hist_work = pd.read_csv(folder + folder_hist+ file_hist_work, 
#                               engine='python', delimiter=',', index_col=0)
#    return hist_home, hist_work


#def extract_hist(hist, comms):
#    """ returns histograms for communes
#    """
#    labels = ['UU', 'ZE', 'Status', 'Dep']
#    cols_h = hist.columns.drop(labels)
#    return pd.DataFrame({c: np.asarray(hist.loc[c,cols_h]) for c in comms}).transpose()
    

#def compute_load_from_ss(load_by_comm, load_profiles, SS, ss):
#    """Returns the load profile for the substation ss, 
#    where Substation data is stored in SS DataFrame (namely communes assigned) 
#    and load data in load_profiles and load_by_comm
#    """
#    if not ss in SS.Communes:
#        raise ValueError('Invalid Substation %s' %ss)
#    comms = SS.Communes[ss]
#    try: 
#        factors = load_by_comm.loc[comms, load_profiles.columns].sum() / (8760)
#    except:
#        factors = pd.DataFrame({key: 0 for key in load_profiles.columns})
#    return load_profiles * factors


#def get_max_load_week(load, step=30, buffer_before=0, buffer_after=0):
#    """ Returns the week of max load. It adds Xi buffer days before and after
#    """
#    if type(load.index[0]) == str:
#        fmtdt = '%Y-%m-%d %H:%M:%S%z'
#        #parse!
#        load.index = load.index.map(lambda x: dt.datetime.strptime(''.join(x.rsplit(':',1)), fmtdt))
#    idmax = load.idxmax()
#    dwmax = idmax.weekday()
#    dini = idmax - dt.timedelta(days=dwmax+buffer_before, hours=idmax.hour, minutes=idmax.minute)
#    dend = dini + dt.timedelta(days=7+buffer_after+buffer_before) #- dt.timedelta(minutes=30)
#    return load.loc[dini:dend]


    
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
        self.step = step                    # in minutes
        self.ndays = ndays
        self.periods = int(ndays * 24 * 60 / step)
        self.periods_day = int(24 * 60 / step)
        self.periods_hour = int(60/step)
        self.period_dur = step / 60         #in hours
        self.day = 0
        self.days = [(i + init_day)%7 for i in range(ndays + 1)]
        # times is an array that contains of len=nperiods, where for period i:
        # times[i] = [day, hour, day_of_week] ** day_of_week 0 == monday
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
        self.ev_sets = []
        self.ev_grid_sets = []
        self.evs_sets = {}
        self.evs = {}
        print('Grid instantiated')

    def add_aggregator(self, nameagg, **agg_data):
        """ Initiates an Aggregator
        """
        if not hasattr(self, 'aggregators'):
            self.aggregators = []
            self.ev_agg_sets = []
        agg = Aggregator(self, nameagg, **agg_data)
        if self.verbose:
            print('Creating new aggregator')
        self.aggregators.append(agg)
        return agg
        
    def add_evs(self, nameset, n_evs, ev_type, aggregator=None, charge_schedule=None, **ev_params):
        """ Initiates EVs give by the dict ev_data and other ev global params
        """
        ev_types = dict(dumb = EV,
                        mod = EV_Modulated,
                        randstart = EV_RandStart,
                        reverse = EV_DumbReverse,
                        pfc = EV_pfc,
                        optch = EV_optimcharge)
        
        if not (ev_type in ev_types):
            raise ValueError('Invalid EV type "{}" \
                             Accepted types are: {}'.format(ev_type, [i for i in ev_types.keys()]))
        ev_fx = ev_types[ev_type]
        # Check that the nameset doesnt exists
        if nameset in self.ev_sets:
            raise ValueError('EV Nameset "{}" already in the grid. Not created.'.format(nameset))            
        evset = []

        # Create evs
        # TODO: improve this
        # Check if schedule is given:
        if not (charge_schedule is None):
            # If schedule has 'User' column, create each EV with its own schedule
            if 'User' in charge_schedule:
                users = charge_schedule.User.unique()
                n_evs = len(users)
                for i in users:
                    evset.append(ev_fx(self, name=str(i), boss=aggregator, 
                                       charge_schedule=charge_schedule[charge_schedule.User==i].reset_index(drop=True),
                                       **ev_params))
            # else, all EVs with same schedule
            else:
                for i in range(n_evs): 
                    evset.append(ev_fx(self, name=nameset+str(i), boss=aggregator, 
                                       charge_schedule=charge_schedule, **ev_params))
        else:   
            for i in range(n_evs):
                evset.append(ev_fx(self, name=nameset+str(i), boss=aggregator, **ev_params))
        if self.verbose:
            print('Created EV set {} containing {} {} EVs'.format(
                    nameset, n_evs, ev_type))
        # Save in local variables
        self.ev_sets.append(nameset)
        # Check if the evs are assigned to an aggregator
        if aggregator == None:
            self.ev_grid_sets.append(nameset)
        else:
            if not (aggregator in self.aggregators):
                raise ValueError('Invalid aggregator')
            self.ev_agg_sets.append(nameset)
            aggregator.evs += evset
            aggregator.nevs += n_evs
        self.evs_sets[nameset] = evset
        for ev in evset:
            self.evs[ev.name] = ev
        self.init_ev_vectors(nameset)
        return self.evs_sets[nameset]
                
    def init_load_vector(self, load):
        """ Creates empty array for global variables"""
        self.ev_load = dict(Total = np.zeros(self.periods))
        self.ev_potential = dict(Total = np.zeros(self.periods))
        self.ev_off_peak_potential = dict(Total = np.zeros(self.periods))
        self.ev_up_flex  = dict(Total = np.zeros(self.periods))
        self.ev_dn_flex = dict(Total = np.zeros(self.periods))
        self.ev_mean_flex = dict(Total = np.zeros(self.periods))
        self.ev_batt = dict(Total = np.zeros(self.periods))
        if type(load) == int:
            #TODO: load as DataFrame?
            #no base load given
            self.base_load = np.zeros(self.periods)
        elif len(load) != self.periods:
            #TODO: automatic correction
            raise ValueError('Base load does not match periods')
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
        ev = self.evs_sets[evtype]
        if len(ev) > len(available_buses):
            strg = ('Not sufficient slots in buses for the number of EVs\n'+
                '# slots {}, # EVs {}'.format(len(available_buses), len(ev)))
            raise ValueError(strg)
        for i in range(len(ev)):
            ev[i].bus = available_buses[i]
    
    def add_freq_data(self, freq, step_f=10, max_dev=0.2, type_f='dev', base_f=50):
        """ Computes scaling factor for frequency response (mu).
        Input data is frequency array, either on absolute Hz or in deviation from base frequency (50Hz as default)
        step_f is the time step of the frequency array (in seconds)
        saves mu array, which is the average frequency deviation for each step
        """
        if not type_f=='dev':
            freq = freq - base_f
        # number of frequency measures per simulation period
        nsteps_period = int(self.period_dur * 3600 / step_f) 
        
        # if i dont have enough freq data, I repeat ntimes the data
        ntimes = (nsteps_period * self.periods) / len(freq)
        if ntimes > 1:
            print('Insuficient data, replicating it')
            freq = np.tile(freq, int(np.ceil(ntimes)))
        
        mu = np.zeros(self.periods)
        mu_up = np.zeros(self.periods)
        mu_dn = np.zeros(self.periods)
        dt_up = np.zeros(self.periods)
        dt_dn = np.zeros(self.periods)
        
        freq = (freq / max_dev).clip(-1,1)
        for i in range(self.periods):
            mu[i]  = -freq[i*nsteps_period:(i+1)*nsteps_period].mean()
            mu_up[i] = -(freq[i*nsteps_period:(i+1)*nsteps_period][freq[i*nsteps_period:(i+1)*nsteps_period]<=0]).mean()
            mu_dn[i] = -(freq[i*nsteps_period:(i+1)*nsteps_period][freq[i*nsteps_period:(i+1)*nsteps_period]>0]).mean() 
            dt_up[i] = (freq[i*nsteps_period:(i+1)*nsteps_period]<=0).mean()
            dt_dn[i] = (freq[i*nsteps_period:(i+1)*nsteps_period]>0).mean()
            
        self.mu = mu
        self.mu_up = np.nan_to_num(mu_up)
        self.mu_dn = np.nan_to_num(mu_dn)
        self.dt_up = dt_up
        self.dt_dn = dt_dn
        
    def add_prices(self, prices, step_p=1):
        """ Adds price vector.
        Input: Price vector (can be one day, week or the whole duration of the sim)
        step_p: Length of each price vector step, in hours
        Converts input price vector in step_p hours to 
        output price vector in grid step
        Prices in c€/kWh
        """
        # number of simulation steps per input price step
        nps = int(step_p/self.period_dur)        
        
        # if i dont have enough price data, I repeat n_times the data
        ntimes = (self.periods) / (len(prices) * nps)
        if ntimes > 1:
            print('Insuficient data, replicating it')
            prices = np.tile(prices, int(np.ceil(ntimes)))
        self.prices = np.repeat(prices, nps)[:self.periods]
        
    def add_evparam_from_dataframe(self, param, df):
        """ Add params to EVs from pd.DataFrame.
        df.columns are ev.names
        """
        for c in df:
            if c in self.evs:
                setattr(self.evs[c], param, df[c].values)
                
    def add_evparam_from_dict(self, param, dic):
        """ Add params to EVs from dict.
        dict.keys are ev.names
        
        """
        for c in dic:
            if c in self.evs:
                setattr(self.evs[c], param, dic[c])
    
    def new_day(self):
        """ Iterates over evs to compute new day 
        """
        for types in self.ev_grid_sets:
            for ev in self.evs_sets[types]:
                ev.new_day(self)
        if hasattr(self, 'aggregators'):
            for agg in self.aggregators:
                agg.new_day()
    
    def compute_per_bus_data(self):
        """ Computes aggregated ev load per bus and ev type
        """
        load_ev = {}
        for types in self.ev_sets:
            for ev in self.evs_sets[types]:
                if (types, ev.bus) in load_ev:
                    load_ev[types, ev.bus] += ev.charging * 1
                else:
                    load_ev[types, ev.bus] = ev.charging * 1
        return load_ev
    
    def compute_agg_data(self) :    
        """ Computes aggregated charging per type of EV and then total for the grid in MW
        """
        total = 'Total'
        if self.verbose:
            print('Grid {}: Computing aggregated data'.format(self.name))
        for types in self.ev_sets:
            for ev in self.evs_sets[types]:
                self.ev_potential[types] += ev.potential / util.k
                self.ev_load[types] += ev.charging / util.k
                self.ev_off_peak_potential[types] += ev.off_peak_potential / util.k
                self.ev_up_flex[types] += ev.up_flex / util.k
                self.ev_dn_flex[types] += ev.dn_flex / util.k
                self.ev_mean_flex[types] += ev.mean_flex_traj / util.k
                self.ev_batt[types] += ev.soc * ev.batt_size / util.k
        self.ev_potential[total] = sum([self.ev_potential[types] for types in self.evs_sets])
        self.ev_load[total] = sum([self.ev_load[types] for types in self.evs_sets])
        self.ev_off_peak_potential[total] = sum([self.ev_off_peak_potential[types] for types in self.evs_sets])
        self.ev_up_flex[total] = sum([self.ev_up_flex[types] for types in self.evs_sets])
        self.ev_dn_flex[total] = sum([self.ev_dn_flex[types] for types in self.evs_sets])
        self.ev_mean_flex[total] = sum([self.ev_mean_flex[types] for types in self.evs_sets])
        self.ev_batt[total] = sum([self.ev_batt[types] for types in self.evs_sets])
        
    def do_days(self, agg_data=True):
        """Iterates over days to compute charging 
        """
        if self.verbose:
            t = time.time()
            print('Starting simulation, Grid {}'.format(self.name))
            k = -1
        for d in range(self.ndays):
            if self.verbose:
                if (d*20)// self.ndays > k:
                    k = (d*20)// self.ndays
                    print('\tComputing day {}'.format(self.day))
            self.new_day()
            self.day += 1
        if agg_data:
            self.compute_agg_data()
        if self.verbose:
            print('Finished simulation, Grid {}\nElapsed time {}h {:02d}:{:04.01f}'.format(self.name, *util.sec_to_time(time.time()-t)))
        
    def set_aspect_plot(self, ax, day_ini=0, days=-1, **plot_params):
        """ Set the aspect of the plot to fit in the specified timeframe and adds Week names as ticks
        """
        x = [t[0] * 24 + t[1] for t in self.times]
        
        if days == -1:
            days = self.ndays
        days = min(self.ndays - day_ini, days)
        t0 = self.periods_day * day_ini
        tf = self.periods_day * (days + day_ini)
        
        daylbl = [util.dsnms[self.times[i][2]] for i in np.arange(t0, tf, self.periods_day)]
        
        ax.set_xlabel('Time [days]')
        if 'title' in plot_params:
            ax.set_title(plot_params['title'])
        else:
            ax.set_title('Load at {}'.format(self.name))
        if 'ylim' in plot_params:
            ax.set_ylim(top=plot_params['ylim'])
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(self.ndays+1) * 24)
        ax.set_xticklabels(np.tile(daylbl, int(np.ceil((self.ndays+1)/7))))
        ax.grid(axis='x')
        ax.set_xlim(x[t0], x[tf-1])
        
        ax.legend(loc=1)
        
    def plot_ev_load(self, day_ini=0, days=-1, opp=False, **plot_params):
        """ Stacked plot of EV charging load
        """
        load = np.array([self.ev_load[types] for types in self.ev_sets])
        tot = 'Total'
        x = [t[0] * 24 + t[1] for t in self.times]
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
            del plot_params['ax']
        ax.stackplot(x, load, labels=self.ev_sets)
        if opp:
            ax.plot(x, self.ev_potential[tot], label='EV potential')
            if not (self.ev_potential[tot] == self.ev_off_peak_potential[tot]).all():
                ax.plot(x, self.ev_off_peak_potential[tot], label='EV off-peak potential')
        ax.set_ylabel('Power [MW]')

        self.set_aspect_plot(ax, day_ini=day_ini, days=days, **plot_params)
        return ax
     
    def plot_total_load(self, day_ini=0, days=-1, **plot_params):
        """ Stacked plot of EV charging load + base load
        """
        tot = 'Total'
        x = [t[0] * 24 + t[1] for t in self.times]
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
            del plot_params['ax']
        ax.stackplot(x, [self.base_load, self.ev_load[tot]], labels=['Base Load', 'EV Load'])
        ax.set_ylabel('Power [MW]')
        if self.ss_pmax > 0:
            ax.axhline(self.ss_pmax, label='Pmax', linestyle='--', color='red')
            
        self.set_aspect_plot(ax, day_ini=day_ini, days=days, **plot_params)
        return ax
        
    def plot_flex_pot(self, day_ini=0, days=-1, trajectory=False, **plot_params):
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
        ax.set_ylabel('EV energy storage [MWh]')
        
        self.set_aspect_plot(ax, day_ini=day_ini, days=days, **plot_params)
        return ax
    
    def get_global_data(self):
        """ Some global info
        """
        total = 'Total'
        total_ev_charge = self.ev_load[total].sum() * self.period_dur #MWh
        flex_pot = self.ev_off_peak_potential[total].sum() * self.period_dur
        extra_charge = sum(ev.extra_energy.sum()
                            for ev in self.get_evs()) / util.k
        ev_flex_ratio = 1-total_ev_charge / flex_pot
        max_ev_load = self.ev_load[total].max()
        max_load = (self.ev_load[total] + self.base_load).max()
        max_base_load = self.base_load.max()
        peak_charge = max_load / self.ss_pmax
        h_overload = ((self.ev_load[total] + self.base_load) > self.ss_pmax).sum() * self.period_dur
        return dict(Tot_ev_charge_MWh = total_ev_charge,
                    Extra_charge_MWh = extra_charge,
                    Base_load_MWh= self.base_load.sum() * self.period_dur,
                    Flex_ratio = ev_flex_ratio,
                    Max_ev_load_MW = max_ev_load,
                    Max_base_load_MW = max_base_load,
                    Max_load_MW = max_load,
                    Peak_ss_charge_pu = peak_charge,
                    Hours_overload = h_overload
                )
    
    def get_ev_data(self):
        """ EV charge data per subset
        """
        types = [t for t in self.evs_sets]
        charge = [self.ev_load[t].sum() * self.period_dur 
                  for t in types]
        nevs = [len(self.evs_sets[t])
                for t in types]
        extra_charge = [sum(ev.extra_energy.sum()
                            for ev in self.evs_sets[t]) / util.k
                        for t in types]
        flex_ratio = [1 - self.ev_load[t].sum() / self.ev_off_peak_potential[t].sum() 
                    for t in types]
        max_load = [self.ev_load[t].max() 
                    for t in types]
        avg_d = [np.mean([ev.dist_wd 
                         for ev in self.evs_sets[t]])
                for t in types]
        avg_plugin = [np.mean([ev.ch_status.sum() 
                         for ev in self.evs_sets[t]]) / self.ndays
                for t in types]
        return dict(EV_sets = types,
                    N_EVs = nevs,
                    EV_charge_MWh = charge,
                    Extra_charge = extra_charge,
                    Flex_ratio = flex_ratio,
                    Max_load = max_load,
                    Avg_daily_dist = avg_d,
                    Avg_plug_in_ratio= avg_plugin)
        
        
    def hist_dist(self, weekday=True, **plot_params):
        """ Do histogram of distances (weekday)
        """
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        d = np.array([ev.dist_wd if weekday else ev.dist_we 
                                 for types in self.evs_sets
                                 for ev in self.evs_sets[types]])
        avg = d.mean()
        h, _, _ = ax.hist(d, bins=np.arange(0,100,2), **plot_params)
        ax.axvline(avg, color='r', linestyle='--')
        ax.text(x=avg+1, y=h.max()*0.75, s='Average one-way trip distance {} km'.format(np.round(avg,decimals=1)))
        ax.set_xlim([0,100])
        ax.set_title('Histogram of trip distances')
        ax.set_xlabel('km')
        ax.set_ylabel('Frequency')
    
    
    def hist_ncharging(self, **plot_params):
        """ Do histogram of number of charging sessions
        """
        if not 'ax' in plot_params:
            f, ax = plt.subplots(1,1)
        else:
            ax = plot_params['ax']
        d = np.array([ev.n_plugs for ev in self.get_evs()])/(self.ndays/7)
        avg = d.mean()
        bins = np.arange(0, 9, 1)
        bins[-1] = 10
        h, _, _ = ax.hist(d, bins=bins, **plot_params)
        ax.set_xlim([0, 8])
        ax.axvline(avg, color='r', linestyle='--')
        ax.text(x=avg+1, y=h.max()*0.75, s='Average charging sessions per week: {}'.format(np.round(avg,decimals=1)))
        ax.set_title('Histogram of charging sessions per week')
        ax.set_xlabel('# charging sessions per week')
        ax.set_ylabel('Frequency')
        ax.set_xticklabels([str(i) for i in range(8)] + ['$\infty$'] )
    
    def get_ev(self):
        """ Returns random ev
        """
        return np.random.choice(list(self.evs.values()))
    
    def get_evs(self, key=None):
        """ Returns list of evs
        """
        if key in self.evs_sets:
            return self.evs_sets[key]
        return list(self.evs.values())
    
    def export_ev_data(self, atts=''):
        """ returns a dict with ev data
        atts : attributes to export
        """
        if atts == '':
            atts = ['name', 'batt_size', 'charging_power', 'bus', 'dist_wd', 'dist_we']
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
    
    def reset(self):
        """ Resets the status of the grid and of EVs
        """
        self.day = 0
        self.init_load_vector(self.base_load)
        for types in self.evs_sets:
            self.init_ev_vectors(types)
        for ev in self.evs.values():
            ev.reset(self)
    
    def set_evs_param(self, param, value, sets='all'):
        if sets == 'all':
            evs = self.get_evs()
        else:
            evs = self.evs[sets]
        for ev in evs:
            setattr(ev, param, value)
            ev.compute_derived_params(self)        
#    
#    def save_data(self, micro=False, macro=True, ):
        
        
class EV:
    """ Basic EV model with dumb charging
    """
    bins_dist = np.linspace(0, 100, num=51)
    def __init__(self, model, name, 
                 dist_wd=None,
                 dist_we=None,
                 var_dist_wd=0,
                 var_dist_we=0,
                 charging_power=3.6, 
                 charging_eff=0.95,
                 discharging_eff=0.95,
                 charging_type='if_needed',
                 tou_ini=0,
                 tou_end=0,
                 tou_we=False,
                 tou_ini_we=0,
                 tou_end_we=0,
                 driving_eff=0.2, 
                 batt_size=40,
                 range_anx_factor=1.5,
                 n_if_needed=0,
                 extra_trip_proba=0,
                 arrival_departure_data_wd=dict(),
                 arrival_departure_data_we=dict(mu_arr=16, mu_dep=8,
                                                std_arr=2, std_dep=2),
                 charge_schedule=None,
                 bus='',
                 target_soc=1,
                 ovn=True,
                 flex_time=0,
                 vcc=None,
                 boss=None,
                 **kwargs):
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
        # dist_wx can be a {} with either 's', 'm', 'loc' for lognorm params or with 'cdf', 'bins'
        self.dist_wd = set_dist(dist_wd)
        self.dist_we = set_dist(dist_we) 
        self.var_dist_wd = var_dist_wd
        self.var_dist_we = var_dist_we
        # Discrete random distribution (non correlated) for battery & charging power
        if type(charging_power) is int or type(charging_power) is float:
            self.charging_power = charging_power
        elif len(charging_power) == 2:
            self.charging_power = discrete_random_data(charging_power[0], charging_power[1])
        else:
            ValueError('Invalid charging_power value')     
        if type(batt_size) is int or type(batt_size) is float:
            self.batt_size = batt_size
        elif len(batt_size) == 2:
            self.batt_size = discrete_random_data(batt_size[0], batt_size[1])
        else:
            ValueError('Invalid batt_size value')               
        self.charging_eff = charging_eff                # Charging efficiency, in pu
        self.discharging_eff = discharging_eff          # Discharging efficiency, in pu
        self.driving_eff = driving_eff                  # Driving efficiency kWh / km
        self.min_soc = 0.2                              # Min SOC of battery to define plug-in
        self.max_soc = 1                                # Max SOC of battery to define plug-in
        self.target_soc = target_soc                    # Target SOC for charging process
        self.n_trips = 2                                # Number of trips per day (Go and come back)
        self.extra_trip_proba = extra_trip_proba        # probability of extra trip
        if not charging_type in ['if_needed', 'if_needed_sunday', 'all_days', 'if_needed_weekend', 'weekdays', 'weekdays+1']:
            ValueError('Invalid charging type %s' %charging_type) 
        self.charging_type = charging_type              # Charging behavior (every day or not)
        if charging_type in ['if_needed_weekend']: # Forced charging on Friday, Saturday or Sunday
            self.forced_day = np.random.randint(low=0,high=3,size=1)+4
        elif charging_type == 'if_needed_sunday':
            self.forced_day = 6
        else:
            self.forced_day = 8                         # No forced day 
        self.range_anx_factor = range_anx_factor        # Range anxiety factor for "if needed" charging
        self.n_if_needed = n_if_needed                  # Factor for probabilitic "if needed" charging. High means high plug in rate, low means low plug in rate
        self.tou_ini = tou_ini                          # Time of Use (low tariff) start time (default=0) 
        self.tou_end = tou_end                          # Time of Use (low tariff) end time (default=0)
        self.tou_we = tou_we                            # Time of Use for weekend. If false, it's off peak the whole weekend
        if tou_we:
            self.tou_ini_we = tou_ini_we
            self.tou_end_we = tou_end_we
        if charge_schedule is None:    
            self.arrival_departure_data_wd = arrival_departure_data_wd
            self.arrival_departure_data_we = arrival_departure_data_we
            self.ovn = ovn                                  # Overnight charging Bool
            self.charge_schedule = None
        else:
            cols = ['ArrTime', 'ArrDay', 'DepTime', 'DepDay', 'TripDistance']
            for c in cols:
                if not (c in charge_schedule):
                    raise ValueError('Charge schedule should have the following columns: {}'.format(cols))
            self.charge_schedule = charge_schedule
            
        # Parameter to compute 'physical realisations' of flexibility 
        # Flex service corresponds to a sustained injection during flex_time minutes
        if flex_time:
            if type(flex_time) == int:
                flex_time = [flex_time]
            for f in flex_time:
                if f % model.step > 0:
                    raise ValueError('Flexibility time [{} minutes] should be a ' +
                                     'multiple of model.step [{} minutes]'.format(flex_time, model.step))
        self.flex_time = flex_time                  # array of Time (minutes) for which the flex needs to be delivered
        
        # Variable capacity contract/limit. It's either a np.array of length>= model.periods
        # or a constant value
        if type(vcc) in (float, int):
            self.vcc = vcc * np.ones(model.periods)
        else:
            self.vcc = vcc
            
        # DERIVED PARAMS
        self.compute_derived_params(model)
        # Correct target SOC to minimum charged energy
        if self.target_soc < 1:
            required_min_soc = float(min(self.dist_wd * self.n_trips * self.range_anx_factor * self.driving_eff / self.batt_size, 1))
            if required_min_soc > self.target_soc:
                self.target_soc = required_min_soc
        
        # Grid Params
        self.bus = ''
        
        # Aggregator params
        self.boss = boss
        
        # RESULTS/VARIABLES
        self.soc_ini = np.zeros(model.ndays)        #list of SOC ini at each day (of charging session)
        self.soc_end = np.zeros(model.ndays)        #list of SOC end at each day (of charging session)
        self.energy_trip = np.zeros(model.ndays)    #list of energy consumed per day in trips
        self.charged_energy = np.zeros(model.ndays) # charged energy into the battery [kWh]
        self.extra_energy = np.zeros(model.ndays)   # extra charge needed during the day (bcs too long trips, not enough batt!) [kWh]
        self.ch_status = np.zeros(model.ndays)      # Charging status for each day (connected or not connected)
        self.n_plugs = 0
        
        self.set_ch_vector(model)
        self.set_off_peak(model)
        
    def compute_derived_params(self, model):    
        """ Computes derived params that are useful
        """
        self.eff_per_period = model.period_dur * self.charging_eff 
        self.soc_eff_per_period = self.eff_per_period / self.batt_size
        self.soc_v2geff_per_period = model.period_dur / self.batt_size / self.discharging_eff
    

    def set_arrival_departure(self, mu_arr = 18, mu_dep = 8, 
                                    std_arr = 2, std_dep = 2,
                                    **kwargs):
        """ Sets arrival and departure times
        """ 
        # If data is a 2d pdf (correlated arrival and departure)
        if 'pdf_a_d' in kwargs:
            if 'bins' in kwargs:
                bins = kwargs['bins'] 
            else:
                bins = bins_hours
            self.arrival, self.departure = random_from_2d_pdf(kwargs['pdf_a_d'], bins)
            # THIS IS SEMI-GOOD! CORRECT!!
            dt = (self.departure - self.arrival if self.departure > self.arrival
                           else self.departure + 24 - self.arrival)
        # else, if data is two cdf (not correlated)
        # else, random from a normal distribution with mu and std_dev from inputs
        else:
            dt = 0
            # Why this 3! completely arbitrary!!!
            while dt < 3:
                if 'cdf_arr' in kwargs:
                    self.arrival = random_from_cdf(kwargs['cdf_arr'], bins_hours)
                else:
                    self.arrival = np.random.randn(1) * std_arr + mu_arr                    
                if 'cdf_dep' in kwargs:
                    self.departure = random_from_cdf(kwargs['cdf_dep'], bins_hours)
                else:
                    self.departure = np.random.randn(1) * std_dep + mu_dep
                dt = (self.departure - self.arrival if not self.ovn
                      else self.departure + 24 - self.arrival)
        self.dt = dt
        
    def set_ch_vector(self, model):
        # Grid view
        self.charging = np.zeros(model.periods)             # Charging power at time t  [kW]
        self.off_peak_potential = np.zeros(model.periods)   # Connected and chargeable power (only off-peak period and considering VCC) [kW]
        self.potential = np.zeros(model.periods)            # Connected power at time t [kW]
        self.up_flex = np.zeros(model.periods)              # Battery flex capacity, upper bound [kWh]
        self.dn_flex = np.zeros(model.periods)              # Battery flex capacity, lower bound (assumes bidir charger) [kWh]
        self.mean_flex_traj = np.zeros(model.periods)       # Mean trajectory to be used to compute up & dn flex [soc?]
        self.soc = np.zeros(model.periods)                  # SOC at time t [pu]
        if self.flex_time:                                 # kW of flexibility for self.flex_time minutes, for diffs baselines
            self.up_flex_kw = np.zeros([len(self.flex_time), model.periods])
            self.dn_flex_kw = np.zeros([len(self.flex_time), model.periods])
#            self.up_flex_kw_meantraj =  np.zeros(model.periods)
#            self.up_flex_kw_immediate =  np.zeros(model.periods)
#            self.up_flex_kw_delayed =  np.zeros(model.periods)
#            self.dn_flex_kw_meantraj =  np.zeros(model.periods)
#            self.dn_flex_kw_immediate =  np.zeros(model.periods)
#            self.dn_flex_kw_delayed =  np.zeros(model.periods)
   
    def set_off_peak(self, model):
        """ Sets vector for off-peak period (EV will charge only during this period)
        """
        # TODO: expand to different off-peak hours during the weekend
        self.off_peak = np.ones(model.periods)
        if self.tou_ini != self.tou_end:
            delta_op = self.tou_end>self.tou_ini
            # Compute one day. Assume Off peak hours are less than On peak so less modifs to 1
            op_day = np.zeros(model.periods_day)
            op_day_we = np.ones(model.periods_day)
            for i in range(model.periods_day):
                if delta_op:
                    if (self.tou_ini <= i * model.period_dur < self.tou_end):
                        op_day[i] = 1
                else:
                    if not (self.tou_end <= i*model.period_dur < self.tou_ini):
                        op_day[i] = 1
            if self.tou_we:
                delta_op = self.tou_end_we > self.tou_ini_we
                for i in range(model.periods_day):
                    if delta_op:
                        if not (self.tou_ini_we <= i * model.period_dur < self.tou_end_we):
                            op_day_we[i] = 0
                    else:
                        if (self.tou_end_we <= i*model.period_dur < self.tou_ini_we):
                            op_day_we[i] = 0
            
            for d in range(model.ndays):
                if not (model.days[d] in model.weekends):
                    self.off_peak[d * model.periods_day: (d+1) * model.periods_day] = op_day
                elif self.tou_we:
                    self.off_peak[d * model.periods_day: (d+1) * model.periods_day] = op_day_we
#        if self.tou_ini < self.tou_end:
#            for i in range(model.periods):
#                if ((not self.tou_we) and (model.times[i][2] in model.weekends)):
#                    continue
#                    # This checks that there is no special ToU in weekends, and that it is not the weekend
#                if model.times[i][1] < self.tou_ini or model.times[i][1] >= self.tou_end:
#                    self.off_peak[i] = 0
#        elif self.tou_ini > self.tou_end:
#            for i in range(model.periods):
#                if ((not self.tou_we) and (model.times[i][2] in model.weekends)):
#                    continue
#                if self.tou_end <= model.times[i][1] < self.tou_ini:
#                        self.off_peak[i] = 0
                    
    def compute_energy_trip(self, model):
        """ Computes the energy used during the current day trips and to be charged
        in the current session.
        """
        # TODO: extend to add stochasticity
        dist = (self.dist_wd + max(0, np.random.normal() * self.var_dist_wd) 
                    if model.days[model.day] < 5 
                    else self.dist_we + max(0, np.random.normal() * self.var_dist_we))
        if dist * self.n_trips * self.driving_eff > self.batt_size:
            #This means that home-work trip is too long to do it without extra charge, 
            # so forced work charging (i.e one trip)
            self.energy_trip[model.day] = dist * self.driving_eff
            self.extra_energy[model.day] = (self.n_trips - 1) * self.energy_trip[model.day]
        else:
            extra_trip = 0 
            if np.random.rand(1) < self.extra_trip_proba:
                # Extra trip probability, normal distribution around 5km +- 1.5 km
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
            self.extra_energy[model.day] += (0.05 - self.soc_ini[model.day]) * self.batt_size
            self.soc_ini[model.day] = 0.05
            
    def define_charging_status(self, model, next_trip_energy=None):
        """ Defines charging status for the session. 
        True means it will charge this session
        """
        # TODO: How to compute next_trip?
        if next_trip_energy is None:
            next_trip_energy = ((self.dist_wd if model.days[model.day + 1] < 5 
                                                else self.dist_we) * 
                                self.n_trips * self.driving_eff)
        min_soc_trip = max(next_trip_energy * self.range_anx_factor / self.batt_size, self.min_soc)
        
        # TODO: Other types of charging ?
        if self.charging_type == 'all_days':
            return True
        if self.charging_type in 'weekdays':
            if not model.days[model.day] in model.weekends:
                return True
            return False
        if self.charging_type == 'weekdays+1':
            if not model.days[model.day] in model.weekends:
                return True
            return np.random.rand() > 0.5
#        if self.charging_type == 'weekends':
#            # TODO: Complete if charging is needed
#            if model.days[model.day] in model.weekends:
#                return True
#            return False
        if self.charging_type in ['if_needed', 'if_needed_sunday']:
        # Enough kWh in batt to do next trip?
            if model.days[model.day] == self.forced_day:
                # Force charging for this EV in this day of the week
                return True
            if (self.soc_ini[model.day]  < min_soc_trip):
                # Charging because it is needed for expected trips of next day
                return True
            if self.soc_ini[model.day] >= self.max_soc:
                # Not charging beceause EV has more than the max SOC
                return False
            if self.n_if_needed >=100:
                # Deterministic case, where if it is not needed, no prob of charging
                return True
            # If n_if_needed = 0, deterministic If_needed
            # else: Probabilistic charging: higher proba if low SOC
            # n_if_needed == 1 is a linear probability
            p = np.random.rand(1)
            p_cut = 1-((self.soc_ini[model.day] - min_soc_trip) / (self.max_soc - min_soc_trip)) ** self.n_if_needed
            return p < p_cut

    def do_charging(self, model):
        """ Computes charging potential and calls charging function
        """
        # Computes index for charging session
        delta_day = model.day * model.periods_day
        tini = int(self.arrival * model.periods_hour)
        delta_session = int((self.arrival + self.dt) * model.periods_hour)
#        if self.departure < self.arrival:
#            tend = int((self.departure + 24) * model.periods_hour)    
#        else:
#            tend = int(self.departure *  model.periods_hour)
        idx_tini = max([0, delta_day + tini])
        idx_tend = min([delta_session + delta_day, model.periods-1])
#        idx_tend = min([delta + tend, model.periods-1])
        
        if idx_tini >= idx_tend:
            self.do_zero_charge(model, idx_tini, idx_tend)
            self.compute_soc_end(model, idx_tend)
            return idx_tini, idx_tend
        # Potential charging vector
        potential = np.ones(idx_tend+1-idx_tini) * self.charging_power
        # Correct for arrival period 
        potential[0] = ((model.period_dur - self.arrival % model.period_dur ) / 
                 model.period_dur * self.charging_power)
        # And correct for departure period
        potential[-1] = (self.departure % model.period_dur / 
                 model.period_dur * self.charging_power)
        
        # Check for aggregators limit
        if not (self.boss is None):
            if self.boss.param_update in ['capacity', 'all']:
                potential = np.min([potential, self.boss.available_capacity[idx_tini:idx_tend+1]], axis=0)
        # Check for own variable capacity limit
        if not (self.vcc is None):
            potential = np.min([potential, self.vcc[idx_tini:idx_tend+1]], axis=0)
            
        # Save in potential and off peak vectors
        self.potential[idx_tini:idx_tend+1] = potential
        self.off_peak_potential[idx_tini:idx_tend+1] = (
                potential * self.off_peak[idx_tini:idx_tend+1])
        
        # calls functions that update things
        self.compute_up_dn_flex(model, idx_tini, idx_tend)
        self.compute_charge(model, idx_tini, idx_tend)
        self.compute_soc_end(model, idx_tend)
        if self.flex_time:
            self.compute_up_dn_flex_kw(model, idx_tini, idx_tend)
        return idx_tini, idx_tend
    
    def do_zero_charge(self, model, idx_tini, idx_tend):
        """ Does a default 0 charge. Used in case SOC_ini > target_soc
        """
        self.charged_energy[model.day] = 0
        self.soc[idx_tini:idx_tend+1] = self.soc_ini[model.day]
        self.charging[idx_tini:idx_tend+1] = 0
            
    def compute_charge(self, model, idx_tini, idx_tend):
        """ do default charge: dumb
        This function should be extended by classes to do smart charging algos
        """
        if self.soc_ini[model.day] >= self.target_soc:
            self.do_zero_charge(model, idx_tini, idx_tend)
            return
        # off peak potential in charging power (kW)
        opp = self.off_peak_potential[idx_tini:idx_tend+1]
        
        # SOC is computed as the cumulative sum of charged energy  (Potential[pu] * Ch_Power*Efficiency*dt / batt_size) 
        soc = (opp.cumsum() * self.soc_eff_per_period + self.soc_ini[model.day]).clip(0, self.target_soc)
        # charging power
        power = (soc - np.concatenate([[self.soc_ini[model.day]], soc[:-1]])) / self.soc_eff_per_period 
        
        # charged_energy
        self.charged_energy[model.day] = (soc[-1]-self.soc_ini[model.day]) * self.batt_size
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
        target = max(self.target_soc, self.soc_ini[model.day])
        
        soc_up = (potential.cumsum() * self.soc_eff_per_period  + self.soc_ini[model.day]).clip(0,1)
        soc_dn = (-potential.cumsum() * self.soc_v2geff_per_period + self.soc_ini[model.day]).clip(m_soc,1)
        
        soc_end = min(soc_up[-1], target)
        target_v2g = min(soc_up[-1], self.target_soc)
        downwards_soc = np.concatenate([[target_v2g], target_v2g - potential[:-1][::-1].cumsum() * self.soc_eff_per_period])[::-1]
        soc_dn = np.maximum(soc_dn, downwards_soc)
        
        avg_ch_pu = (soc_end - self.soc_ini[model.day]) / (self.dt / model.period_dur)
        
        self.up_flex[idx_tini:idx_tend+1] = soc_up * self.batt_size
        self.dn_flex[idx_tini:idx_tend+1] = soc_dn * self.batt_size
        self.mean_flex_traj[idx_tini:idx_tend+1] = (self.soc_ini[model.day] + pu_pot.cumsum() * avg_ch_pu) * self.batt_size

    def compute_up_dn_flex_kw(self, model, idx_tini, idx_tend):
        """ Computes up and down flex in terms of power [kW] that can be sustained for a flex_time, 
        to be delivered according to computed SOC trajectoy.
        This values are Power to be seen from the grid, not flexibility wrt a given baseline
        """
        # Flex time, in model steps
        for i, f in enumerate(self.flex_time):
            flex_steps = int(f / model.step)
            
            if flex_steps > (idx_tend-idx_tini):
                return
            
            # Upper bounds and lower bounds on SOC, considering the flex_steps shift:
            soc_end = min(self.soc[idx_tend], self.target_soc) * self.batt_size
    #        soc_ini = self.soc_ini[model.day] * self.batt_size
            low_bound = np.concatenate((self.dn_flex[idx_tini+(flex_steps-1):idx_tend+1], 
                                       np.ones(flex_steps-1) * soc_end))  
            high_bound = np.concatenate((self.up_flex[idx_tini+(flex_steps-1):idx_tend+1], 
                                        np.ones(flex_steps-1) * self.batt_size)) # Max high bound is always max batt_size
            
            # SOC baselines:
            soc_b = np.concatenate(([self.soc_ini[model.day]], self.soc[idx_tini:idx_tend])) * self.batt_size
                
            # Up and down kWs based on allowed battery trajectories:
            kwh_to_kw_up = 1/((f / 60) * self.charging_eff)
            kwh_to_kw_dn = 1/(f / 60) * self.discharging_eff
            
            self.up_flex_kw[i, idx_tini:idx_tend+1]  = ((high_bound-soc_b) * kwh_to_kw_up).clip(-self.charging_power, self.charging_power)
            self.dn_flex_kw[i, idx_tini:idx_tend+1]  = ((low_bound-soc_b) * kwh_to_kw_dn).clip(-self.charging_power, self.charging_power)
     
    def compute_soc_end(self, model, idx_tend=False):
        """ Calculates SOC at the end of the charging session
        """
        if self.ch_status[model.day]:
            self.soc_end[model.day] = self.soc[idx_tend]
        else:
            self.soc_end[model.day] = self.soc_ini[model.day]
            
    def new_day(self, model, get_idxs=False):
        # Update for a new day:
        # Compute arrivals and departures
        if not (self.charge_schedule is None):
           idxs = self.scheduled(model, get_idxs=True)
           return idxs
        if model.days[model.day] in model.weekends:
                self.set_arrival_departure(**self.arrival_departure_data_we)                    
        else:
            self.set_arrival_departure(**self.arrival_departure_data_wd)
        # Computes initial soc based on effected trips
        self.compute_energy_trip(model)
        self.compute_soc_ini(model)
        # Defines if charging is needed
        self.ch_status[model.day] = self.define_charging_status(model)
        if self.ch_status[model.day]:
            self.n_plugs += 1
            idxs = self.do_charging(model)
        else:
            self.compute_soc_end(model)
            idxs =  0, 0
        if get_idxs:
            return idxs
        
    def scheduled(self, model, get_idxs=False):
        # Update for a new day, for an EV with given schedules (a DataFrame of arrivals, departures and trips)
        sessions = self.charge_schedule[self.charge_schedule.ArrDay == model.day]
        n = 0
        idxs = [0,0]
        if (len(sessions) == 0) & (self.soc_end[model.day]==0):
            self.soc_end[model.day] = self.soc_end[model.day-1]
        for i, s in sessions.iterrows():            
            # do session:
            # set arr, dep
            self.arrival = s.ArrTime
            self.departure = s.DepTime
            self.dt = s.DepTime - s.ArrTime + 24 * (s.DepDay - s.ArrDay)
            # set Energy of trip:
            self.energy_trip[model.day] += min(self.batt_size, s.TripDistance * self.driving_eff)
            self.extra_energy[model.day] += max(s.TripDistance * self.driving_eff - self.batt_size, 0)
            # compute soc ini:
            if n == 0:
                self.compute_soc_ini(model)
            else:
                self.soc_ini[model.day] = max(self.soc_end[model.day] - (s.TripDistance * self.driving_eff / self.batt_size),0)
            # def charging status, next trip distance is known from schedule
            try:
                next_trip_energy = self.charge_schedule.TripDistance[i+1]*self.driving_eff
            except:
                next_trip_energy = self.charge_schedule.TripDistance[i]*self.driving_eff
            self.ch_status[model.day] = self.define_charging_status(model, 
                                        next_trip_energy=next_trip_energy)
            # do charging (or not)
            if self.ch_status[model.day]:
                self.n_plugs += 1
                ixs = self.do_charging(model)
                # Updating indexes to consider
                if idxs == [0,0]:
                    idxs = list(ixs)
                else:
                    idxs[1] = ixs[1]
            else:
                self.compute_soc_end(model)
            # next session
            n += 1
            # Correcting End soc for sessions lasting multiple days (same soc end for those days)
            if s.DepDay > s.ArrDay+1:
                self.soc_end[model.day:s.DepDay-1] = self.soc_end[model.day]
        return idxs
        
    def reset(self, model):
        self.soc_ini = self.soc_ini * 0                 #list of SOC ini at each day (of charging session)
        self.soc_end = self.soc_end * 0                 #list of SOC end at each day (of charging session)
        self.energy_trip = self.energy_trip * 0         #list of energy consumed per day in trips
        self.charged_energy = self.charged_energy * 0   # charged energy into the battery
        self.extra_energy = self.extra_energy * 0       # extra charge needed during the day (bcs too long trips, not enough batt!)
        self.ch_status = self.ch_status * 0             # Charging status for each day (connected or not connected)
        self.n_plugs = 0

        self.set_ch_vector(model)
        self.set_off_peak(model)

class EV_Modulated(EV):
    """ Class of EV that does a flat charge over the whole period of connection (considering ToU constraints)
    """
    def __init__(self, model, name, pmin_charger=0.6, **params):
        super().__init__(model, name, **params)
        self.pmin_charger = pmin_charger                    # in pu of max charging power
        
    def compute_charge(self, model, idx_tini, idx_tend):
        """ compute Modulated charge
        """
        if self.soc_ini[model.day] >= self.target_soc:
            self.do_zero_charge(model, idx_tini, idx_tend)
            return
        # off peak potential in "per unit" of time step
        opp = self.off_peak_potential[idx_tini:idx_tend+1] / self.charging_power  
        needed_soc = self.target_soc - self.soc_ini[model.day]
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
        self.charged_energy[model.day] = (soc[-1]-self.soc_ini[model.day]) * self.batt_size
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
        if self.soc_ini[model.day] >= self.target_soc:
            self.do_zero_charge(model, idx_tini, idx_tend)
            return
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
        self.charged_energy[model.day] = (soc[-1]-self.soc_ini[model.day]) * self.batt_size
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
        if self.soc_ini[model.day] >= self.target_soc:
            self.do_zero_charge(model, idx_tini, idx_tend)
            return
        # off peak potential in "per unit" of time step
        opp = self.off_peak_potential[idx_tini:idx_tend+1]  
        
        
        # reverse charging
        rev_ch = (opp[::-1].cumsum()[::-1] * self.soc_eff_per_period).clip(
                            max=self.target_soc-self.soc_ini[model.day])
        soc = self.soc_ini[model.day] + (rev_ch.max() - rev_ch)
        # charging power
        power = (soc - np.concatenate([[self.soc_ini[model.day]], soc[:-1]])) / self.soc_eff_per_period 
        
        # charged_energy
        self.charged_energy[model.day] = (soc[-1]-self.soc_ini[model.day]) * self.batt_size
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power
        

class EV_pfc(EV):
    """ Simulates the participation of an EV in symetrical frequency response.
    It follows P. Codani algorithm (see Codani PhD Thesis, 2016, U. Paris Saclay)
    
    """
    def __init__(self, model, name, pop_dur=1, **params):
        super().__init__(model, name, **params)
        self.low_bound = 0.2  
        self.high_bound = max(0.9, self.target_soc)
        self.pop_dur = pop_dur # Duration of Prefered Operating Point, in Hours
        self.steps_pop = int(pop_dur / model.period_dur)
        self.pop = np.zeros(model.periods)
        self.pbid = np.zeros(model.periods)
        
        self.boundsoc = np.zeros(model.periods)
     
    def POP(self, soc, dn_soc, pop_d):
        """ Define Prefered Operating Point
        """
        ph = min(self.charging_power, max(((self.high_bound-soc) * self.batt_size / pop_d), -self.charging_power))
        pl = max(-self.charging_power, min(((dn_soc - soc) * self.batt_size / pop_d), self.charging_power))
        pop = (ph+pl)/2
        pbid = self.charging_power - abs(pop)
        return pop, pbid
        
    def compute_charge(self, model, idx_tini, idx_tend):
        """ Compute charge given system frequency
        """        
        dsteps = idx_tend + 1 - idx_tini
        # The max attainable SOC of the session
        max_soc_d = self.soc_ini[model.day] + self.dt * self.charging_power * self.charging_eff / self.batt_size
        opp = self.off_peak_potential[idx_tini:idx_tend+1]
        if max_soc_d <= self.target_soc:
            soc = (opp.cumsum() * self.soc_eff_per_period + self.soc_ini[model.day])
            power = opp
            pop = power
            pbid = np.zeros(dsteps)
        elif self.steps_pop >= dsteps:
            self.do_zero_charge(model, idx_tini, idx_tend)
            return
        else:
            # define bounds for SOC
            dn_soc = (self.target_soc - np.concatenate((opp[::-1].cumsum()[:0:-1] * self.soc_eff_per_period, [0]))).clip(min=self.low_bound)
            # Frequency deviations signals from Grid
            mus = model.mu[idx_tini:idx_tend+1]
            mus_up = model.mu_up[idx_tini:idx_tend+1]
            mus_dn = model.mu_dn[idx_tini:idx_tend+1] 
            dts_up = model.dt_up[idx_tini:idx_tend+1]
            dts_dn = model.dt_dn[idx_tini:idx_tend+1]
            
            
            # charging power and soc evolution
            power = np.zeros(dsteps)
            soc = np.zeros(dsteps)
            pop = np.zeros(dsteps)
            pbid = np.zeros(dsteps)
            
            # Compute POP for first iteration
            popt, pbidt =  self.POP(self.soc_ini[model.day], dn_soc[self.steps_pop], self.pop_dur)
            for i in range(dsteps):
            # for each Frequency time definition (1h?), define POP as Eq. 3.4 Codani Thesis. 
                if i >0 :
                    if (model.times[idx_tini + i][1] % self.pop_dur) == 0:
                        k = i + self.steps_pop
                        if k < dsteps:
                            popt, pbidt = self.POP(soc[i-1], dn_soc[k], self.pop_dur)
                        else:
                            popt, pbidt = self.POP(soc[i-1], dn_soc[-1], (dsteps-i) * model.period_dur)
            # for each time step get \mu (PFC operating point, according to freq dev)
                pop[i] = popt
                pbid[i] = pbidt
                power[i] = popt + pbidt * mus[i] 
                dnch = popt + pbidt*mus_dn[i]
                upch = popt + pbidt*mus_up[i]
                if dnch < 0:
                    k = self.soc_v2geff_per_period
                else:
                    k = self.soc_eff_per_period
                if upch < 0:
                    j = self.soc_v2geff_per_period
                else:
                    j = self.soc_eff_per_period
                delta_soc = dnch * k * dts_dn[i] + upch * j * dts_up[i]
                if i == 0:
                    soc[i] = self.soc_ini[model.day] + delta_soc
                else:
                    soc[i] = soc[i-1] + delta_soc
    
        # charged_energy
        self.charged_energy[model.day] = (soc[-1]-soc[0]) * self.batt_size
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power
        self.pop[idx_tini:idx_tend+1] = pop
        self.pbid[idx_tini:idx_tend+1] = pbid
        
        
class EV_optimcharge(EV):
    """ EV optimizes charging costs given a a variable (hourly, for ex) price of electricity
    """
    def __init__(self, model, name, **params):
        super().__init__(model, name, **params)
        self.cost = np.zeros(model.ndays)
    
    def compute_charge(self, model, idx_tini, idx_tend):
        """ Compute charge given system frequency
        """
        if self.soc_ini[model.day] >= self.target_soc:
            self.do_zero_charge(model, idx_tini, idx_tend)
            return
        if self.boss == None:
            prices = model.prices[idx_tini:idx_tend+1]
        else:
            prices = self.boss.prices[idx_tini:idx_tend+1]
        min_ch = (self.target_soc - self.soc_ini[model.day]) * self.batt_size # Minimum charge needed in kWh
        opp = self.off_peak_potential[idx_tini:idx_tend+1]
        max_ch = opp.sum() * self.eff_per_period  # Maximum charge in the session (considering capacity limits) in  kWh
        n = len(prices)
        if max_ch <= min_ch:
            soc = (opp.cumsum() * self.soc_eff_per_period + self.soc_ini[model.day])
            power = opp
        else:
            # Optimize V1G charge (x) using CVXOPT
            # min prices * x
            # st:   Gx <= h
            # -sum(x) <= min_charge/eff*dt
            # 0 <= x <= opp[t]
            G = np.vstack([-np.ones(n), np.eye(n), -np.eye(n)])
            h = np.hstack([-min_ch/self.eff_per_period, opp, np.zeros(n)])
            r = cvxopt.solvers.lp(cvxopt.matrix(prices), cvxopt.matrix(G), cvxopt.matrix(h), verbose=True)
            
            power = np.squeeze(r['x'])
            soc = self.soc_ini[model.day] + power.cumsum() * self.soc_eff_per_period
        
        # charged_energy
        self.charged_energy[model.day] = (soc[-1]-self.soc_ini[model.day]) * self.batt_size
        self.cost[model.day] = (power * prices).sum()
        self.soc[idx_tini:idx_tend+1] = soc
        self.charging[idx_tini:idx_tend+1] = power

#class EVSE(EV):
#    """ Public EVSE. Occupation varies each day. Can have multiple sessions per day.
#    """
#    def __init__(self, model, name, 
#                 arrival_departure_data_wd,
#                 arrival_departure_data_we,
#                 charging_power,
#                 requested_kwh,
#                 charging_proba,
#                 **params):
#        super.__init__(model, name, 
#                       **params)
#        self.requested_kwh_distr = requested_kwh_distr
#        self.requested_kwh = np.zeros(model.ndays)
##        self.model = model
##        self.name = name
##        self.charging_power = charging_power
##        
##        # arrival departure times
##        self.arrival_departure_data_wd = arrival_departure_data_wd
##        self.arrival_departure_data_we = arrival_departure_data_we
##        
##        # requested kwh per charging session
##        # this can be a distribution
##        self.requested_kwh = requested_kwh
##        # probability of charging session per day (<=1)
##        self.charging_proba = charging_proba
##        # EV charging params
##        self.soc_end = soc_end
##        self.soc_ini = soc_ini
##        # EV params
##        self.batt_size = batt_size
##     
#    
#     def compute_energy_trip(self, model):
#        self.requested_kwh[model.day] = random_from_cdf(self.requested_kwh_distr, bins)
#        
#        
#     def new_day(self, model, get_idxs=False):
#        # Update for a new day:
#        # Proba if there is a charging session:
#        if not self.check_charging_session(model):
#            return
#        # Compute arrival and departure
#        if model.days[model.day] in model.weekends:
#                self.set_arrival_departure(**self.arrival_departure_data_we)                    
#        else:
#            self.set_arrival_departure(**self.arrival_departure_data_wd)
#        # Computes initial soc based on requested kwhs
#        self.compute_energy_trip(model)
#        self.compute_soc_ini(model)
#        # Defines if charging is needed
#        self.ch_status[model.day] = self.define_charging_status(model)
#        if self.ch_status[model.day]:
#            idxs = self.do_charging(model)
#        else:
#            self.compute_soc_end(model)
#            idxs =  0, 0
#        if get_idxs:
#            return idxs   
#                
class Aggregator():
    """ Aggregator can interact with grid and EVs
    Updates EV signals (prices or limits)
    """
    def __init__(self, model, name, param_update='prices', 
                 price_elasticity=0.02, capacity=100):
        """  model= Grid
        name = Aggregator ID
        param_update = For each EV iteration, Aggregator updates Dynamic prices or available capacity
            options = 'e_prices', 'capacity', 'all'
        price_elasticity = To update dynamic price in c€/kW, for each extra kW of EV charge, prices change in c€
            It is based on Van Amstel thesis (2018), 
            Approx slope of NL market, 4 [€/MWh]/GW, Average size of neighborhood = 165 EVs out of 8M in country
            => Equivalent price elasticity of neighborhood => 4 [€/MWh] / GW * 8M/165 [National/local load]
            => 19.4 c€/MWh => 0.019 c€/kWh
            This value should be updated for considered Fleet sizes, EV penetrations, price slopes, etc
        capacity = Available capacity for the fleet [kW]. It can be also a numpy array of len(model.periods)
        Approach quite similar to game-theoretic algorithms (though simpler) found in 
            Beaude 2016 'Reducing the Impact of EV Charging Operations on
            the Distribution Network'
            
        """
        self.name = name
        self.model = model
        self.param_update = param_update
        valid_params = ['prices', 'capacity', 'all']
        if param_update in ['prices', 'capacity', 'all']:
            raise ValueError('Invalid parameter to update, valid are {}'.format(valid_params))
        if param_update in ['prices', 'all']:
            if hasattr(model, 'prices'):
                self.prices = model.prices
            else:
                raise ValueError('No prices defined for aggregator wanting to update prices')
                
        if param_update in ['capacity', 'all']:
            self.capacity = np.ones(model.periods) * capacity
            self.available_capacity = np.ones(model.periods) * capacity
        
        self.price_elasticity = price_elasticity
        self.nevs = 0
        self.evs = []    
        self.ev_charge = np.zeros(model.periods)

    def get_evs_randord(self):
        """ Return a list of EVs in a random order
        """
        np.random.shuffle(self.evs)
        return self.evs
    
    def update_prices(self, ev, idx_tini, idx_tend):
        """ Update internal prices to be sent to next evs
        """
        self.prices[idx_tini:idx_tend+1] += ev.charging[idx_tini:idx_tend+1] * self.price_elasticity
       
    def update_available_capacity(self, ev, idx_tini, idx_tend):
        """ Update 
        """
        self.available_capacity[idx_tini:idx_tend+1] -= ev.charging[idx_tini:idx_tend+1]
    
    def new_day(self):
        """ Computes new day
        Loop
            Calls an EV, sends updated info, EV computes day, sends back charging profile
            Updates prices/capacity
        """
        for ev in self.get_evs_randord():
            idx_tini, idx_tend = ev.new_day(self.model, get_idxs=True)
            if self.param_update in ['prices', 'all']:
                self.update_prices(ev, idx_tini, idx_tend)
            if self.param_update in ['capacity', 'all']:
                self.update_available_capacity(ev, idx_tini, idx_tend)