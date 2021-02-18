# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:51:45 2020

Useful functions for Pandapower grid

@author: U546416
"""

import pandapower as pp
import pandapower.topology as ppt
import pandapower.plotting as ppp
import pandapower.control as ppc

import util

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as ptc
#import polygons as pg
import matplotlib.patheffects as pe

import time

def plot_v_profile(net, ax=None, vmin=0.95, vmax=1.05):
    """ Plots voltage profile of a Pandapower grid
    """

    b0 = net.ext_grid.bus[0]
    distance_bus = ppt.calc_distance_to_bus(net, b0)

    if ax is None:
        fig, ax = plt.subplots() 
    # Draw lines 
    lines_c = [[(dbf, vbf), (dbt, vbt)]
               for dbt, vbt, dbf, vbf in 
               zip(distance_bus[net.line.from_bus],
                   net.res_bus.vm_pu[net.line.from_bus],
                   distance_bus[net.line.to_bus],
                   net.res_bus.vm_pu[net.line.to_bus],)]
    lc = LineCollection(lines_c, linestyle='--', color='k', label='Lines')
    ax.add_collection(lc)
    # Draw trafos
    trafos_c = [[(dbf, vbf), (dbt, vbt)]
                for dbt, vbt, dbf, vbf in 
                zip(distance_bus[net.trafo.hv_bus],
                    net.res_bus.vm_pu[net.trafo.hv_bus],
                    distance_bus[net.trafo.lv_bus],
                    net.res_bus.vm_pu[net.trafo.lv_bus],)]
    lt = LineCollection(trafos_c, linestyle='--', color='b', label='Tranformers')
    ax.add_collection(lt)
    
    plt.autoscale()
    
    if 'Feeder' in net.bus:
        fs = net.bus.Feeder.unique()
        # Writes Feeder name at the end of it
        for f in fs:
            if not ((f is None) or not (f==f)):
                if not 'SS' in f:
                    bm = distance_bus[net.bus[net.bus.Feeder == f].index].idxmax()
                    dm = distance_bus[bm]
                    vm = net.res_bus.vm_pu[bm]
                    ax.text(x=dm+0.5, y=vm, s=f)
        
    #Draws horizontal line at min and max V (in pu)
    ax.axhline(vmin, linestyle='dashed', color='red')
    ax.axhline(vmax, linestyle='dashed', color='red')
    
    # Plot nodes
    ok = (net.res_bus.vm_pu >= vmin) & (net.res_bus.vm_pu <= vmax)
    dok, vok = distance_bus[net.res_bus[ok].index], net.res_bus.vm_pu[net.res_bus[ok].index]
    dnok, vnok = distance_bus[net.res_bus[~ ok].index], net.res_bus.vm_pu[net.res_bus[~ ok].index]
    ax.plot(dok, vok, '*', color='b', markersize=3, picker=8, gid='ok1', label='_')
    ax.plot(dnok, vnok, '*', color='red', markersize=3, picker=8, gid='nok1', label='_')
    
    #plot ext_grid node
    n_eg = net.ext_grid.bus
    ax.plot(distance_bus[n_eg], net.res_bus.vm_pu[n_eg], 's', color='magenta', label='HV grid')     
    # 
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Voltage (pu)')
    ax.legend()
    
def print_losses_feeder(net):
    """ Prints losses info per Feeder for last results stored in the net
    Input: PandaPower net
    """
    if 'Feeder' in net.line:
        fs =  net.line.Feeder.unique()
        fs.sort()
    else:
        'No feeder data'
        print('')
    print('Feeder          Length[km]  Load[MW] ActLosses[MW] ActLosses[%]  ReactLosses[MVAr]')
    for f in fs:
        ls = net.line[net.line.Feeder == f].index
        bs = net.bus[net.bus.zone == f].index
        load = net.load[net.load.bus.isin(bs)].p_mw.sum()
        loss = net.res_line.pl_mw[ls].sum()
        lp = loss/load
        rl = net.res_line.ql_mvar[ls].sum()
        length = net.line.length_km[ls].sum()
        print('{:10}{:12.1f}{:13.3f}{:13.3f}{:13.3f}{:12.3f}'.format(f, length, load, loss, lp, rl))
    print('{:10}{:12.1f}{:13.3f}{:13.3f}{:13.3f}{:12.3f}'.format('TOTAL:', 
          net.line.length_km.sum(), 
          net.load.p_mw.sum(), 
          net.res_line.pl_mw.sum(),
          net.res_line.pl_mw.sum()/net.load.p_mw.sum(),
          net.res_line.ql_mvar.sum()))
    
def get_element_to_bus(net, bus, element='line'):
    
    if element == 'line':
        return list(net.line[net.line.to_bus == bus].index) + list(net.line[net.line.from_bus == bus].index)
    if element == 'trafo':
        return list(net.trafo[net.trafo.hv_bus == bus].index) + list(net.trafo[net.trafo.lv_bus == bus].index)
    
def get_farther_bus(net, zone=None):
    
    b0 = net.ext_grid.bus[0]
    distance_bus =  ppt.calc_distance_to_bus(net, b0)
    if zone is None:
        return distance_bus.idxmax()
    return distance_bus[net.bus[net.bus.zone == zone].index].idxmax()


def add_res_pv_rooftop(net, pv_penetration, iris_data, pv_cap_kw=4, 
                       lv_per_geo=None, loads=None):
    """ Adds rooftop PV at each load as static generator.
    net.load.name is the code of the IRIS
    net.load.n_trafo_iris is the number of LV transformers in the IRIS
    
    iris_data has the number of RES connections and of Shared dwellings:
        The PV per iris will be given by #RES_connections x (1-ShareCollectiveResidences)
        
    If loads is given, Residential PV rooftop will be created only for those loads
    """
    if loads is None:
        loads = net.load.index
    if (lv_per_geo is None):
        if ('n_trafo_iris' in net.load):
            lv_per_geo = net.load.ntrafo_iris
            lv_per_geo.index = net.load.zone
        else:
            print('No data of number of LV trafos per IRIS')
            return
    print('Creating home rooftop PV, with {:.1f}% penetration'.format(pv_penetration*100))
    cum_mw = 0
    npvs = 0
    for i, t in net.load.loc[loads].iterrows():
        b = t.bus
        iris = t.zone
        # Share of MW at each LV transfo
        n_trafos = lv_per_geo[iris]
        # number of PV
        npv = int(iris_data.Nb_RES[iris] * (100 - iris_data.Taux_logements_collectifs[iris])/100 * pv_penetration) 
        # MW of PV
        p_mw =  npv * pv_cap_kw/1000 * 1/n_trafos
        # cummulative MW
        cum_mw += p_mw
        npvs += npv * 1/n_trafos
        # Create transfo as static gen
        pp.create_sgen(net, bus=b, p_mw=p_mw, q_mvar=0, name='RES_PV_' + str(t['name']), type='RES_PV')    
    print('Created {:.3f} MW of PV, equivalent to {} installations'.format(cum_mw, int(npvs)))

def add_random_pv_rooftop(net, mean_size=0.1, std_dev=0.5, 
                          total_mw=2, buses=None, replace=False, name='randomPV', sgtype='PV'):
    """ Adds total_mw of random PV generator(s) as static generator to net.
    
    The size of PV plant will be given by a normal dist with mean mean_size and \sigma=std_dev
    # TODO: Use scipy any func
    
    It will continue to add PV gens until the total_mw is reached.
    
    The location of the generator is choosen randomly from available buses, with or without replacement
    (if no buses given, it can be in any bus of net)
    
    """
    if (buses is None):
        buses = list(net.bus.index)
    else:
        buses = list(buses)
    if replace:
        bb = buses
    # number of pv created
    npv = 0
    # cummulative capacity
    cum_mw = 0
    print('Creating random PV plants')
    while True:
        # Getting random location and size
        b = np.random.choice(buses)
        p_mw = np.random.normal(loc=mean_size, scale=std_dev)
        # creating sgen
        pp.create_sgen(net, bus=b, p_mw=p_mw, name=name+str(npv), type=sgtype)
        npv += 1
        cum_mw += p_mw
        # checking break
        if cum_mw>=total_mw:
            break
        if replace:
            buses.remove(b)
            # if i used all  
            if len(buses) == 0:
                buses = bb
    print('{} random PV plants created, for a total of {:.1f} MW'.format(npv, cum_mw))
    
#class droop_controller(ppc.basic_controller3):
    
            
class Profiler():
    """ It consists of two pandas tables:
        profile_idx = [columns=element, idx, variable, active, profile]
        profile_dv = [columns=profile, index=Series of time_stamps]
    """ 
    
    def __init__(self, net, profiles_db=None, profiles_idx=None):
        self.net = net
        self.profiles_db = profiles_db
        if profiles_idx is None:
            self.profiles_idx = pd.DataFrame(columns=['element', 'idx', 'variable', 'active', 'profile'])            
        else:
            self.profiles_idx = profiles_idx
            
    def add_profile_db(self, profile):
        if self.profiles_db is None:
            self.profiles_db = profile
        else:
            self.profiles_db = pd.concat([self.profiles_db, profile], axis=1)
    
    def add_profile_idx(self, element, idx, variable, profile, active=True):
        """ add profile idxs to internal table
        idx can be a series
        """
        df = []
        for i in idx:
            df.append([element, i, variable, active, profile])
        df = pd.DataFrame(df, columns=['element', 'idx', 'variable', 'active', 'profile'])
        self.profiles_idx = self.profiles_idx.append(df, ignore_index=True)
       
    def check_profiles(self):
        """ Checks that all active profiles have their correspondent in database
        """
        errs = []
        for p in self.profiles_idx.profile.unique():
            if not (p in self.profiles_db):
                print('Profile {} not available in databases'.format(p))
                errs.append(('profile', p))
        for i, t in self.profiles_idx.iterrows():
            if not (t.idx in self.net[t.element].index):
                print('Element {} with index {} not existent'.format(t.element, t.idx))
                errs.append(('idx', t.element, t.idx))
            if not (t.variable in self.net[t.element].columns):
                print('Variable {} for element {} not existent'.format(t.variable, t.element))
                errs.append(('variable', t.variable))
        return errs
    
    def update(self, ts):
        """ Update values for all profiles
        """ 
        if len(self.check_profiles()) > 0:
            raise ValueError('Invalid profiles!')
            
        for i, t in self.profiles_idx.iterrows():
            if t.active:
                self.net[t.element][t.variable][t.idx] = self.profiles_db.loc[ts, t.profile]
                
    def check_time_steps(self, tss):
        check= True
        for ts in tss:
            if not (ts in self.profiles_db.index):
                print('Time step {} not in profile database'.format(ts))
                check = False
        return check
   
class output_writer():
    def __init__(self, net, outputfolder='', feeder_pq=True):
        self.net = net
        self.res_v = {}
        self.res_line_load = {}
        self.res_loss_mw = {}
        self.res_loss_mvar = {}
        self.res_tot_load = {}
        self.res_trafo_mw = {}
        self.res_trafo_tap_pos = {}
        self.outputfolder = outputfolder
        self.consolidated = False
        self.sgen_gen = {}
        self.feeder_pq = feeder_pq
        if self.feeder_pq:
            self.p_feeder = {}
            self.q_feeder = {}
            self.lines0 = net.line[net.line.from_bus==0].index
            self.feeder = [f[-3:] for f in net.line[net.line.from_bus==0].name]
        
    def get_results(self, ts):
        self.res_v[ts] = self.net.res_bus.vm_pu
        self.res_line_load[ts] = self.net.res_line.loading_percent.max()
        self.res_loss_mw[ts] = self.net.res_line.pl_mw.sum()
        self.res_loss_mvar[ts] = self.net.res_line.ql_mvar.sum()
        self.res_tot_load[ts] = self.net.res_load.p_mw.sum()
        self.res_trafo_mw[ts] = self.net.res_trafo.p_hv_mw.sum()
        self.res_trafo_tap_pos[ts] = self.net.trafo.tap_pos[0]
        self.sgen_gen[ts] = self.net.res_sgen.p_mw.sum()
        if self.feeder_pq:
            self.p_feeder[ts] = self.net.res_line.p_from_mw[self.lines0]
            self.q_feeder[ts] = self.net.res_line.q_from_mvar[self.lines0]
        
    def consolidate(self):
        self.res_v = pd.DataFrame(self.res_v).T
        self.global_res = pd.DataFrame([self.res_line_load, self.res_loss_mw, self.res_loss_mvar, 
                                        self.res_tot_load, self.res_trafo_mw, self.res_trafo_tap_pos,
                                        self.sgen_gen],
                                        index=['MaxLineLoading_perc', 'ActLosses_MW', 'ReactLosses_MVar',
                                               'TotLoad_MW', 'TrafoOut_MW', 'TrafoTapPos', 'Static_gen']).T
        if self.feeder_pq:
            self.p_feeder = pd.DataFrame(self.p_feeder, index=self.feeder).T
            self.q_feeder = pd.DataFrame(self.q_feeder, index=self.feeder).T
        
    def save_results(self, outputfolder=None):
        if outputfolder is None:
            of = self.outputfolder
        else:
            of = outputfolder
        util.create_folder(of)
        self.res_v.to_csv(of + r'/vm_pu.csv')
        self.global_res.to_csv(of + r'/global_res.csv')
        
        if self.feeder_pq:
            self.p_feeder.to_csv(of + r'/p_feeder.csv')
            self.q_feeder.to_csv(of + r'/q_feeder.csv')
        

class Iterator():
    def __init__(self, net,  profiler=None, ow=None, feeder_pq=False):
        self.net = net
        if ow is None:
            self.ow = output_writer(net, feeder_pq=feeder_pq)
        else:
            self.ow = ow
        if profiler is None:
            self.profiler = Profiler(net=net)
        elif not profiler.net == net:
            print('iterator and profiler dont have same pp.net. Check data')
            return
        self.profiler = profiler
    
    def iterate(self, time_steps, save=True, outputfolder='', ultraverbose=False):
        if not self.profiler.check_time_steps(tss=time_steps):
            return
        li =  1
        print('Starting to iterate')
        t = [time.time()]
        dt = 10 if len(time_steps)<1000 else 20
        for i, ts in enumerate(time_steps):          
            if ultraverbose:
                print(i)
            if (i % (len(time_steps)/dt)) < li:
                t.append(time.time())
                print('\tComputing: {:2d}% done. Elapsed time {:2d}h{:02d}:{:02.2f} '.format(int(i//(len(time_steps)/100)), *util.sec_to_time(t[-1]-t[0])))
            li = i % (len(time_steps)/dt)
            self.profiler.update(ts)
            pp.runpp(self.net, run_control=True)
            self.ow.get_results(ts)
        self.ow.consolidate()
        print('\tDone!')
        if save:
            print('Saving Results')
            self.ow.save_results(outputfolder=outputfolder)
            print('\tDone!')
            