# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:33:39 2020

@author: U546416
"""

import pandapower as pp
#import control
import pandas as pd
import pandapower.timeseries as ts
import pandapower.control as ppc
import numpy as np

class VoltVar(ppc.basic_controller.Controller):
    """ Volt var regulator for static generator.
    It updates the sgen Q given the voltage at local bus, following a stepwise curve
    Q(Vmin) = qmax_pu
    Q(Vmax) = qmin_pu
    Q(1-deadband<V<1+deadband) = 0
    """
    def __init__(self, net, gid, 
                 vmin=0.95, vmax=1.05, deadband=0.02, 
                 qmin_pu=-0.4, qmax_pu=+0.4, in_service=True,
                 tolerance=0.01,
                 recycle=False, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                    initial_powerflow = True, **kwargs)
        
        # read generator attributes from net
        self.gid = gid  # index of the controlled generator
        self.bus = net.sgen.at[gid, "bus"]
#        self.p_mw = net.sgen.at[gid, "p_mw"]
#        self.q_mvar = net.sgen.at[gid, "q_mvar"]
        self.scaling = net.sgen.at[gid, "scaling"]
        self.sn_mva = net.sgen.at[gid, "sn_mva"]
        if not (self.sn_mva ==  self.sn_mva):
            self.sn_mva = net.sgen.at[gid, "p_mw"]
        self.name = net.sgen.at[gid, "name"]
        self.gen_type = net.sgen.at[gid, "type"]
        self.in_service = net.sgen.at[gid, "in_service"]
        self.applied = False

        # profile attributes
        self.vmin = vmin
        self.vmax = vmax
        self.qmin = qmin_pu
        self.qmax = qmax_pu
        self.deadband = deadband
        self.m_vmin = qmax_pu / (vmin-(1-deadband)) 
        self.m_vmax = qmin_pu / (vmax-(1+deadband)) 
        
        self.tolerance = tolerance # in kW
        
        
    def get_q_v(self):
        """ Returns the Q(V), given by the sections [vmin, 1-deadband], [1+- deadband], [1+deadband,vmax]
        Q(V,P)
        Q(V) in pu
        """
        v = self.net.res_bus.at[self.bus, 'vm_pu']
#        p = self.net.res_sgen.at[self.gid, 'p_mw']
        if abs(v-1) <= self.deadband:
            return 0
        if v < 1-self.deadband:
            return min(self.qmax, (v-(1-self.deadband)) * self.m_vmin)
        else:
            return max(self.qmin, (v-(1+self.deadband)) * self.m_vmax)
    
     # In case the controller is not yet converged, the control step is executed. In the example it simply
    # adopts a new value according to the previously calculated target and writes back to the net.
    def control_step(self):
        # Call write_to_net and set the applied variable True
        q_out = self.get_q_v() * self.sn_mva
        # write q
        self.net.sgen.at[self.gid, 'q_mvar'] = q_out
    
    def is_converged(self):
        q_exp = self.get_q_v() * self.sn_mva
        q_control = self.net.sgen.at[self.gid, 'q_mvar']
#        print('q_exp', q_exp)
#        print('q_control', q_control)
        return abs(q_exp - q_control) < self.tolerance
    
#%%    
#import matplotlib.pyplot as plt
#
#net = pp.create_empty_network()
#pp.create_bus(net, 20)
#pp.create_bus(net, 20)
#pp.create_line(net, 0, 1, 20, std_type='NAYY 4x50 SE')
##pp.create_load(net, 1, 1)
#pp.create_ext_grid(net, 0)
#
#vs = np.arange(0.9,1.1,0.001)
##v0 = []
##for l in vs:
##    net.ext_grid.vm_pu = l
##    pp.runpp(net)
##    v0.append(net.res_bus.vm_pu[1])
##
##v1 = []
##
##for l in loads:
##    net.load.p_mw = l
##    pp.runpp(net)
##    v1.append(net.res_bus.vm_pu[1])
##    
#v2 = []
#q2 = []
#q3 = []
#gen = pp.create_sgen(net, 1, 0, sn_mva=2, scaling=1.5)
#gen2 = pp.create_sgen(net, 1, 0, sn_mva=2, scaling=1.5)
#voltvar =  VoltVar(net, gen, vmin=0.95, vmax=1.05)
#voltvar =  VoltVar(net, gen2, vmin=0.95, vmax=1.05)
#for l in vs:
#    net.ext_grid.vm_pu = l
#    pp.runpp(net, run_control=True)
#    v2.append(net.res_bus.vm_pu[1])
#    q2.append(net.res_sgen.q_mvar[0])
#    q3.append(net.res_sgen.q_mvar[1])
#
#plt.figure()
#plt.plot(v2, q2, label='pv2')
#plt.plot(v2, q2, label='pv1')
#plt.xticks(np.arange(0.9,1.11,0.01))
#plt.grid()
#plt.title('Qout')
#plt.figure()
#plt.plot(vs, v2, label='with gen+voltvar')
#plt.title('V/v')
#plt.legend()
#
##%%
#
#net2 = pp.create_empty_network()
#pp.create_bus(net2, 20)
#pp.create_ext_grid(net2, 0)
#gen = pp.create_sgen(net2, 0, 1)
#voltvar =  VoltVar(net2, gen)
#
#rangev = np.arange(0.9,1.1,0.01)
#q = []
#for v in rangev:
#    net2.ext_grid.vm_pu = v
#    pp.runpp(net2, run_control=True)
#    q.append(net2.res_sgen.q_mvar[0])
#    
#plt.figure()
#plt.plot(rangev, q)
