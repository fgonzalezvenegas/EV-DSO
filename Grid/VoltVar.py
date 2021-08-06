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
                 vmin=0.96, vmax=1.045, deadband=0.03, 
                 qmin_pu=-0.4, qmax_pu=+0.5, in_service=True,
                 tolerance=0.01, alpha=0.7,
                 recycle=False, order=0, level=0, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                    initial_powerflow = True, **kwargs)
        # Note: we set default values according to Enedis standards
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
        self.alpha = alpha

        # profile attributes
        self.vmin = vmin                            # V at which Q injection is max
        self.vmax = vmax                            # V at which Q absorption is maximal
        self.qmin = qmin_pu                         # Min Q (absorption, at Vmax) 
        self.qmax = qmax_pu                         # Max Q injection (at vmin)
        self.deadband = deadband                    # Symmetric deadband between 1+-deadband
        self.m_vmin = qmax_pu / (vmin-(1-deadband)) # Slope of Q gain for undervoltages
        self.m_vmax = qmin_pu / (vmax-(1+deadband)) # Slope of Q gain for overvoltages
        
        self.tolerance = tolerance # in MW
        
        
    def get_q_v(self,v=None):
        """ Returns the Q(V), given by the sections [vmin, 1-deadband], [1+- deadband], [1+deadband,vmax]
        Q(V,P)
        Q(V) in pu
        """
#        if v is None:
        v = self.net.res_bus.at[self.bus, 'vm_pu']
#        p = self.net.res_sgen.at[self.gid, 'p_mw']
        if abs(v-1) <= self.deadband:
            return 0
        if v <= 1-self.deadband:
            return min(self.qmax, (v-(1-self.deadband)) * self.m_vmin)
        else:
            return max(self.qmin, (v-(1+self.deadband)) * self.m_vmax)
    
     # In case the controller is not yet converged, the control step is executed. In the example it simply
    # adopts a new value according to the previously calculated target and writes back to the net.
    def control_step(self):
        # Call write_to_net and set the applied variable True
        q_out = self.get_q_v() * self.sn_mva
        q0 = self.net.sgen.at[self.gid, 'q_mvar']
        if self.net.res_bus.at[self.bus, 'vm_pu'] > self.vmax+0.01:
            self.net.sgen.at[self.gid, 'q_mvar'] = q_out
        else:
            # write q
            self.net.sgen.at[self.gid, 'q_mvar'] = (q_out * self.alpha) + (q0*(1-self.alpha))
    
    def is_converged(self):
        q_exp = self.get_q_v() * self.sn_mva
        q_control = self.net.sgen.at[self.gid, 'q_mvar']
#        print('q_exp', q_exp)
#        print('q_control', q_control)
        return abs(q_exp - q_control) < self.tolerance
    
#    def plot_law(self):
#        import matplotlib.pyplot as plt
#        x = np.arange(self.vmin-0.01, self.vmax+0.01, 0.001)
#        l = [self.get_q_v(v=v) for v in x]
#        plt.plot(x,l)
#        plt.axvline(self.vmin,color='k',linestyle='--', linewidth=1)
#        plt.axvline(self.vmax,color='k',linestyle='--', linewidth=1)
#        plt.axvline(1-self.deadband,color='k',linestyle='--', linewidth=1)
#        plt.axvline(1+self.deadband,color='k',linestyle='--', linewidth=1)
#        
#        plt.xlabel('Voltage [pu]')
#        plt.ylabel('Q [pu]')
#        plt.tight_layout()
#        plt.gcf().set_size_inches(3.5,3)
    
#%%    
#import matplotlib.pyplot as plt
##        
#def get_q_v(v, deadband, qmax, qmin):
#        """ Returns the Q(V), given by the sections [vmin, 1-deadband], [1+- deadband], [1+deadband,vmax]
#        Q(V,P)
#        Q(V) in pu
#        """
#        m_vmin = qmax / (vmin-(1-deadband))
#        m_vmax = qmin / (vmax-(1+deadband))
#        if abs(v-1) <= deadband:
#            return 0
#        if v <= 1-deadband:
#            return min(qmax, (v-(1-deadband)) * m_vmin)
#        else:
#            return max(qmin, (v-(1+deadband)) * m_vmax)
#
## Plot regulation law:
#vmin=0.96
#vmax=1.045
#deadband=0.03
#qmin=-0.4
#qmax=+0.5
#x = np.arange(.94, 1.06,0.001)
#
#
#l = [get_q_v(v, deadband, qmax, qmin) for v in x]  
#plt.plot(x,l)
#plt.axvline(vmin,color='grey',linestyle='--', linewidth=1)
#plt.axvline(vmax,color='grey',linestyle='--', linewidth=1)
#plt.axvline(1-deadband,color='grey',linestyle='--', linewidth=1)
#plt.axvline(1+deadband,color='grey',linestyle='--', linewidth=1)
#
#plt.xlabel('Voltage [pu]')
#plt.ylabel('Q [pu]')
#
#
#plt.gcf().set_size_inches(3.5,3)    
#plt.tight_layout()
#plt.savefig(r'c:\user\U546416\Pictures\pandapower\voltvar.pdf')
#plt.savefig(r'c:\user\U546416\Pictures\pandapower\voltvar.png',dpi=300)

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
