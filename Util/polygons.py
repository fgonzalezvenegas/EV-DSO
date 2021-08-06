# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:40:00 2019

@author: U546416
"""

import numpy as np
import matplotlib.pyplot as plt

class Polygons:
    def __init__(self, corners, types='Polygon'):
        self.corners = corners
        
    def get_x(self):
        return [c[0] for c in self.corners]

    def get_y(self):
        return [c[1] for c in self.corners]        

    def contains(self, point):
        """
        point -- x and y coordinates of point
        """
        
        num = len(self.corners)
        i = 0
        j = num - 1
        c = False
        for i in range(num):
            if ((self.corners[i][1] > point[1]) != (self.corners[j][1] > point[1])) and \
                    (point[0] < self.corners[i][0] + (self.corners[j][0] - self.corners[i][0]) * (point[1] - self.corners[i][1]) /
                                      (self.corners[j][1] - self.corners[i][1])):
                c = not c
            j = i
        return c
    
    def get_center(self):
        return [np.mean(self.get_x()), np.mean(self.get_y())]
    
    def plot(self, ax='', **kwargs):
        if ax=='':
            f, ax = plt.subplots()
        x = self.get_x()
        y = self.get_y()
        ax.plot(x + [x[0]], y + [y[0]], **kwargs)
        return ax
    
class multiPolygons:
    def __init__(self, multicorners):
        self.polygons = []
        for mc in multicorners:
             self.polygons.append(Polygons(mc[0]))
    
    def contains(self, point):
        inside = [p.contains(point) for p in self.polygons]
        return sum(inside) > 0
    
    def get_center(self):
        centers = [p.get_center() for p in self.polygons]
        return [np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers])]
    
    def plot(self, ax=''):
        if ax=='':
            f, ax = plt.subplots()
        for p in self.polygons:
            p.plot(ax)
    