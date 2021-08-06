# -*- coding: utf-8 -*-
"""
Created on Mon May  3 03:03:36 2021

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


nevs =20
pch = 7
th = 0.8

x = np.arange(16,24,0.05)
prof = stats.norm(loc=17, scale=1).pdf(x) * nevs * pch

prof1 = pd.Series(prof, index=x)
prof2 = prof1.copy(deep=True)

baseline = prof1[17:20].mean()

prof1.loc[18:19] = -nevs * pch

prof2.loc[18:19] = -nevs * pch*0.5

obj = -20*7
thr = baseline + (obj-baseline) * th

f,axs = plt.subplots(1,2, sharey=True)

plt.sca(axs[0])
plt.plot(x, prof1, 'k', label='Realisation', zorder=9)
plt.axhline(baseline, color='orange', label='Baseline', linestyle='--')
plt.axhline(thr, color='r', label='Threshold', linestyle='--')
plt.axhline(obj, color='g', label='Objective', linestyle='--')
plt.axhline(0, color='grey', label='_', linestyle='--', linewidth=0.5, alpha=0.8)
plt.title('(a) Successful', y=-0.28)
plt.xlim(17,20)
plt.ylim(-150,75)
plt.xticks((17,18,19,20))
plt.yticks(np.arange(-200,200,50))
plt.xlabel('Time')
plt.ylabel('Power [kW]')

plt.sca(axs[1])
plt.plot(x, prof2, 'k', label='_', zorder=9)
plt.axhline(baseline, color='orange', label='_', linestyle='--')
plt.axhline(obj, color='g', label='_', linestyle='--')
plt.axhline(thr, color='r', label='_', linestyle='--')
plt.axhline(0, color='grey', label='_', linestyle='--', linewidth=0.5, alpha=0.8)
plt.xlim(17,20)
plt.xticks((17,18,19,20))
plt.yticks(np.arange(-200,200,50))
plt.ylim(-150,75)
plt.xlabel('Time')
plt.title('(b) Not successful', y=-0.28)


f.set_size_inches(7.57,3)
plt.tight_layout()
f.legend(loc=9, ncol=5)
for ax in axs:
    pos = ax.get_position()
    dy= 0.05
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height-dy])
    
plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\activations.jpg', dpi=300)

plt.savefig(r'c:\user\U546416\Pictures\FlexTenders\activations.pdf')
#%%